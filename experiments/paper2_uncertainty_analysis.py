"""Final Paper 2 analysis on the saved lesion-grouped HAM10000 MC predictions.

Primary design:
- lesion-grouped split generated with seed 2026 (zero lesion overlap)
- checkpoint selected on validation macro-F1 only
- T=30 MC Dropout passes
- uncertainty threshold selected on VALIDATION at target 60% coverage and frozen
- threshold then applied once to the held-out TEST set
- comparative uncertainty scores: 1-MSP, predictive entropy, expected entropy,
  mutual information, variation ratio
- calibration, error detection, risk-coverage/AURC, class-specific referral
- bootstrap 95% CIs conditional on the held-out prediction set
- deterministic-inference ablation of the exact same dropout-trained checkpoint
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    classification_report, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

EPS = 1e-12
TARGET_COVERAGE = 0.60
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 20260810
NUM_CLASSES = 7
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

BASE = Path('/kaggle/working/paper2_data')
OUT = BASE / 'grouped_mc_seed2026'
NPZ_PATH = OUT / 'paper2_grouped_mc_predictions.npz'
CKPT_PATH = OUT / 'model_best_val_macro_f1.pt'
SPLIT_CSV = BASE / 'splits' / 'ham10000_lesion_group_split_seed2026.csv'
IMG_DIR = BASE / 'HAM10000_images'


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, EPS, 1.0)
    return x / x.sum(axis=-1, keepdims=True)


def uncertainty_scores(mc_probs):
    p = normalize_probs(mc_probs)
    mean_p = p.mean(axis=1)
    predictive_entropy = -np.sum(mean_p * np.log(mean_p), axis=1)
    expected_entropy = -np.mean(np.sum(p * np.log(p), axis=2), axis=1)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    one_minus_msp = 1.0 - np.max(mean_p, axis=1)
    votes = np.argmax(p, axis=2)
    variation_ratio = np.empty(len(votes), dtype=np.float64)
    for i, row in enumerate(votes):
        counts = np.bincount(row, minlength=mean_p.shape[1])
        variation_ratio[i] = 1.0 - counts.max() / float(len(row))
    return mean_p, {
        'one_minus_msp': one_minus_msp,
        'predictive_entropy': predictive_entropy,
        'expected_entropy': expected_entropy,
        'mutual_information': mutual_information,
        'variation_ratio': variation_ratio,
    }


def multiclass_nll(mean_p, labels):
    return float(-np.mean(np.log(np.clip(mean_p[np.arange(len(labels)), labels], EPS, 1.0))))


def multiclass_brier(mean_p, labels):
    y = np.zeros_like(mean_p)
    y[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((mean_p - y) ** 2, axis=1)))


def ece_top1(mean_p, labels, n_bins=15):
    conf = mean_p.max(axis=1)
    pred = mean_p.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b+1]
        mask = (conf >= lo) & ((conf <= hi) if b == n_bins-1 else (conf < hi))
        if np.any(mask):
            acc = float(correct[mask].mean())
            cf = float(conf[mask].mean())
            n = int(mask.sum())
            ece += n / len(labels) * abs(acc - cf)
            rows.append({'lo': float(lo), 'hi': float(hi), 'n': n, 'accuracy': acc, 'confidence': cf})
    return float(ece), rows


def error_metrics(labels, mean_p, score):
    pred = mean_p.argmax(axis=1)
    err = (pred != labels).astype(int)
    return {
        'auroc': float(roc_auc_score(err, score)),
        'auprc': float(average_precision_score(err, score)),
    }


def risk_coverage(score, correct):
    order = np.argsort(score, kind='mergesort')
    c = correct[order].astype(float)
    k = np.arange(1, len(c)+1)
    cov = k / len(c)
    risk = 1.0 - np.cumsum(c) / k
    aurc = float(np.trapezoid(np.r_[risk[0], risk], np.r_[0.0, cov]))
    points = []
    for target in np.arange(0.1, 1.01, 0.1):
        idx = int(np.argmin(np.abs(cov-target)))
        points.append({'coverage': float(cov[idx]), 'risk': float(risk[idx])})
    return aurc, points


def validation_threshold(score, target_coverage=0.60):
    # Select a frozen scalar threshold from validation only. The order-statistic
    # threshold targets the requested coverage without looking at test labels/scores.
    s = np.sort(np.asarray(score, dtype=float), kind='mergesort')
    k = max(1, min(len(s), int(round(target_coverage * len(s)))))
    return float(s[k-1])


def selective_metrics(mean_p, labels, score, threshold):
    pred = mean_p.argmax(axis=1)
    keep = np.asarray(score) <= threshold
    retained_n = int(keep.sum())
    if retained_n == 0:
        raise RuntimeError('Validation-derived threshold retained zero test samples')
    y = labels[keep]
    p = pred[keep]
    report = classification_report(
        y, p, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    class_rows = {}
    for i, name in enumerate(CLASS_NAMES):
        total_n = int(np.sum(labels == i))
        retained_cls_n = int(np.sum(y == i))
        referred_n = total_n - retained_cls_n
        tp = int(np.sum((y == i) & (p == i)))
        fn = retained_cls_n - tp
        class_rows[name] = {
            'total_n': total_n,
            'retained_n': retained_cls_n,
            'referred_n': referred_n,
            'class_coverage': float(retained_cls_n/total_n) if total_n else None,
            'tp': tp,
            'fn': fn,
            'selective_sensitivity': float(report[name]['recall']) if retained_cls_n else None,
            'precision': float(report[name]['precision']) if retained_cls_n else None,
            'f1': float(report[name]['f1-score']) if retained_cls_n else None,
        }
    return {
        'threshold': float(threshold),
        'retained_n': retained_n,
        'referred_n': int(len(labels)-retained_n),
        'coverage': float(retained_n/len(labels)),
        'referral_rate': float(1-retained_n/len(labels)),
        'accuracy': float(accuracy_score(y,p)),
        'macro_f1': float(f1_score(y,p,average='macro',zero_division=0)),
        'weighted_f1': float(f1_score(y,p,average='weighted',zero_division=0)),
        'class_metrics': class_rows,
    }


def bootstrap_ci(labels, mean_p, score, threshold, b=1000, seed=20260810):
    rng = np.random.default_rng(seed)
    n = len(labels)
    pred = mean_p.argmax(axis=1)
    vals = {'accuracy': [], 'macro_f1': [], 'melanoma_recall': [], 'coverage': [], 'selective_accuracy': [], 'selective_macro_f1': [], 'selective_melanoma_recall': []}
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]; p = pred[idx]; s = score[idx]
        vals['accuracy'].append(accuracy_score(y,p))
        vals['macro_f1'].append(f1_score(y,p,average='macro',zero_division=0))
        mel = y == CLASS_TO_IDX['mel']
        vals['melanoma_recall'].append(float(np.mean(p[mel] == CLASS_TO_IDX['mel'])) if np.any(mel) else np.nan)
        keep = s <= threshold
        vals['coverage'].append(float(np.mean(keep)))
        if np.any(keep):
            vals['selective_accuracy'].append(accuracy_score(y[keep],p[keep]))
            vals['selective_macro_f1'].append(f1_score(y[keep],p[keep],average='macro',zero_division=0))
            mel2 = y[keep] == CLASS_TO_IDX['mel']
            vals['selective_melanoma_recall'].append(float(np.mean(p[keep][mel2] == CLASS_TO_IDX['mel'])) if np.any(mel2) else np.nan)
        else:
            vals['selective_accuracy'].append(np.nan); vals['selective_macro_f1'].append(np.nan); vals['selective_melanoma_recall'].append(np.nan)
    out = {}
    for k, arr in vals.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[k] = [float(np.quantile(arr,0.025)), float(np.quantile(arr,0.975))]
    return out


class HAMDataset(Dataset):
    def __init__(self, frame, transform):
        self.df = frame.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(IMG_DIR / f"{row['image_id']}.jpg") as im:
            x = self.transform(im.convert('RGB'))
        return x, CLASS_TO_IDX[row['dx']]


class MCDropoutResNet50(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        base = models.resnet50(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc1 = nn.Linear(in_features,512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(512,NUM_CLASSES)
    def forward(self,x):
        x=self.backbone(x); x=self.fc1(x); x=self.relu(x); x=self.dropout(x); return self.fc2(x)


@torch.no_grad()
def deterministic_same_model():
    if not (CKPT_PATH.exists() and SPLIT_CSV.exists()):
        return {'available': False}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    df = pd.read_csv(SPLIT_CSV)
    test = df[df['split']=='test'].copy()
    loader = DataLoader(HAMDataset(test,tfm),batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    model = MCDropoutResNet50(0.2).to(device)
    ckpt = torch.load(CKPT_PATH,map_location=device,weights_only=False)
    model.load_state_dict(ckpt['model_state']); model.eval()
    ys=[]; probs=[]
    for x,y in loader:
        x=x.to(device,non_blocking=True)
        with torch.amp.autocast('cuda',enabled=(device.type=='cuda')):
            logits=model(x)
        probs.append(torch.softmax(logits.float(),dim=1).cpu().numpy()); ys.extend(y.numpy().tolist())
    mean_p=np.concatenate(probs); labels=np.asarray(ys,dtype=int); pred=mean_p.argmax(1)
    rep=classification_report(labels,pred,labels=list(range(NUM_CLASSES)),target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    ece,_=ece_top1(mean_p,labels,15)
    return {
        'available': True,
        'accuracy': float(accuracy_score(labels,pred)),
        'macro_f1': float(f1_score(labels,pred,average='macro',zero_division=0)),
        'weighted_f1': float(f1_score(labels,pred,average='weighted',zero_division=0)),
        'melanoma_recall': float(rep['mel']['recall']),
        'nll': multiclass_nll(mean_p,labels),
        'brier': multiclass_brier(mean_p,labels),
        'ece15': ece,
    }


def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)
    d=np.load(NPZ_PATH,allow_pickle=False)
    val_mc=d['val_mc_probs']; val_y=d['val_labels'].astype(int)
    test_mc=d['id_mc_probs']; test_y=d['id_labels'].astype(int)
    val_mean,val_scores=uncertainty_scores(val_mc)
    test_mean,test_scores=uncertainty_scores(test_mc)
    test_pred=test_mean.argmax(1); test_correct=test_pred==test_y
    test_rep=classification_report(test_y,test_pred,labels=list(range(NUM_CLASSES)),target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    ece,rel=ece_top1(test_mean,test_y,15)
    result={
        'design': {
            'split':'lesion-grouped, zero lesion overlap', 'split_seed':2026,
            'train_n':8012,'validation_n':1011,'test_n':992,
            'mc_passes':30,'dropout_p':0.2,'threshold_selected_on':'validation only','target_validation_coverage':TARGET_COVERAGE,
            'bootstrap_replicates':BOOTSTRAP_B,
        },
        'mc_full_test': {
            'accuracy':float(accuracy_score(test_y,test_pred)),
            'macro_f1':float(f1_score(test_y,test_pred,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(test_y,test_pred,average='weighted',zero_division=0)),
            'melanoma_recall':float(test_rep['mel']['recall']),
            'nll':multiclass_nll(test_mean,test_y),
            'brier':multiclass_brier(test_mean,test_y),
            'ece15':ece,
            'reliability_bins':rel,
        },
        'deterministic_same_dropout_trained_checkpoint': deterministic_same_model(),
        'uncertainty_methods':{},
    }
    for name in ['one_minus_msp','predictive_entropy','expected_entropy','mutual_information','variation_ratio']:
        threshold=validation_threshold(val_scores[name],TARGET_COVERAGE)
        sel=selective_metrics(test_mean,test_y,test_scores[name],threshold)
        aurc,points=risk_coverage(test_scores[name],test_correct)
        method={
            'validation_threshold':threshold,
            'validation_actual_coverage':float(np.mean(val_scores[name] <= threshold)),
            'error_detection':error_metrics(test_y,test_mean,test_scores[name]),
            'aurc':aurc,
            'risk_coverage_points':points,
            'test_selective':sel,
        }
        method['bootstrap_95ci']=bootstrap_ci(test_y,test_mean,test_scores[name],threshold,BOOTSTRAP_B,BOOTSTRAP_SEED)
        result['uncertainty_methods'][name]=method
    print('PAPER2_FINAL_ANALYSIS_JSON_BEGIN')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print('PAPER2_FINAL_ANALYSIS_JSON_END')

main()
