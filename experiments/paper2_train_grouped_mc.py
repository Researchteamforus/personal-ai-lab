"""Train a leakage-safe, lesion-grouped MC Dropout ResNet50 for Paper 2.

This is a NEW robustness experiment, not a reconstruction of the manuscript's original
image-level split. It uses the original HAM10000 images/metadata already prepared under
/kaggle/working/paper2_data and the lesion-group split generated with seed 2026.

Protocol aligned to the manuscript where specified:
- ResNet50 ImageNet initialization
- 224x224, ImageNet normalization
- Dense 512 + Dropout(p=0.2) + 7-way classifier
- Adam, lr=1e-4, class-weighted cross entropy
- batch size 32, 30 epochs
- T=30 stochastic MC Dropout passes

Additional defensibility choice: select the checkpoint by validation macro-F1, never by test data.
"""

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

SEED = 2026
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
DROPOUT_P = 0.2
MC_PASSES = 30
NUM_CLASSES = 7
NUM_WORKERS = 4

BASE = Path('/kaggle/working/paper2_data')
IMG_DIR = BASE / 'HAM10000_images'
SPLIT_CSV = BASE / 'splits' / 'ham10000_lesion_group_split_seed2026.csv'
OUT = BASE / 'grouped_mc_seed2026'
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class HAMDataset(Dataset):
    def __init__(self, frame, transform):
        self.df = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = IMG_DIR / f"{row['image_id']}.jpg"
        with Image.open(path) as im:
            image = im.convert('RGB')
        image = self.transform(image)
        label = CLASS_TO_IDX[row['dx']]
        return image, label, row['image_id']


class MCDropoutResNet50(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            base = models.resnet50(weights=weights)
            print('IMAGENET_WEIGHTS|IMAGENET1K_V2')
        except Exception as exc:
            print('IMAGENET_V2_ERROR|', repr(exc))
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
            print('IMAGENET_WEIGHTS|DEFAULT')
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc1 = nn.Linear(in_features, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def enable_mc_dropout(model):
    # Keep BatchNorm in eval mode; activate Dropout only.
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def make_loaders(df):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    frames = {s: df[df['split'] == s].copy() for s in ['train', 'validation', 'test']}
    loaders = {
        'train': DataLoader(
            HAMDataset(frames['train'], tfm), batch_size=BATCH_SIZE,
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True,
        ),
        'validation': DataLoader(
            HAMDataset(frames['validation'], tfm), batch_size=BATCH_SIZE,
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True,
        ),
        'test': DataLoader(
            HAMDataset(frames['test'], tfm), batch_size=BATCH_SIZE,
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True,
        ),
    }
    return frames, loaders


def class_weights(train_df, device):
    counts = train_df['dx'].value_counts().reindex(CLASS_NAMES).values.astype(np.float64)
    n = counts.sum()
    w = n / (NUM_CLASSES * counts)
    print('TRAIN_CLASS_COUNTS|', dict(zip(CLASS_NAMES, counts.astype(int).tolist())))
    print('CLASS_WEIGHTS|', dict(zip(CLASS_NAMES, [float(x) for x in w])))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def deterministic_eval(model, loader, device, criterion):
    model.eval()
    losses = []
    ys, preds = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(float(loss.item()) * len(y))
        ys.extend(y.cpu().numpy().tolist())
        preds.extend(logits.argmax(1).cpu().numpy().tolist())
    n = len(ys)
    return {
        'loss': float(sum(losses) / n),
        'accuracy': float(accuracy_score(ys, preds)),
        'macro_f1': float(f1_score(ys, preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(ys, preds, average='weighted', zero_division=0)),
    }


def train_model(model, loaders, frames, device):
    cw = class_weights(frames['train'], device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_f1 = -1.0
    best_epoch = -1
    history = []
    start_all = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        y_true, y_pred = [], []
        t0 = time.perf_counter()

        for x, y, _ in loaders['train']:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * len(y)
            seen += len(y)
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(logits.detach().argmax(1).cpu().numpy().tolist())

        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        val = deterministic_eval(model, loaders['validation'], device, criterion)
        elapsed = time.perf_counter() - t0

        row = {
            'epoch': epoch,
            'train_loss': running_loss / seen,
            'train_accuracy': train_acc,
            'train_macro_f1': train_f1,
            'val_loss': val['loss'],
            'val_accuracy': val['accuracy'],
            'val_macro_f1': val['macro_f1'],
            'val_weighted_f1': val['weighted_f1'],
            'epoch_seconds': elapsed,
        }
        history.append(row)
        print('EPOCH_JSON|', json.dumps(row, sort_keys=True))

        torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'row': row}, OUT / 'model_final.pt')
        if val['macro_f1'] > best_val_f1:
            best_val_f1 = val['macro_f1']
            best_epoch = epoch
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'row': row}, OUT / 'model_best_val_macro_f1.pt')
            print(f'BEST_CHECKPOINT|epoch={epoch}|val_macro_f1={best_val_f1:.6f}')

    pd.DataFrame(history).to_csv(OUT / 'training_history.csv', index=False)
    print(f'TRAINING_DONE|best_epoch={best_epoch}|best_val_macro_f1={best_val_f1:.6f}|seconds={time.perf_counter()-start_all:.2f}')
    return criterion


@torch.no_grad()
def mc_predict(model, loader, device, t=30):
    enable_mc_dropout(model)
    all_passes = []
    labels_ref = None
    ids_ref = None
    t0 = time.perf_counter()

    for pass_id in range(t):
        probs_list, labels_list, ids_list = [], [], []
        for x, y, image_ids in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x)
            probs_list.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels_list.extend(y.numpy().tolist())
            ids_list.extend(list(image_ids))
        pass_probs = np.concatenate(probs_list, axis=0)
        all_passes.append(pass_probs)
        if labels_ref is None:
            labels_ref = np.asarray(labels_list, dtype=np.int64)
            ids_ref = np.asarray(ids_list, dtype='U20')
        else:
            assert np.array_equal(labels_ref, np.asarray(labels_list, dtype=np.int64))
            assert np.array_equal(ids_ref, np.asarray(ids_list, dtype='U20'))
        print(f'MC_PASS_DONE|{pass_id+1}/{t}')

    # [T,N,C] -> [N,T,C]
    mc = np.stack(all_passes, axis=0).transpose(1, 0, 2)
    elapsed = time.perf_counter() - t0
    print(f'MC_PREDICT_DONE|N={len(labels_ref)}|T={t}|seconds={elapsed:.2f}|ms_per_image_pass={(elapsed*1000)/(len(labels_ref)*t):.4f}')
    return mc, labels_ref, ids_ref


def summarize_mc(mc, labels, split_name):
    mean_p = mc.mean(axis=1)
    pred = mean_p.argmax(axis=1)
    report = {
        'split': split_name,
        'n': int(len(labels)),
        'accuracy': float(accuracy_score(labels, pred)),
        'macro_f1': float(f1_score(labels, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(labels, pred, average='weighted', zero_division=0)),
        'per_class_recall': {},
    }
    r = classification_report(labels, pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    for c in CLASS_NAMES:
        report['per_class_recall'][c] = float(r[c]['recall'])
    print('MC_SUMMARY_JSON|', json.dumps(report, sort_keys=True))
    return report


def main():
    seed_all(SEED)
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f'Missing split CSV: {SPLIT_CSV}')
    if len(list(IMG_DIR.glob('*.jpg'))) < 10015:
        raise RuntimeError(f'HAM10000 images not prepared in {IMG_DIR}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE|', device)
    if device.type == 'cuda':
        print('GPU|', torch.cuda.get_device_name(0))
        print('GPU_VRAM_GB|', torch.cuda.get_device_properties(0).total_memory / 1024**3)

    df = pd.read_csv(SPLIT_CSV)
    print('SPLIT_COUNTS|', df['split'].value_counts().to_dict())
    frames, loaders = make_loaders(df)

    model = MCDropoutResNet50(DROPOUT_P).to(device)
    criterion = train_model(model, loaders, frames, device)

    best = torch.load(OUT / 'model_best_val_macro_f1.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model_state'])
    print('LOADED_BEST|', best['epoch'], best['row'])

    val_mc, val_y, val_ids = mc_predict(model, loaders['validation'], device, MC_PASSES)
    test_mc, test_y, test_ids = mc_predict(model, loaders['test'], device, MC_PASSES)
    val_summary = summarize_mc(val_mc, val_y, 'validation')
    test_summary = summarize_mc(test_mc, test_y, 'test')

    np.savez_compressed(
        OUT / 'paper2_grouped_mc_predictions.npz',
        val_mc_probs=val_mc.astype(np.float32),
        val_labels=val_y,
        val_image_ids=val_ids,
        id_mc_probs=test_mc.astype(np.float32),
        id_labels=test_y,
        id_image_ids=test_ids,
        class_names=np.asarray(CLASS_NAMES, dtype='U10'),
    )
    with open(OUT / 'mc_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'best_epoch': int(best['epoch']), 'validation': val_summary, 'test': test_summary}, f, indent=2)

    print('PREDICTIONS_NPZ|', OUT / 'paper2_grouped_mc_predictions.npz')
    print('GROUPED_MC_TRAINING_PIPELINE_DONE')


main()
