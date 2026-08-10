"""Four additional training seeds for Paper 2 on the fixed lesion-grouped split.

The existing seed 2026 run is retained as seed 1. This runner is parameterized by
TRAIN_SEEDS injected by github_dispatch.py, so Node2 and Node10 can execute two
additional seeds each in parallel. The fixed split is never regenerated.

For every seed: train ResNet50 + Dense512 + Dropout(0.2) for 30 epochs, select the
checkpoint by validation macro-F1 only, run T=30 MC Dropout on validation/test, and
report classification, calibration, MI error detection/AURC, and a validation-frozen
60% MI referral operating point. Full MC predictions are saved locally for later use.
"""

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

TRAIN_SEEDS = globals().get('TRAIN_SEEDS', [11])
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
DROPOUT_P = 0.2
MC_PASSES = 30
NUM_WORKERS = 4
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_NAMES)}
MEL = CLASS_TO_IDX['mel']
EPS = 1e-12
BASE = Path('/kaggle/working/paper2_data')
IMG_DIR = BASE / 'HAM10000_images'
SPLIT_CSV = BASE / 'splits' / 'ham10000_lesion_group_split_seed2026.csv'
MULTI_BASE = BASE / 'multiseed_fixed_split'
MULTI_BASE.mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class HAMDataset(Dataset):
    def __init__(self, frame, tfm): self.df=frame.reset_index(drop=True); self.tfm=tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]
        with Image.open(IMG_DIR / f"{r['image_id']}.jpg") as im: x=self.tfm(im.convert('RGB'))
        return x, CLASS_TO_IDX[r['dx']], r['image_id']


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        base=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        d=base.fc.in_features; base.fc=nn.Identity(); self.backbone=base
        self.fc1=nn.Linear(d,512); self.relu=nn.ReLU(inplace=True); self.dropout=nn.Dropout(DROPOUT_P); self.fc2=nn.Linear(512,7)
    def forward(self,x): return self.fc2(self.dropout(self.relu(self.fc1(self.backbone(x)))))


def loaders_for(df, seed):
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tr=df[df.split=='train'].copy(); va=df[df.split=='validation'].copy(); te=df[df.split=='test'].copy()
    g=torch.Generator(); g.manual_seed(seed)
    return tr, {
        'train':DataLoader(HAMDataset(tr,tfm),batch_size=BATCH_SIZE,shuffle=True,generator=g,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True),
        'validation':DataLoader(HAMDataset(va,tfm),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True),
        'test':DataLoader(HAMDataset(te,tfm),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True),
    }


def class_weights(tr, device):
    c=tr.dx.value_counts().reindex(CLASS_NAMES).values.astype(float); w=len(tr)/(7*c)
    return torch.tensor(w,dtype=torch.float32,device=device)


@torch.no_grad()
def eval_det(model,loader,device,criterion):
    model.eval(); ys=[]; ps=[]; loss=0.; n=0
    for x,y,_ in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.amp.autocast('cuda',enabled=device.type=='cuda'): z=model(x); l=criterion(z,y)
        loss += float(l.item())*len(y); n += len(y); ys.extend(y.cpu().numpy()); ps.extend(z.argmax(1).cpu().numpy())
    return {'loss':loss/n,'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro',zero_division=0),'weighted_f1':f1_score(ys,ps,average='weighted',zero_division=0)}


def train_one(seed, df, device):
    seed_all(seed); out=MULTI_BASE/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
    tr,ld=loaders_for(df,seed); model=Net().to(device); criterion=nn.CrossEntropyLoss(weight=class_weights(tr,device)); opt=torch.optim.Adam(model.parameters(),lr=LR)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); best=-1.; best_epoch=-1; t0=time.perf_counter()
    for ep in range(1,EPOCHS+1):
        model.train(); ys=[]; ps=[]; loss_sum=0.; n=0
        for x,y,_ in ld['train']:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=device.type=='cuda'): z=model(x); loss=criterion(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum+=float(loss.item())*len(y); n+=len(y); ys.extend(y.cpu().numpy()); ps.extend(z.detach().argmax(1).cpu().numpy())
        val=eval_det(model,ld['validation'],device,criterion)
        row={'seed':seed,'epoch':ep,'train_loss':loss_sum/n,'train_accuracy':accuracy_score(ys,ps),'train_macro_f1':f1_score(ys,ps,average='macro',zero_division=0),'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1'],'val_loss':val['loss']}
        print('MULTISEED_EPOCH|'+json.dumps(row,sort_keys=True),flush=True)
        if val['macro_f1']>best:
            best=val['macro_f1']; best_epoch=ep; torch.save({'model_state':model.state_dict(),'epoch':ep,'val_macro_f1':best},out/'best.pt')
    train_sec=time.perf_counter()-t0
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model_state'])
    val_mc,val_y,val_ids=mc_predict(model,ld['validation'],device)
    test_mc,test_y,test_ids=mc_predict(model,ld['test'],device)
    summary=summarize(seed,best_epoch,best,train_sec,val_mc,val_y,test_mc,test_y)
    np.savez_compressed(out/'mc_predictions.npz',val_mc_probs=val_mc.astype(np.float32),val_labels=val_y,val_image_ids=val_ids,id_mc_probs=test_mc.astype(np.float32),id_labels=test_y,id_image_ids=test_ids,class_names=np.asarray(CLASS_NAMES,dtype='U10'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    print('MULTISEED_RESULT|'+json.dumps(summary,sort_keys=True),flush=True)
    del model, ld; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def mc_predict(model,loader,device):
    model.eval()
    for m in model.modules():
        if isinstance(m,nn.Dropout): m.train()
    passes=[]; yref=None; idref=None
    for t in range(MC_PASSES):
        pp=[]; yy=[]; ii=[]
        for x,y,ids in loader:
            x=x.to(device,non_blocking=True)
            with torch.amp.autocast('cuda',enabled=device.type=='cuda'): z=model(x)
            pp.append(torch.softmax(z.float(),1).cpu().numpy()); yy.extend(y.numpy()); ii.extend(list(ids))
        passes.append(np.concatenate(pp));
        if yref is None: yref=np.asarray(yy,int); idref=np.asarray(ii,dtype='U20')
    return np.stack(passes).transpose(1,0,2),yref,idref


def ece15(p,y):
    conf=p.max(1); pred=p.argmax(1); cor=(pred==y).astype(float); edges=np.linspace(0,1,16); e=0.
    for b in range(15):
        mask=(conf>=edges[b]) & ((conf<=edges[b+1]) if b==14 else (conf<edges[b+1]))
        if mask.any(): e += mask.mean()*abs(cor[mask].mean()-conf[mask].mean())
    return float(e)


def mi_score(mc):
    p=np.clip(mc.astype(np.float64),EPS,1); p/=p.sum(-1,keepdims=True); mp=p.mean(1)
    pe=-np.sum(mp*np.log(mp),1); ee=-np.mean(np.sum(p*np.log(p),2),1); return np.maximum(pe-ee,0),mp


def aurc(score,correct):
    o=np.argsort(score,kind='mergesort'); c=correct[o].astype(float); k=np.arange(1,len(c)+1); cov=k/len(c); risk=1-np.cumsum(c)/k
    return float(np.trapezoid(np.r_[risk[0],risk],np.r_[0.,cov]))


def val_threshold(score,target=.60):
    s=np.sort(score,kind='mergesort'); k=max(1,min(len(s),int(round(target*len(s))))); return float(s[k-1])


def summarize(seed,best_epoch,best_val,train_sec,val_mc,val_y,test_mc,test_y):
    vs,vp=mi_score(val_mc); ts,tp=mi_score(test_mc); pred=tp.argmax(1); err=(pred!=test_y).astype(int); mel=test_y==MEL
    thr=val_threshold(vs,.60); keep=ts<=thr; melkeep=mel&keep
    return {
        'seed':seed,'best_epoch':best_epoch,'best_val_macro_f1':float(best_val),'train_seconds':float(train_sec),
        'test_accuracy':float(accuracy_score(test_y,pred)),'test_macro_f1':float(f1_score(test_y,pred,average='macro',zero_division=0)),
        'test_weighted_f1':float(f1_score(test_y,pred,average='weighted',zero_division=0)),'test_melanoma_recall':float(np.mean(pred[mel]==MEL)),
        'test_ece15':ece15(tp,test_y),'mi_aurc':aurc(ts,pred==test_y),'mi_error_auroc':float(roc_auc_score(err,ts)),'mi_error_auprc':float(average_precision_score(err,ts)),
        'mi_val_threshold_60':thr,'mi_test_coverage_at_val60':float(keep.mean()),'mi_test_selective_accuracy_at_val60':float(accuracy_score(test_y[keep],pred[keep])),
        'mi_test_melanoma_coverage_at_val60':float(melkeep.sum()/mel.sum()),'mi_test_selective_melanoma_recall_at_val60':float(np.mean(pred[melkeep]==MEL)) if melkeep.any() else None,
    }


def main():
    if not SPLIT_CSV.exists(): raise FileNotFoundError(SPLIT_CSV)
    if len(list(IMG_DIR.glob('*.jpg'))) < 10015: raise RuntimeError('HAM10000 images are not prepared on this worker')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('MULTISEED_DEVICE|',device,flush=True)
    if device.type=='cuda': print('MULTISEED_GPU|',torch.cuda.get_device_name(0),flush=True)
    print('MULTISEED_ASSIGNED|',TRAIN_SEEDS,flush=True); df=pd.read_csv(SPLIT_CSV); results=[]
    for seed in TRAIN_SEEDS: results.append(train_one(int(seed),df,device))
    print('MULTISEED_NODE_COMPLETE|'+json.dumps(results,sort_keys=True),flush=True)

main()
