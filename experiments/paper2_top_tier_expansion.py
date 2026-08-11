"""Top-tier reviewer expansion experiments for Paper 2.

Distributed by worker hostname across three T4 nodes:
- Node2: TTA, Grad-CAM escaped melanoma analysis, focal-loss ResNet50, EfficientNet-B0 robustness.
- Node10: outer lesion-grouped folds 0-2 of a 5-fold nested grouped evaluation.
- Nabila: outer lesion-grouped folds 3-4 plus deeper spatial-dropout ablations.

All model-selection decisions use validation data only. Outer-fold test lesions are never
used for checkpoint selection or uncertainty threshold selection.
"""
import gc, hashlib, json, random, shutil, socket, tarfile, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF

BASE=Path('/kaggle/working/paper2_data')
IMG_DIR=BASE/'HAM10000_images'
DL=BASE/'downloads'
SPLIT_CSV=BASE/'splits/ham10000_lesion_group_split_seed2026.csv'
OUTROOT=BASE/'top_tier_expansion'
OUTROOT.mkdir(parents=True,exist_ok=True); DL.mkdir(parents=True,exist_ok=True); IMG_DIR.mkdir(parents=True,exist_ok=True)
HOST=socket.gethostname()
HOST_ROLE={'b44f0ce87fe6':'node2_explain_tta_focal_backbone','836b08d4b34d':'node10_cv_0_1_2','7e5f29eab5cb':'nabila_cv_3_4_deepdrop'}
ROLE=HOST_ROLE.get(HOST,'unknown')
CLASSES=['akiec','bcc','bkl','df','mel','nv','vasc']; C2I={c:i for i,c in enumerate(CLASSES)}; MEL=C2I['mel']; EPS=1e-12
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH=32; WORKERS=4; LR=1e-4; EPOCHS=20; PATIENCE=5; MC_T=30
print('EXPANSION_START|host='+HOST+'|role='+ROLE+'|device='+str(DEVICE),flush=True)
if DEVICE.type=='cuda': print('GPU|'+torch.cuda.get_device_name(0),flush=True)
HAM_FILES={'HAM10000_images_part_1.zip':(3172585,1366522108),'HAM10000_images_part_2.zip':(3172584,1403566547),'HAM10000_metadata.tab':(4338392,830428)}

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.benchmark=True

def norm(p):
 p=np.clip(np.asarray(p,float),EPS,None); return p/p.sum(-1,keepdims=True)
def entropy(p):
 p=np.clip(np.asarray(p,float),EPS,1); return -(p*np.log(p)).sum(-1)
def nll(p,y):
 p=norm(p); return float(-np.log(p[np.arange(len(y)),y]).mean())
def brier(p,y):
 p=norm(p); return float(np.mean(np.sum((p-np.eye(len(CLASSES))[y])**2,axis=1)))
def ece(p,y,bins=15):
 p=norm(p); pred=p.argmax(1); conf=p.max(1); ok=(pred==y).astype(float); edges=np.linspace(0,1,bins+1); v=0.
 for i in range(bins):
  m=(conf>=edges[i])&((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
  if m.any(): v+=m.mean()*abs(ok[m].mean()-conf[m].mean())
 return float(v)
def aurc_order(correct):
 c=np.asarray(correct,float); k=np.arange(1,len(c)+1); risk=1-np.cumsum(c)/k; cov=k/len(c); return float(np.trapezoid(np.r_[risk[0],risk],np.r_[0,cov]))
def aurc(score,pred,y):
 o=np.argsort(score,kind='mergesort'); return aurc_order(pred[o]==y[o])
def eaurc(score,pred,y): return aurc(score,pred,y)-aurc_order(np.sort((pred==y).astype(int))[::-1])
def threshold_for_coverage(score,target=.60):
 s=np.sort(np.asarray(score),kind='mergesort'); k=max(1,min(len(s),int(round(target*len(s))))); return float(s[k-1])
def safety(p,y,score,thr):
 p=norm(p); pr=p.argmax(1); keep=np.asarray(score)<=thr; mel=y==MEL; mk=keep&mel; fn=mel&(pr!=MEL); rfn=fn&keep; rn=int(mk.sum()); mn=int(mel.sum())
 return {'coverage':float(keep.mean()),'selective_accuracy':float(accuracy_score(y[keep],pr[keep])) if keep.any() else np.nan,'selective_macro_f1':float(f1_score(y[keep],pr[keep],average='macro',zero_division=0)) if keep.any() else np.nan,'melanoma_n':mn,'melanoma_coverage':float(rn/mn) if mn else np.nan,'retained_melanoma_n':rn,'retained_melanoma_fn':int(rfn.sum()),'retained_melanoma_sensitivity':float((mk&(pr==MEL)).sum()/rn) if rn else np.nan,'retained_melanoma_fnr':float(rfn.sum()/rn) if rn else np.nan,'automatic_melanoma_miss_rate':float(rfn.sum()/mn) if mn else np.nan,'melanoma_fn_escape_rate':float(rfn.sum()/fn.sum()) if fn.sum() else 0.0}
def metrics(p,y,score):
 p=norm(p); pr=p.argmax(1); err=(pr!=y).astype(int)
 return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pr)),'macro_f1':float(f1_score(y,pr,average='macro',zero_division=0)),'weighted_f1':float(f1_score(y,pr,average='weighted',zero_division=0)),'melanoma_recall':float(recall_score(y==MEL,pr==MEL,zero_division=0)),'nll':nll(p,y),'brier':brier(p,y),'ece15':ece(p,y),'aurc':aurc(score,pr,y),'eaurc':eaurc(score,pr,y),'error_auroc':float(roc_auc_score(err,score)) if len(np.unique(err))>1 else np.nan,'error_auprc':float(average_precision_score(err,score)) if len(np.unique(err))>1 else np.nan}

def download_ham():
 for name,(fid,size) in HAM_FILES.items():
  dest=DL/name
  if not dest.exists() or dest.stat().st_size!=size:
   if dest.exists(): dest.unlink()
   print('HAM_DOWNLOAD_START|'+name,flush=True)
   with requests.get(f'https://dataverse.harvard.edu/api/access/datafile/{fid}',stream=True,timeout=(30,180)) as r:
    r.raise_for_status(); tmp=Path(str(dest)+'.part')
    if tmp.exists(): tmp.unlink()
    with open(tmp,'wb') as f:
     for ch in r.iter_content(8*1024*1024):
      if ch: f.write(ch)
    tmp.replace(dest)
   if dest.stat().st_size!=size: raise RuntimeError(f'{name} size {dest.stat().st_size} != {size}')
   print('HAM_DOWNLOAD_DONE|'+name+'|'+str(size),flush=True)
 for name in ['HAM10000_images_part_1.zip','HAM10000_images_part_2.zip']:
  marker=IMG_DIR/(Path(name).stem+'.extracted')
  if not marker.exists() or len(list(IMG_DIR.glob('*.jpg')))<10000:
   print('HAM_EXTRACT_START|'+name,flush=True)
   with zipfile.ZipFile(DL/name) as z:
    for member in z.namelist():
     if member.lower().endswith(('.jpg','.jpeg','.png')):
      with z.open(member) as src, open(IMG_DIR/Path(member).name,'wb') as dst: shutil.copyfileobj(src,dst,4*1024*1024)
   marker.write_text('ok')
 print('HAM_IMAGES|'+str(len(list(IMG_DIR.glob('*.jpg')))),flush=True)
 if len(list(IMG_DIR.glob('*.jpg')))<10015: raise RuntimeError('HAM images incomplete')

def metadata():
 p=DL/'HAM10000_metadata.tab'; df=pd.read_csv(p,sep='\t')
 if len(df.columns)==1: df=pd.read_csv(p)
 df.columns=[str(c).strip() for c in df.columns]
 for c in ['lesion_id','image_id','dx']: df[c]=df[c].astype(str).str.strip()
 return df
BASE_TFM=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class HamDS(Dataset):
 def __init__(self,df,tfm=BASE_TFM): self.df=df.reset_index(drop=True); self.tfm=tfm
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]
  with Image.open(IMG_DIR/f"{r.image_id}.jpg") as im: x=self.tfm(im.convert('RGB'))
  return x,C2I[r.dx],r.image_id
def loader(df,shuffle=False,batch=BATCH): return DataLoader(HamDS(df),batch_size=batch,shuffle=shuffle,num_workers=WORKERS,pin_memory=True,persistent_workers=(WORKERS>0))

class HeadResNet50(nn.Module):
 def __init__(self,p=.2,weights=True):
  super().__init__(); b=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if weights else None); d=b.fc.in_features; b.fc=nn.Identity(); self.backbone=b; self.fc1=nn.Linear(d,512); self.relu=nn.ReLU(inplace=True); self.dropout=nn.Dropout(p); self.fc2=nn.Linear(512,7)
 def forward(self,x): return self.fc2(self.dropout(self.relu(self.fc1(self.backbone(x)))))
class SpatialDropResNet50(nn.Module):
 def __init__(self,stages=('layer4',),p=.2):
  super().__init__(); b=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); self.conv1=b.conv1; self.bn1=b.bn1; self.relu=b.relu; self.maxpool=b.maxpool; self.layer1=b.layer1; self.layer2=b.layer2; self.layer3=b.layer3; self.layer4=b.layer4; self.avgpool=b.avgpool; self.stages=set(stages); self.sd=nn.Dropout2d(p); self.fc1=nn.Linear(b.fc.in_features,512); self.hrelu=nn.ReLU(inplace=True); self.dropout=nn.Dropout(p); self.fc2=nn.Linear(512,7)
 def forward(self,x):
  x=self.maxpool(self.relu(self.bn1(self.conv1(x)))); x=self.layer1(x); x=self.layer2(x); x=self.layer3(x)
  if 'layer3' in self.stages: x=self.sd(x)
  x=self.layer4(x)
  if 'layer4' in self.stages: x=self.sd(x)
  x=torch.flatten(self.avgpool(x),1); return self.fc2(self.dropout(self.hrelu(self.fc1(x))))
class EffB0(nn.Module):
 def __init__(self,p=.2):
  super().__init__(); b=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1); d=b.classifier[1].in_features; b.classifier=nn.Identity(); self.backbone=b; self.fc1=nn.Linear(d,512); self.relu=nn.ReLU(inplace=True); self.dropout=nn.Dropout(p); self.fc2=nn.Linear(512,7)
 def forward(self,x): return self.fc2(self.dropout(self.relu(self.fc1(self.backbone(x)))))

def weights_for(df):
 counts=df.dx.value_counts().reindex(CLASSES).values.astype(float); w=len(df)/(len(CLASSES)*counts); return torch.tensor(w,dtype=torch.float32,device=DEVICE)
class FocalLoss(nn.Module):
 def __init__(self,weight,gamma=2.): super().__init__(); self.weight=weight; self.gamma=gamma
 def forward(self,logits,y):
  ce=F.cross_entropy(logits,y,weight=self.weight,reduction='none'); pt=torch.softmax(logits,1)[torch.arange(len(y),device=y.device),y]; return (((1-pt)**self.gamma)*ce).mean()
@torch.no_grad()
def det_probs(model,ld):
 model.eval(); ps=[]; ys=[]; ids=[]
 for x,y,i in ld:
  x=x.to(DEVICE,non_blocking=True)
  with torch.amp.autocast('cuda',enabled=DEVICE.type=='cuda'): z=model(x)
  ps.append(torch.softmax(z.float(),1).cpu().numpy()); ys.extend(y.numpy()); ids.extend(list(i))
 return np.concatenate(ps),np.asarray(ys,int),np.asarray(ids,dtype='U20')
def enable_mc(model):
 model.eval()
 for m in model.modules():
  if isinstance(m,(nn.Dropout,nn.Dropout2d)): m.train()
@torch.no_grad()
def mc_probs(model,ld,T=MC_T):
 enable_mc(model); arr=[]; yref=None; iref=None
 for t in range(T):
  ps=[]; ys=[]; ids=[]
  for x,y,i in ld:
   x=x.to(DEVICE,non_blocking=True)
   with torch.amp.autocast('cuda',enabled=DEVICE.type=='cuda'): z=model(x)
   ps.append(torch.softmax(z.float(),1).cpu().numpy()); ys.extend(y.numpy()); ids.extend(list(i))
  arr.append(np.concatenate(ps))
  if yref is None: yref=np.asarray(ys,int); iref=np.asarray(ids,dtype='U20')
  if (t+1)%5==0 or t==0: print(f'MC|{t+1}/{T}',flush=True)
 return np.stack(arr).transpose(1,0,2),yref,iref
def mc_scores(mc):
 p=norm(mc); mp=p.mean(1); pe=entropy(mp); ee=entropy(p).mean(1); return mp,{'pe':pe,'ee':ee,'mi':np.maximum(pe-ee,0),'msp':1-mp.max(1)}
def save_fp16(model,path,meta=None):
 sd={k:v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu() for k,v in model.state_dict().items()}; torch.save({'state_dict_fp16':sd,'meta':meta or {}},path)

def train_one(model,tr,val,out,seed,loss_kind='ce',epochs=EPOCHS):
 seed_all(seed); model=model.to(DEVICE); trld=loader(tr,True); vald=loader(val,False); cw=weights_for(tr); criterion=FocalLoss(cw,2.) if loss_kind=='focal' else nn.CrossEntropyLoss(weight=cw); opt=torch.optim.Adam(model.parameters(),lr=LR); scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda'); best=-1.; best_sd=None; hist=[]; stale=0
 for ep in range(1,epochs+1):
  model.train(); losses=[]; yy=[]; pp=[]
  for x,y,_ in trld:
   x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True)
   with torch.amp.autocast('cuda',enabled=DEVICE.type=='cuda'): z=model(x); loss=criterion(z,y)
   scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.item())*len(y)); yy.extend(y.detach().cpu().numpy()); pp.extend(z.detach().argmax(1).cpu().numpy())
  vp,vy,_=det_probs(model,vald); vf=float(f1_score(vy,vp.argmax(1),average='macro',zero_division=0)); row={'epoch':ep,'train_loss':sum(losses)/len(tr),'train_macro_f1':float(f1_score(yy,pp,average='macro',zero_division=0)),'val_macro_f1':vf,'val_accuracy':float(accuracy_score(vy,vp.argmax(1)))}; hist.append(row); print('TRAIN|'+json.dumps(row),flush=True)
  if vf>best+1e-6: best=vf; best_sd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
  else: stale+=1
  if ep>=8 and stale>=PATIENCE: print('EARLY_STOP|'+str(ep),flush=True); break
 model.load_state_dict(best_sd); pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False); save_fp16(model,out/'best_model_fp16.pt',{'best_val_macro_f1':best,'seed':seed,'loss':loss_kind}); return model,best

def eval_pair(model,val,test,out):
 vld=loader(val); tld=loader(test); vd,vy,vid=det_probs(model,vld); td,ty,tid=det_probs(model,tld); vm,_,_=mc_probs(model,vld); tm,_,_=mc_probs(model,tld); vmp,vs=mc_scores(vm); tmp,ts=mc_scores(tm); dthr=threshold_for_coverage(1-vd.max(1),.60); mthr=threshold_for_coverage(vs['pe'],.60); res={'deterministic':metrics(td,ty,1-td.max(1)),'deterministic_60':safety(td,ty,1-td.max(1),dthr),'mc':metrics(tmp,ty,ts['pe']),'mc_60':safety(tmp,ty,ts['pe'],mthr),'thresholds':{'det_msp':dthr,'mc_pe':mthr}}; np.savez_compressed(out/'predictions.npz',val_det=vd.astype(np.float32),test_det=td.astype(np.float32),val_mc=vm.astype(np.float16),test_mc=tm.astype(np.float16),val_y=vy,test_y=ty,val_ids=vid,test_ids=tid); (out/'summary.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); return res

def run_cv(df,folds):
 root=OUTROOT/'cv5'; root.mkdir(exist_ok=True); outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026); splits=list(outer.split(df.image_id,df.dx,df.lesion_id)); rows=[]
 for fold in folds:
  trainval_idx,test_idx=splits[fold]; tv=df.iloc[trainval_idx].reset_index(drop=True); test=df.iloc[test_idx].reset_index(drop=True); inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=5200+fold); itr,iv=next(inner.split(tv.image_id,tv.dx,tv.lesion_id)); tr=tv.iloc[itr].reset_index(drop=True); val=tv.iloc[iv].reset_index(drop=True); leak=len(set(tr.lesion_id)&set(val.lesion_id))+len(set(tr.lesion_id)&set(test.lesion_id))+len(set(val.lesion_id)&set(test.lesion_id)); print(f'CV_FOLD|{fold}|tr={len(tr)}|val={len(val)}|test={len(test)}|leak={leak}',flush=True); fo=root/f'fold_{fold}'; fo.mkdir(exist_ok=True); pd.concat([tr.assign(split='train'),val.assign(split='validation'),test.assign(split='test')]).to_csv(fo/'split.csv',index=False); model,best=train_one(HeadResNet50(.2),tr,val,fo,2026+fold,'ce'); res=eval_pair(model,val,test,fo); row={'fold':fold,'best_val_macro_f1':best,**{f'det_{k}':v for k,v in res['deterministic'].items() if isinstance(v,(int,float))},**{f'mc_{k}':v for k,v in res['mc'].items() if isinstance(v,(int,float))},'det_auto_mel_miss':res['deterministic_60']['automatic_melanoma_miss_rate'],'mc_auto_mel_miss':res['mc_60']['automatic_melanoma_miss_rate']}; rows.append(row); del model; gc.collect(); torch.cuda.empty_cache(); pd.DataFrame(rows).to_csv(root/f'partial_{HOST}.csv',index=False)
 return rows

def load_fixed_split(df):
 if SPLIT_CSV.exists():
  s=pd.read_csv(SPLIT_CSV); return s[s.split=='train'].copy(),s[s.split=='validation'].copy(),s[s.split=='test'].copy()
 sg=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=2026); fold=np.empty(len(df),int)
 for f,(_,idx) in enumerate(sg.split(df.image_id,df.dx,df.lesion_id)): fold[idx]=f
 s=df.copy(); s['group_fold']=fold; s['split']='train'; s.loc[s.group_fold==8,'split']='validation'; s.loc[s.group_fold==9,'split']='test'; SPLIT_CSV.parent.mkdir(exist_ok=True); s.to_csv(SPLIT_CSV,index=False); return s[s.split=='train'].copy(),s[s.split=='validation'].copy(),s[s.split=='test'].copy()
def find_seed2026_checkpoint():
 cands=list((BASE/'grouped_mc_seed2026').rglob('model_best_val_macro_f1.pt'))+list((BASE/'grouped_mc_seed2026').rglob('*best*.pt'))
 if not cands: raise FileNotFoundError('seed2026 checkpoint missing')
 return cands[0]
def load_seed2026():
 m=HeadResNet50(.2,weights=False).to(DEVICE); obj=torch.load(find_seed2026_checkpoint(),map_location=DEVICE,weights_only=False); st=obj.get('model_state',obj.get('state_dict',obj)); m.load_state_dict(st,strict=True); m.eval(); return m

def tta_transform(kind):
 def f(im):
  im=im.resize((224,224))
  if kind=='hflip': im=TF.hflip(im)
  elif kind=='vflip': im=TF.vflip(im)
  elif kind=='rotp10': im=TF.rotate(im,10)
  elif kind=='rotn10': im=TF.rotate(im,-10)
  elif kind=='crop': im=im.resize((240,240)); im=TF.center_crop(im,[216,216]); im=im.resize((224,224))
  elif kind=='hrot': im=TF.rotate(TF.hflip(im),10)
  elif kind=='vrot': im=TF.rotate(TF.vflip(im),-10)
  x=TF.to_tensor(im); return TF.normalize(x,[.485,.456,.406],[.229,.224,.225])
 return f
def run_tta(model,val,test,out):
 kinds=['identity','hflip','vflip','rotp10','rotn10','crop','hrot','vrot']; arr={}
 for split,frame in [('val',val),('test',test)]:
  ps=[]; yy=None; ids=None
  for k in kinds:
   ld=DataLoader(HamDS(frame,tta_transform(k)),batch_size=64,shuffle=False,num_workers=WORKERS,pin_memory=True); p,y,i=det_probs(model,ld); ps.append(p); yy=y; ids=i; print('TTA|'+split+'|'+k,flush=True)
  a=np.stack(ps,1); mp=norm(a.mean(1)); score=entropy(mp); arr[split]=(a,mp,score,yy,ids)
 v=arr['val']; t=arr['test']; thr=threshold_for_coverage(v[2],.60); res={'tta':metrics(t[1],t[3],t[2]),'tta_60':safety(t[1],t[3],t[2],thr),'threshold_pe':thr,'augmentations':kinds}; np.savez_compressed(out/'tta_predictions.npz',val_tta=v[0].astype(np.float16),test_tta=t[0].astype(np.float16),val_y=v[3],test_y=t[3],val_ids=v[4],test_ids=t[4]); (out/'tta_summary.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); return res

def locate_saved_mc():
 c=list((BASE/'grouped_mc_seed2026').rglob('*predictions*.npz')); return c[0] if c else None
def gradcam_escaped(model,val,test,out):
 import matplotlib.pyplot as plt
 p=locate_saved_mc(); escaped=[]
 if p:
  z=np.load(p,allow_pickle=False); keys=z.files
  def pick(names):
   for n in names:
    if n in keys: return z[n]
   return None
  vm=pick(['val_mc_probs']); tm=pick(['id_mc_probs','test_mc_probs']); ty=pick(['id_labels','test_labels']); tids=pick(['id_image_ids','test_image_ids'])
  if vm is not None and tm is not None:
   _,vs=mc_scores(vm); tmp,ts=mc_scores(tm); thr=threshold_for_coverage(vs['pe'],.60); pr=tmp.argmax(1); keep=ts['pe']<=thr; ix=np.where((ty==MEL)&(pr!=MEL)&keep)[0]
   for j in ix: escaped.append({'image_id':str(tids[j]),'pred':int(pr[j]),'confidence':float(tmp[j].max()),'pe':float(ts['pe'][j]),'mi':float(ts['mi'][j]),'thr':thr})
 if not escaped: return {'n':0,'note':'No escaped set could be reconstructed from saved MC predictions.'}
 camdir=out/'gradcam_escaped'; camdir.mkdir(exist_ok=True); rows=[]; acts={}; grads={}; target=model.backbone.layer4[-1]; h1=target.register_forward_hook(lambda m,i,o: acts.__setitem__('x',o)); h2=target.register_full_backward_hook(lambda m,gi,go: grads.__setitem__('x',go[0])); model.eval()
 for r in escaped:
  iid=r['image_id']; path=IMG_DIR/f'{iid}.jpg'
  with Image.open(path) as im0: im=im0.convert('RGB'); x=BASE_TFM(im).unsqueeze(0).to(DEVICE)
  model.zero_grad(set_to_none=True); z=model(x); pred=int(z.argmax(1)); z[0,pred].backward(); a=acts['x'][0].detach(); g=grads['x'][0].detach(); w=g.mean((1,2)); cam=torch.relu((w[:,None,None]*a).sum(0)); cam=F.interpolate(cam[None,None],size=(224,224),mode='bilinear',align_corners=False)[0,0]; cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8); cam=cam.cpu().numpy(); fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(im.resize((224,224))); ax.imshow(cam,cmap='jet',alpha=.38); ax.axis('off'); ax.set_title(f"{iid}: true mel, pred {CLASSES[pred]}\nconf={r['confidence']:.3f}, PE={r['pe']:.3f}",fontsize=8); fig.tight_layout(); fig.savefig(camdir/f'{iid}_gradcam.png',dpi=180,bbox_inches='tight'); plt.close(fig); rows.append({**r,'pred_class':CLASSES[pred]})
 h1.remove(); h2.remove(); pd.DataFrame(rows).to_csv(out/'escaped_melanoma_gradcam_metadata.csv',index=False); return {'n':len(rows),'threshold':escaped[0]['thr']}
def run_fixed_training(tr,val,test,out,name,model,loss='ce',seed=9001):
 d=out/name; d.mkdir(parents=True,exist_ok=True); m,b=train_one(model,tr,val,d,seed,loss); res=eval_pair(m,val,test,d); res['best_val_macro_f1']=b; return res
def aggregate_local(role,summary):
 d=OUTROOT/f'{role}_{HOST}'; d.mkdir(exist_ok=True); (d/'role_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); return d

def main():
 if ROLE=='unknown': raise RuntimeError('Unknown worker hostname '+HOST)
 download_ham(); df=metadata(); tr,val,test=load_fixed_split(df); summary={'host':HOST,'role':ROLE,'started':time.time()}
 if ROLE=='node10_cv_0_1_2': summary['cv_rows']=run_cv(df,[0,1,2]); role_dir=aggregate_local(ROLE,summary)
 elif ROLE=='nabila_cv_3_4_deepdrop':
  summary['cv_rows']=run_cv(df,[3,4]); summary['deep_dropout']={}; summary['deep_dropout']['layer4']=run_fixed_training(tr,val,test,OUTROOT,'deepdrop_layer4',SpatialDropResNet50(('layer4',),.2),'ce',9301); gc.collect(); torch.cuda.empty_cache(); summary['deep_dropout']['layer3_layer4']=run_fixed_training(tr,val,test,OUTROOT,'deepdrop_layer3_layer4',SpatialDropResNet50(('layer3','layer4'),.2),'ce',9302); role_dir=aggregate_local(ROLE,summary)
 elif ROLE=='node2_explain_tta_focal_backbone':
  fixed=load_seed2026(); tdir=OUTROOT/'tta_gradcam'; tdir.mkdir(exist_ok=True); summary['tta']=run_tta(fixed,val,test,tdir); summary['gradcam']=gradcam_escaped(fixed,val,test,tdir); del fixed; gc.collect(); torch.cuda.empty_cache(); summary['focal']=run_fixed_training(tr,val,test,OUTROOT,'focal_loss_resnet50',HeadResNet50(.2),'focal',9401); gc.collect(); torch.cuda.empty_cache(); summary['efficientnet_b0']=run_fixed_training(tr,val,test,OUTROOT,'efficientnet_b0',EffB0(.2),'ce',9402); role_dir=aggregate_local(ROLE,summary)
 summary['finished']=time.time(); (role_dir/'role_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); archive=BASE/f'Paper2_TOP_TIER_{ROLE}.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=4) as tf:
  if ROLE=='node10_cv_0_1_2':
   for f in [0,1,2]: tf.add(OUTROOT/'cv5'/f'fold_{f}',arcname=f'cv5/fold_{f}')
  elif ROLE=='nabila_cv_3_4_deepdrop':
   for f in [3,4]: tf.add(OUTROOT/'cv5'/f'fold_{f}',arcname=f'cv5/fold_{f}')
   tf.add(OUTROOT/'deepdrop_layer4',arcname='deepdrop_layer4'); tf.add(OUTROOT/'deepdrop_layer3_layer4',arcname='deepdrop_layer3_layer4')
  else:
   tf.add(OUTROOT/'tta_gradcam',arcname='tta_gradcam'); tf.add(OUTROOT/'focal_loss_resnet50',arcname='focal_loss_resnet50'); tf.add(OUTROOT/'efficientnet_b0',arcname='efficientnet_b0')
  tf.add(role_dir,arcname='role')
 sha=hashlib.sha256(archive.read_bytes()).hexdigest(); (Path(str(archive)+'.sha256')).write_text(sha+'  '+archive.name+'\n'); print('EXPANSION_ARCHIVE|'+str(archive)+'|'+str(archive.stat().st_size)+'|'+sha,flush=True); print('TOP_TIER_EXPANSION_DONE|'+ROLE,flush=True)
main()
