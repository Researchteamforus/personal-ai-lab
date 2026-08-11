"""Frozen external validation of Paper 2 models on the official HIBA Skin Lesions cohort.

Primary analysis is restricted to dermoscopic HIBA images whose diagnoses can be
pre-specified into the seven HAM10000 output classes. Squamous-cell carcinoma is
excluded because the trained classifier has no corresponding output class. No HIBA
labels are used for model fitting, score selection, calibration, or threshold tuning.
"""
import json, re, tarfile, hashlib, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

BASE=Path('/kaggle/working/paper2_data')
OUT=BASE/'external_hiba'; OUT.mkdir(parents=True,exist_ok=True)
HROOT=Path('/kaggle/working/HIBA_external')
META_URL='https://isic-archive.s3.amazonaws.com/dois/10.34970-559884/hiba-skin-lesions.csv'
ZIP_URL='https://isic-archive.s3.amazonaws.com/dois/10.34970-559884/hiba-skin-lesions.zip'
META=HROOT/'hiba-skin-lesions.csv'; BUNDLE=HROOT/'hiba-skin-lesions.zip'; EXTRACT=HROOT/'bundle'
ARCHIVE=BASE/'Paper2_HIBA_EXTERNAL_RESULTS.tar.gz'; SHAF=BASE/'Paper2_HIBA_EXTERNAL_RESULTS.sha256'
SEEDS=[11,29,47,71,2026]
CLASSES=['akiec','bcc','bkl','df','mel','nv','vasc']; IDX={c:i for i,c in enumerate(CLASSES)}; MEL=IDX['mel']; EPS=1e-12
DX_MAP={'actinic keratosis':'akiec','basal cell carcinoma':'bcc','seborrheic keratosis':'bkl','solar lentigo':'bkl','lichenoid keratosis':'bkl','dermatofibroma':'df','melanoma':'mel','nevus':'nv','vascular lesion':'vasc'}

def norm(p):
 p=np.clip(np.asarray(p,float),EPS,None); return p/p.sum(-1,keepdims=True)
def entropy(p):p=np.clip(p,EPS,1);return -(p*np.log(p)).sum(-1)
def nll(p,y):p=norm(p);return float(-np.log(p[np.arange(len(y)),y]).mean())
def brier(p,y):p=norm(p);return float(np.mean(np.sum((p-np.eye(7)[y])**2,axis=1)))
def ece(p,y,bins=15):
 p=norm(p);pr=p.argmax(1);cf=p.max(1);ok=(pr==y).astype(float);ed=np.linspace(0,1,bins+1);v=0.
 for i in range(bins):
  m=(cf>=ed[i])&((cf<ed[i+1]) if i<bins-1 else (cf<=ed[i+1]))
  if m.any():v+=m.mean()*abs(ok[m].mean()-cf[m].mean())
 return float(v)
def binary_ece(prob,target,bins=15):
 prob=np.asarray(prob,float);target=np.asarray(target,float);ed=np.linspace(0,1,bins+1);v=0.
 for i in range(bins):
  m=(prob>=ed[i])&((prob<ed[i+1]) if i<bins-1 else (prob<=ed[i+1]))
  if m.any():v+=m.mean()*abs(target[m].mean()-prob[m].mean())
 return float(v)
def aurc_order(correct):
 c=np.asarray(correct,float);k=np.arange(1,len(c)+1);risk=1-np.cumsum(c)/k;cov=k/len(c);return float(np.trapezoid(np.r_[risk[0],risk],np.r_[0,cov]))
def aurc(score,pred,y):o=np.argsort(score,kind='mergesort');return aurc_order(pred[o]==y[o])
def eaurc(score,pred,y):return aurc(score,pred,y)-aurc_order(np.sort((pred==y).astype(int))[::-1])
def threshold_for_coverage(score,target=.60):
 s=np.sort(score,kind='mergesort');k=max(1,min(len(s),int(round(target*len(s)))));return float(s[k-1])
def safety(p,y,score,thr):
 p=norm(p);pr=p.argmax(1);keep=score<=thr;mel=y==MEL;mk=keep&mel;fullfn=mel&(pr!=MEL);retfn=fullfn&keep;rn=int(mk.sum());mn=int(mel.sum())
 return {'coverage':float(keep.mean()),'selective_accuracy':float(accuracy_score(y[keep],pr[keep])) if keep.any() else np.nan,'selective_macro_f1':float(f1_score(y[keep],pr[keep],average='macro',zero_division=0)) if keep.any() else np.nan,'melanoma_n':mn,'melanoma_coverage':float(rn/mn) if mn else np.nan,'retained_melanoma_n':rn,'retained_melanoma_tp':int((mk&(pr==MEL)).sum()),'retained_melanoma_fn':int(retfn.sum()),'retained_melanoma_sensitivity':float((mk&(pr==MEL)).sum()/rn) if rn else np.nan,'retained_melanoma_fnr':float(retfn.sum()/rn) if rn else np.nan,'automatic_melanoma_miss_rate':float(retfn.sum()/mn) if mn else np.nan,'melanoma_fn_escape_rate':float(retfn.sum()/fullfn.sum()) if fullfn.sum() else 0.0}
def metrics(p,y,score):
 p=norm(p);pr=p.argmax(1);err=(pr!=y).astype(int)
 return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pr)),'macro_f1':float(f1_score(y,pr,average='macro',zero_division=0)),'weighted_f1':float(f1_score(y,pr,average='weighted',zero_division=0)),'melanoma_recall':float(recall_score(y==MEL,pr==MEL,zero_division=0)),'nll':nll(p,y),'brier':brier(p,y),'ece15':ece(p,y),'classwise_ece_mean':float(np.mean([binary_ece(p[:,c],y==c) for c in range(7)])),'melanoma_binary_ece':binary_ece(p[:,MEL],y==MEL),'aurc':aurc(score,pr,y),'eaurc':eaurc(score,pr,y),'error_auroc':float(roc_auc_score(err,score)) if len(np.unique(err))>1 else np.nan,'error_auprc':float(average_precision_score(err,score)) if len(np.unique(err))>1 else np.nan}
def fit_temperature(p,y):
 from scipy.optimize import minimize_scalar
 p=norm(p)
 def app(t):
  z=np.log(np.clip(p,EPS,1))/t;z-=z.max(1,keepdims=True);q=np.exp(z);return q/q.sum(1,keepdims=True)
 return float(minimize_scalar(lambda t:nll(app(float(t)),y),bounds=(.1,8),method='bounded',options={'xatol':1e-5}).x)
def apply_temperature(p,t):
 p=norm(p);z=np.log(np.clip(p,EPS,1))/t;z-=z.max(1,keepdims=True);q=np.exp(z);return q/q.sum(1,keepdims=True)
def constrained_threshold(p,y,score,max_auto_miss=.10):
 pr=norm(p).argmax(1);mel=y==MEL;o=np.argsort(score,kind='mergesort');best=None
 for k in range(1,len(y)+1):
  keep=np.zeros(len(y),bool);keep[o[:k]]=1;miss=(keep&mel&(pr!=MEL)).sum()/max(1,mel.sum())
  if miss<=max_auto_miss+1e-12:best=(float(score[o[k-1]]),float(k/len(y)),float(miss))
 return best

def seed_dir(seed):
 candidates=[]
 if seed==2026:candidates.append(BASE/'grouped_mc_seed2026')
 root=BASE/'multiseed_fixed_split'
 if root.exists():
  candidates += [d for d in [root]+[x for x in root.rglob('*') if x.is_dir()] if re.search(rf'(?<!\d){seed}(?!\d)',str(d))]
 for d in candidates:
  if d.exists() and (list(d.rglob('*.pt')) or list(d.rglob('*.npz'))):return d
 raise FileNotFoundError(f'No seed directory for {seed}; checked {candidates[:8]}')
def choose(d,names,patterns):
 for n in names:
  direct=d/n
  if direct.exists():return direct
  nested=list(d.rglob(n))
  if nested:return nested[0]
 for pat in patterns:
  x=list(d.glob(pat)) or list(d.rglob(pat))
  if x:return x[0]
 raise FileNotFoundError(f'No matching file in {d}: {names} {patterns}; sample={[str(x) for x in list(d.rglob("*"))[:20]]}')
def checkpoint(seed):return choose(seed_dir(seed),['best.pt','model_best_val_macro_f1.pt'],['*best*.pt','*.pt'])
def predfile(seed):return choose(seed_dir(seed),['mc_predictions.npz','paper2_grouped_mc_predictions.npz'],['*predictions*.npz','*.npz'])
def load_internal(seed):return np.load(predfile(seed),allow_pickle=False)
def npz_pick(z,names):
 for n in names:
  if n in z.files:return z[n]
 raise KeyError(f'Missing {names}; keys={z.files}')
class Net(nn.Module):
 def __init__(self):
  super().__init__();b=models.resnet50(weights=None);d=b.fc.in_features;b.fc=nn.Identity();self.backbone=b;self.fc1=nn.Linear(d,512);self.relu=nn.ReLU(inplace=True);self.dropout=nn.Dropout(.2);self.fc2=nn.Linear(512,7)
 def forward(self,x):return self.fc2(self.dropout(self.relu(self.fc1(self.backbone(x)))))
def model_for(seed,device):
 m=Net().to(device);cp=checkpoint(seed);print(f'HIBA_CHECKPOINT|seed={seed}|{cp}',flush=True);obj=torch.load(cp,map_location=device,weights_only=False);st=obj.get('model_state',obj.get('state_dict',obj)) if isinstance(obj,dict) else obj;st={k.replace('module.','',1) if k.startswith('module.') else k:v for k,v in st.items()};m.load_state_dict(st,strict=True);m.eval();return m
class HibaDataset(Dataset):
 def __init__(self,frame,paths):
  self.f=frame.reset_index(drop=True);self.paths=paths;self.t=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.f)
 def __getitem__(self,i):
  r=self.f.iloc[i];p=self.paths[r.isic_id]
  with Image.open(p) as im:x=self.t(im.convert('RGB'))
  return x,IDX[r.ham_class],r.isic_id
@torch.no_grad()
def predict_det(m,ld,device):
 m.eval();pp=[];yy=[];ids=[]
 for x,y,i in ld:
  x=x.to(device,non_blocking=True)
  with torch.amp.autocast('cuda',enabled=device.type=='cuda'):z=m(x)
  pp.append(torch.softmax(z.float(),1).cpu().numpy());yy.extend(y.numpy());ids.extend(list(i))
 return np.concatenate(pp),np.asarray(yy,int),np.asarray(ids,dtype='U20')
@torch.no_grad()
def predict_keep_state(m,ld,device):
 pp=[];yy=[];ids=[]
 for x,y,i in ld:
  x=x.to(device,non_blocking=True)
  with torch.amp.autocast('cuda',enabled=device.type=='cuda'):z=m(x)
  pp.append(torch.softmax(z.float(),1).cpu().numpy());yy.extend(y.numpy());ids.extend(list(i))
 return np.concatenate(pp),np.asarray(yy,int),np.asarray(ids,dtype='U20')
@torch.no_grad()
def predict_mc(m,ld,device,T=30):
 m.eval()
 for mod in m.modules():
  if isinstance(mod,nn.Dropout):mod.train()
 allp=[];yref=None;iref=None
 for t in range(T):
  p,y,i=predict_keep_state(m,ld,device);allp.append(p)
  if yref is None:yref,iref=y,i
  print(f'HIBA_MC_PASS|{t+1}/{T}',flush=True)
 return np.stack(allp).transpose(1,0,2),yref,iref
def orient_mc(a,n):
 a=np.asarray(a)
 if a.ndim==2:return a[:,None,:]
 if a.shape[0]==n:return a
 if a.shape[1]==n:return a.transpose(1,0,2)
 raise ValueError(a.shape)
def mc_scores(mc):
 p=norm(mc);mp=p.mean(1);pe=entropy(mp);ee=entropy(p).mean(1);return mp,{'pe':pe,'ee':ee,'mi':np.maximum(pe-ee,0),'msp':1-mp.max(1)}
def cluster_boot(p,y,score,les,thr,B=1000,seed=20260811):
 u=np.unique(les);imap={g:np.where(les==g)[0] for g in u};rng=np.random.default_rng(seed);vals={}
 for _ in range(B):
  samp=rng.choice(u,len(u),replace=True);ix=np.concatenate([imap[g] for g in samp]);d=metrics(p[ix],y[ix],score[ix]);d.update({'sel_'+k:v for k,v in safety(p[ix],y[ix],score[ix],thr).items()})
  for k,v in d.items():
   if isinstance(v,(float,int,np.floating)) and np.isfinite(v):vals.setdefault(k,[]).append(v)
 point=metrics(p,y,score);point.update({'sel_'+k:v for k,v in safety(p,y,score,thr).items()});return [{'metric':k,'point':point.get(k,np.nan),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))} for k,v in vals.items()]
def per_class(p,y,score,thr):
 p=norm(p);pr=p.argmax(1);keep=score<=thr;rows=[]
 for c,n in enumerate(CLASSES):
  m=y==c;mk=m&keep;rows.append({'class':n,'n':int(m.sum()),'recall':float(np.mean(pr[m]==c)) if m.any() else np.nan,'coverage':float(mk.sum()/m.sum()) if m.any() else np.nan,'retained_n':int(mk.sum()),'retained_recall':float(np.mean(pr[mk]==c)) if mk.any() else np.nan,'automatic_miss_rate_all_class':float((mk&(pr!=c)).sum()/m.sum()) if m.any() else np.nan})
 return rows
def risk_curve(p,y,score):
 pr=norm(p).argmax(1);o=np.argsort(score,kind='mergesort');cor=(pr[o]==y[o]).astype(float);k=np.arange(1,len(y)+1);return pd.DataFrame({'coverage':k/len(y),'risk':1-np.cumsum(cor)/k,'score':score[o]})
def reliability(p,y):
 p=norm(p);rows=[];ed=np.linspace(0,1,11);cf=p.max(1);pr=p.argmax(1);ok=pr==y
 for kind,prob,target in [('overall',cf,ok),('melanoma_vs_rest',p[:,MEL],y==MEL)]:
  for i in range(10):
   m=(prob>=ed[i])&((prob<ed[i+1]) if i<9 else (prob<=ed[i+1]));rows.append({'kind':kind,'bin_low':ed[i],'bin_high':ed[i+1],'n':int(m.sum()),'mean_probability':float(prob[m].mean()) if m.any() else np.nan,'observed_frequency':float(target[m].mean()) if m.any() else np.nan})
 return rows
def aggregate_lesion(frame,p):
 ys=np.asarray([IDX[x] for x in frame.ham_class]);ps=[];yy=[];lids=frame.lesion_id.astype(str).to_numpy()
 for lid in np.unique(lids):
  ix=np.where(lids==lid)[0];labs=np.unique(ys[ix])
  if len(labs)==1:ps.append(p[ix].mean(0));yy.append(labs[0])
 return norm(np.asarray(ps)),np.asarray(yy,int)
def main():
 print('HIBA_EXTERNAL_START',flush=True);HROOT.mkdir(parents=True,exist_ok=True)
 if not META.exists():urllib.request.urlretrieve(META_URL,META)
 if not BUNDLE.exists():print('HIBA_DOWNLOAD_START',flush=True);urllib.request.urlretrieve(ZIP_URL,BUNDLE);print('HIBA_DOWNLOAD_DONE|'+str(BUNDLE.stat().st_size),flush=True)
 if not EXTRACT.exists() or not any(EXTRACT.rglob('*.jpg')):
  EXTRACT.mkdir(parents=True,exist_ok=True);print('HIBA_EXTRACT_START',flush=True);zipfile.ZipFile(BUNDLE).extractall(EXTRACT);print('HIBA_EXTRACT_DONE',flush=True)
 df=pd.read_csv(META);df['diagnosis_norm']=df.diagnosis.astype(str).str.strip().str.lower();df['ham_class']=df.diagnosis_norm.map(DX_MAP);mapping=df.groupby(['diagnosis_norm','ham_class'],dropna=False).size().reset_index(name='n');mapping.to_csv(OUT/'hiba_taxonomy_mapping_all_images.csv',index=False)
 use=df[(df.image_type.astype(str).str.lower()=='dermoscopic')&df.ham_class.notna()].copy().reset_index(drop=True);excluded=df[~((df.image_type.astype(str).str.lower()=='dermoscopic')&df.ham_class.notna())][['isic_id','image_type','diagnosis','lesion_id','patient_id']].copy();excluded.to_csv(OUT/'hiba_exclusions.csv',index=False)
 paths={p.stem:p for p in EXTRACT.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}};miss=[x for x in use.isic_id if x not in paths]
 if miss:raise FileNotFoundError(f'{len(miss)} HIBA images missing, e.g. {miss[:5]}')
 primary={'images':int(len(use)),'lesions':int(use.lesion_id.nunique()),'patients':int(use.patient_id.nunique()),'class_counts':{k:int(v) for k,v in use.ham_class.value_counts().items()},'excluded_images':int(len(excluded))};print('HIBA_PRIMARY|'+json.dumps(primary),flush=True)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('HIBA_DEVICE|'+str(device),flush=True);ld=DataLoader(HibaDataset(use,paths),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
 members=[];y=None
 for s in SEEDS:
  m=model_for(s,device);p,y,ids=predict_det(m,ld,device);members.append(p);print('HIBA_MEMBER_DONE|'+str(s),flush=True);del m
  if torch.cuda.is_available():torch.cuda.empty_cache()
 members=np.stack(members,1);ens=norm(members.mean(1));les=use.lesion_id.astype(str).to_numpy()
 final=np.load(BASE/'final_stage'/'deep_ensemble_predictions.npz',allow_pickle=False);iv=norm(npz_pick(final,['val_member_probs']).mean(1));z11=load_internal(11);ivy=np.asarray(npz_pick(z11,['val_labels','validation_labels']),int);ens_thr=threshold_for_coverage(1-iv.max(1),.60);temp=fit_temperature(iv,ivy);ivc=apply_temperature(iv,temp);ens_cal_thr=threshold_for_coverage(1-ivc.max(1),.60);ens_safety10=constrained_threshold(iv,ivy,1-iv.max(1),.10)
 m=model_for(2026,device);mc_ext,_,_=predict_mc(m,ld,device,30);del m
 if torch.cuda.is_available():torch.cuda.empty_cache()
 mp_ext,ms_ext=mc_scores(mc_ext);z26=load_internal(2026);ivy26=np.asarray(npz_pick(z26,['val_labels','validation_labels']),int);ivmc=orient_mc(npz_pick(z26,['val_mc_probs','validation_mc_probs']),len(ivy26));ivmp,ivms=mc_scores(ivmc);mc_thr=threshold_for_coverage(ivms['pe'],.60);mc_safety10=constrained_threshold(ivmp,ivy26,ivms['pe'],.10)
 calens=apply_temperature(ens,temp);methods={'deep_ensemble_msp':(ens,1-ens.max(1),ens_thr),'deep_ensemble_calibrated_msp':(calens,1-calens.max(1),ens_cal_thr),'mc_dropout_pe_seed2026':(mp_ext,ms_ext['pe'],mc_thr)}
 summary={'source':{'name':'HIBA Skin Lesions','doi':'10.34970/559884','official_images':1635,'primary_modality':'dermoscopic'},'mapping':mapping.fillna('EXCLUDED').to_dict('records'),'primary_cohort':primary,'frozen_internal':{'ensemble_threshold_60':ens_thr,'ensemble_temperature':temp,'ensemble_calibrated_threshold_60':ens_cal_thr,'ensemble_safety10':ens_safety10,'mc_pe_threshold_60':mc_thr,'mc_safety10':mc_safety10},'methods':{}}
 cirows=[];pcrows=[];relrows=[]
 for j,(name,(p,score,thr)) in enumerate(methods.items()):
  d=metrics(p,y,score);d['frozen_60']=safety(p,y,score,thr);lp,ly=aggregate_lesion(use,p);ls=1-lp.max(1);d['lesion_level']={'n':int(len(ly)),'accuracy':float(accuracy_score(ly,lp.argmax(1))),'macro_f1':float(f1_score(ly,lp.argmax(1),average='macro',zero_division=0)),'melanoma_recall':float(recall_score(ly==MEL,lp.argmax(1)==MEL,zero_division=0)),'aurc_msp':aurc(ls,lp.argmax(1),ly),'eaurc_msp':eaurc(ls,lp.argmax(1),ly)};summary['methods'][name]=d
  x=cluster_boot(p,y,score,les,thr,1000,20260811+j);[r.update({'method':name}) for r in x];cirows+=x;x=per_class(p,y,score,thr);[r.update({'method':name}) for r in x];pcrows+=x;x=reliability(p,y);[r.update({'method':name}) for r in x];relrows+=x;risk_curve(p,y,score).to_csv(OUT/f'risk_coverage_{name}.csv',index=False)
 if ens_safety10:summary['methods']['deep_ensemble_msp']['frozen_safety10']=safety(ens,y,1-ens.max(1),ens_safety10[0])
 if mc_safety10:summary['methods']['mc_dropout_pe_seed2026']['frozen_safety10']=safety(mp_ext,y,ms_ext['pe'],mc_safety10[0])
 sg=[];epr=ens.argmax(1)
 for col in ['sex','anatom_site_general','diagnosis_confirm_type','fitzpatrick_skin_type']:
  for val,g in use.groupby(col,dropna=False):
   ix=g.index.to_numpy()
   if len(ix)>=20:sg.append({'subgroup_variable':col,'subgroup':str(val),'n':len(ix),'accuracy':float(accuracy_score(y[ix],epr[ix])),'macro_f1':float(f1_score(y[ix],epr[ix],average='macro',zero_division=0)),'melanoma_n':int((y[ix]==MEL).sum()),'melanoma_recall':float(recall_score(y[ix]==MEL,epr[ix]==MEL,zero_division=0)) if (y[ix]==MEL).any() else np.nan,'ece15':ece(ens[ix],y[ix])})
 pd.DataFrame(cirows).to_csv(OUT/'hiba_lesion_cluster_ci.csv',index=False);pd.DataFrame(pcrows).to_csv(OUT/'hiba_class_specific_referral.csv',index=False);pd.DataFrame(relrows).to_csv(OUT/'hiba_reliability_bins.csv',index=False);pd.DataFrame(sg).to_csv(OUT/'hiba_subgroup_robustness.csv',index=False);use[['isic_id','lesion_id','patient_id','image_type','diagnosis','ham_class','sex','age_approx','anatom_site_general','diagnosis_confirm_type','fitzpatrick_skin_type']].to_csv(OUT/'hiba_primary_cohort.csv',index=False)
 np.savez_compressed(OUT/'hiba_predictions.npz',image_ids=use.isic_id.astype(str).to_numpy(),labels=y,ensemble_member_probs=members.astype(np.float32),ensemble_probs=ens.astype(np.float32),mc_seed2026_probs=mc_ext.astype(np.float32));(OUT/'hiba_external_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
 with tarfile.open(ARCHIVE,'w:gz') as tf:tf.add(OUT,arcname='external_hiba')
 h=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest();SHAF.write_text(f'{h}  {ARCHIVE.name}\n');print('HIBA_EXTERNAL_SUMMARY|'+json.dumps({'archive':str(ARCHIVE),'size':ARCHIVE.stat().st_size,'sha256':h,'primary':primary,'ensemble':summary['methods']['deep_ensemble_msp'],'mc':summary['methods']['mc_dropout_pe_seed2026']}),flush=True);print('PAPER2_HIBA_EXTERNAL_DONE',flush=True)
if __name__=='__main__':main()
