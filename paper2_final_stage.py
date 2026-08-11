import json, math, os, re, tarfile, hashlib, random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, average_precision_score

BASE = Path('/kaggle/working/paper2_data')
IMG_DIR = BASE / 'HAM10000_images'
OUT = BASE / 'final_stage'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [11, 29, 47, 71, 2026]
CLASS_NAMES = ['akiec','bcc','bkl','df','mel','nv','vasc']
MEL_IDX = CLASS_NAMES.index('mel')
EPS = 1e-12


def sha256_file(path, chunk=8*1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def pick(npz, names):
    for n in names:
        if n in npz.files:
            return npz[n]
    return None


def normalize_mc(arr, n):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape[0] != n: raise ValueError(f'2D prob length mismatch {arr.shape} vs {n}')
        return arr[:, None, :]
    if arr.ndim != 3: raise ValueError(f'Expected 2D/3D probs, got {arr.shape}')
    if arr.shape[0] == n: return arr
    if arr.shape[1] == n: return arr.transpose(1,0,2)
    raise ValueError(f'Cannot orient MC probs {arr.shape} for n={n}')


def locate_seed_dir(seed):
    if seed == 2026:
        p = BASE / 'grouped_mc_seed2026'
        if p.exists(): return p
    root = BASE / 'multiseed_fixed_split'
    candidates = []
    if root.exists():
        for p in [root] + [q for q in root.rglob('*') if q.is_dir()]:
            s = str(p).lower()
            if re.search(rf'(?<!\d){seed}(?!\d)', s):
                candidates.append(p)
    for p in candidates:
        if list(p.glob('*.npz')) and list(p.glob('*.pt')):
            return p
    # broader fallback
    for p in [q for q in BASE.rglob('*') if q.is_dir()]:
        if re.search(rf'(?<!\d){seed}(?!\d)', str(p)) and list(p.glob('*.npz')) and list(p.glob('*.pt')):
            return p
    raise FileNotFoundError(f'No complete seed directory found for seed {seed}')


def choose_file(d, patterns):
    for pat in patterns:
        x = list(d.glob(pat))
        if x: return x[0]
    return None


def load_seed(seed):
    d = locate_seed_dir(seed)
    npz_path = choose_file(d, ['mc_predictions.npz','paper2_grouped_mc_predictions.npz','*predictions*.npz','*.npz'])
    ckpt = choose_file(d, ['best.pt','model_best_val_macro_f1.pt','*best*.pt','*.pt'])
    summary = choose_file(d, ['summary.json','mc_summary.json','*summary*.json'])
    if npz_path is None: raise FileNotFoundError(f'No npz in {d}')
    z = np.load(npz_path, allow_pickle=False)
    vy = pick(z, ['val_labels','validation_labels','val_y','y_val'])
    ty = pick(z, ['id_labels','test_labels','test_y','y_test'])
    vi = pick(z, ['val_image_ids','validation_image_ids','val_ids'])
    ti = pick(z, ['id_image_ids','test_image_ids','test_ids'])
    vp = pick(z, ['val_mc_probs','validation_mc_probs','val_probs','validation_probs'])
    tp = pick(z, ['id_mc_probs','test_mc_probs','test_probs','id_probs'])
    if any(x is None for x in [vy,ty,vp,tp]):
        raise KeyError(f'Unexpected NPZ keys for seed {seed}: {z.files}')
    vy = np.asarray(vy, dtype=np.int64); ty = np.asarray(ty, dtype=np.int64)
    vp = normalize_mc(vp, len(vy)).astype(np.float64)
    tp = normalize_mc(tp, len(ty)).astype(np.float64)
    if vi is None: vi = np.asarray([f'val_{i}' for i in range(len(vy))])
    if ti is None: ti = np.asarray([f'test_{i}' for i in range(len(ty))])
    vi = np.asarray(vi).astype(str); ti = np.asarray(ti).astype(str)
    return {'seed':seed,'dir':d,'npz':npz_path,'ckpt':ckpt,'summary':summary,
            'val_mc':vp,'test_mc':tp,'val_y':vy,'test_y':ty,'val_ids':vi,'test_ids':ti,
            'npz_keys':list(z.files)}


def align_to(ref_ids, ids, arrays):
    if np.array_equal(ref_ids, ids): return arrays
    pos = {x:i for i,x in enumerate(ids.tolist())}
    idx = np.asarray([pos[x] for x in ref_ids.tolist()], dtype=int)
    return [a[idx] for a in arrays]


def entropy(p):
    p = np.clip(p, EPS, 1.0)
    return -(p*np.log(p)).sum(axis=-1)


def uncertainty_scores(mc):
    meanp = mc.mean(axis=1)
    pred_ent = entropy(meanp)
    exp_ent = entropy(mc).mean(axis=1)
    mi = np.maximum(pred_ent-exp_ent, 0.0)
    msp = 1.0-meanp.max(axis=1)
    votes = mc.argmax(axis=2)
    counts = np.stack([(votes==c).sum(axis=1) for c in range(mc.shape[2])], axis=1)
    vr = 1.0-counts.max(axis=1)/mc.shape[1]
    return {'mutual_information':mi,'one_minus_msp':msp,'predictive_entropy':pred_ent,
            'expected_entropy':exp_ent,'variation_ratio':vr}, meanp


def nll(p,y):
    p = np.clip(p, EPS, 1.0)
    return float(-np.log(p[np.arange(len(y)),y]).mean())


def brier(p,y):
    one = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p-one)**2, axis=1)))


def ece(p,y,bins=15):
    conf = p.max(axis=1); pred=p.argmax(axis=1); ok=(pred==y).astype(float)
    edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        mask=(conf>=lo)&((conf<hi) if i<bins-1 else (conf<=hi))
        if mask.any(): out += mask.mean()*abs(ok[mask].mean()-conf[mask].mean())
    return float(out)


def aurc_from_score(score,pred,y):
    order=np.argsort(score, kind='stable')
    err=(pred[order]!=y[order]).astype(float)
    risks=np.cumsum(err)/np.arange(1,len(err)+1)
    cov=np.arange(1,len(err)+1)/len(err)
    return float(np.trapz(risks,cov))


def error_detection(score,pred,y):
    err=(pred!=y).astype(int)
    if err.min()==err.max(): return float('nan'),float('nan')
    return float(roc_auc_score(err,score)), float(average_precision_score(err,score))


def threshold_for_coverage(score,target):
    k=max(1,min(len(score),int(math.ceil(target*len(score)))))
    return float(np.sort(score)[k-1])


def selective_metrics(score,p,y,thr):
    keep=score<=thr
    pred=p.argmax(1); total=len(y)
    if not keep.any(): return {'coverage':0.0,'selective_accuracy':float('nan'),'selective_macro_f1':float('nan')}
    ym=y[keep]; pm=pred[keep]
    mel_all=(y==MEL_IDX); mel_keep=keep & mel_all
    full_fn=mel_all & (pred!=MEL_IDX)
    retained_fn=full_fn & keep
    return {
      'coverage':float(keep.mean()), 'retained_n':int(keep.sum()), 'referred_n':int((~keep).sum()),
      'selective_accuracy':float(accuracy_score(ym,pm)),
      'selective_macro_f1':float(f1_score(ym,pm,average='macro',zero_division=0)),
      'melanoma_coverage':float(mel_keep.sum()/max(1,mel_all.sum())),
      'melanoma_retained_n':int(mel_keep.sum()),
      'melanoma_selective_recall':float(recall_score(ym==MEL_IDX,pm==MEL_IDX,zero_division=0)) if mel_keep.any() else float('nan'),
      'melanoma_full_false_negatives':int(full_fn.sum()),
      'melanoma_retained_false_negatives':int(retained_fn.sum()),
      'melanoma_fn_escape_rate':float(retained_fn.sum()/max(1,full_fn.sum()))
    }


def basic_metrics(p,y):
    pred=p.argmax(1)
    scores_stub={}
    return {
      'n':int(len(y)), 'accuracy':float(accuracy_score(y,pred)),
      'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
      'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
      'melanoma_recall':float(recall_score(y==MEL_IDX,pred==MEL_IDX,zero_division=0)),
      'nll':nll(p,y),'brier':brier(p,y),'ece15':ece(p,y,15)
    }


def summarize_seed(seed_data):
    vs,vp=uncertainty_scores(seed_data['val_mc'])
    ts,tp=uncertainty_scores(seed_data['test_mc'])
    base=basic_metrics(tp,seed_data['test_y'])
    val_rows={}; test_rows={}
    for name in vs:
        vpred=vp.argmax(1); tpred=tp.argmax(1)
        vau=aurc_from_score(vs[name],vpred,seed_data['val_y'])
        tau=aurc_from_score(ts[name],tpred,seed_data['test_y'])
        auroc,auprc=error_detection(ts[name],tpred,seed_data['test_y'])
        val_rows[name]={'aurc':vau}
        test_rows[name]={'aurc':tau,'error_auroc':auroc,'error_auprc':auprc}
    return base,val_rows,test_rows,vs,vp,ts,tp


def power_temp_samples(samples,T):
    lp=np.log(np.clip(samples,EPS,1.0))/T
    lp=lp-lp.max(axis=-1,keepdims=True)
    q=np.exp(lp); q/=q.sum(axis=-1,keepdims=True)
    return q


def fit_temperature(samples,y):
    try:
        from scipy.optimize import minimize_scalar
        def obj(T): return nll(power_temp_samples(samples,float(T)).mean(axis=1),y)
        res=minimize_scalar(obj,bounds=(0.1,8.0),method='bounded',options={'xatol':1e-4})
        return float(res.x),float(res.fun)
    except Exception:
        grid=np.exp(np.linspace(np.log(0.15),np.log(6.0),120))
        vals=[nll(power_temp_samples(samples,t).mean(axis=1),y) for t in grid]
        j=int(np.argmin(vals)); return float(grid[j]),float(vals[j])


def aggregate_rows(df, metric_cols):
    rows=[]
    for c in metric_cols:
        x=pd.to_numeric(df[c],errors='coerce').dropna().values
        rows.append({'metric':c,'mean':float(np.mean(x)),'sd':float(np.std(x,ddof=1)) if len(x)>1 else 0.0,'min':float(np.min(x)),'max':float(np.max(x)),'n':int(len(x))})
    return pd.DataFrame(rows)


def make_model_and_load(ckpt_path, device):
    import torch
    import torch.nn as nn
    from torchvision import models
    class M(nn.Module):
        def __init__(self):
            super().__init__(); base=models.resnet50(weights=None); inf=base.fc.in_features; base.fc=nn.Identity()
            self.backbone=base; self.fc1=nn.Linear(inf,512); self.relu=nn.ReLU(inplace=True); self.dropout=nn.Dropout(p=0.2); self.fc2=nn.Linear(512,7)
        def forward(self,x): return self.fc2(self.dropout(self.relu(self.fc1(self.backbone(x)))))
    model=M().to(device)
    obj=torch.load(ckpt_path,map_location=device,weights_only=False)
    state=obj.get('model_state',obj.get('state_dict',obj)) if isinstance(obj,dict) else obj
    state={k.replace('module.','',1) if k.startswith('module.') else k:v for k,v in state.items()}
    model.load_state_dict(state,strict=True); model.eval(); return model


def deterministic_member_probs(seed_data, device, split):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from torchvision import transforms
    from PIL import Image
    ids=seed_data[f'{split}_ids']; y=seed_data[f'{split}_y']
    class D(Dataset):
        def __len__(self): return len(ids)
        def __getitem__(self,i):
            with Image.open(IMG_DIR/f'{ids[i]}.jpg') as im: x=im.convert('RGB')
            return tfm(x),int(y[i])
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    loader=DataLoader(D(),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
    model=make_model_and_load(seed_data['ckpt'],device)
    out=[]
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device,non_blocking=True)
            with torch.amp.autocast('cuda',enabled=(device.type=='cuda')): logits=model(x)
            out.append(torch.softmax(logits.float(),1).cpu().numpy())
    return np.concatenate(out,0).astype(np.float64)


def extra_mc_passes(seed_data, device, split, n_passes=20):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from torchvision import transforms
    from PIL import Image
    ids=seed_data[f'{split}_ids']; y=seed_data[f'{split}_y']
    class D(Dataset):
        def __len__(self): return len(ids)
        def __getitem__(self,i):
            with Image.open(IMG_DIR/f'{ids[i]}.jpg') as im: x=im.convert('RGB')
            return tfm(x),int(y[i])
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    loader=DataLoader(D(),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
    model=make_model_and_load(seed_data['ckpt'],device)
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout): m.train()
    passes=[]
    with torch.no_grad():
        for t in range(n_passes):
            buf=[]
            for x,_ in loader:
                x=x.to(device,non_blocking=True)
                with torch.amp.autocast('cuda',enabled=(device.type=='cuda')): logits=model(x)
                buf.append(torch.softmax(logits.float(),1).cpu().numpy())
            passes.append(np.concatenate(buf,0)); print(f'EXTRA_MC|{split}|{t+1}/{n_passes}',flush=True)
    return np.stack(passes,1).astype(np.float64)


def risk_curve(score,p,y):
    order=np.argsort(score,kind='stable'); err=(p.argmax(1)[order]!=y[order]).astype(float)
    risk=np.cumsum(err)/np.arange(1,len(err)+1); cov=np.arange(1,len(err)+1)/len(err)
    return cov,risk


def main():
    print('FINAL_STAGE_START',flush=True)
    seeds={s:load_seed(s) for s in SEEDS}
    ref=seeds[2026]
    for s,d in seeds.items():
        if s==2026: continue
        d['val_mc'],d['val_y'] = align_to(ref['val_ids'],d['val_ids'],[d['val_mc'],d['val_y']]); d['val_ids']=ref['val_ids'].copy()
        d['test_mc'],d['test_y'] = align_to(ref['test_ids'],d['test_ids'],[d['test_mc'],d['test_y']]); d['test_ids']=ref['test_ids'].copy()
        if not np.array_equal(d['val_y'],ref['val_y']) or not np.array_equal(d['test_y'],ref['test_y']): raise RuntimeError(f'label mismatch seed {s}')

    inventory=[]; per_seed=[]; val_auc_rows=[]; test_uq_rows=[]; cache={}
    for s,d in seeds.items():
        base,va,ta,vs,vp,ts,tp=summarize_seed(d); cache[s]=(vs,vp,ts,tp)
        row={'seed':s,**base}
        for name in va:
            val_auc_rows.append({'seed':s,'score':name,'validation_aurc':va[name]['aurc']})
            test_uq_rows.append({'seed':s,'score':name,**ta[name]})
        per_seed.append(row)
        inventory.append({'seed':s,'directory':str(d['dir']),'npz':str(d['npz']),'checkpoint':str(d['ckpt']) if d['ckpt'] else None,'npz_keys':d['npz_keys'],'val_shape':list(d['val_mc'].shape),'test_shape':list(d['test_mc'].shape)})
        print('SEED_METRICS|'+json.dumps(row,sort_keys=True),flush=True)
    per_seed_df=pd.DataFrame(per_seed); per_seed_df.to_csv(OUT/'multiseed_metrics.csv',index=False)
    val_auc_df=pd.DataFrame(val_auc_rows); val_auc_df.to_csv(OUT/'validation_uncertainty_auc.csv',index=False)
    pd.DataFrame(test_uq_rows).to_csv(OUT/'test_uncertainty_metrics_by_seed.csv',index=False)
    agg=aggregate_rows(per_seed_df,['accuracy','macro_f1','weighted_f1','melanoma_recall','nll','brier','ece15']); agg.to_csv(OUT/'multiseed_mean_sd.csv',index=False)

    score_mean=val_auc_df.groupby('score')['validation_aurc'].mean().sort_values()
    primary=str(score_mean.index[0])
    pd.DataFrame({'score':score_mean.index,'mean_validation_aurc':score_mean.values}).to_csv(OUT/'validation_score_selection.csv',index=False)
    sel_rows=[]
    for s,d in seeds.items():
        vs,vp,ts,tp=cache[s]; thr=threshold_for_coverage(vs[primary],0.60); m=selective_metrics(ts[primary],tp,d['test_y'],thr)
        sel_rows.append({'seed':s,'primary_score':primary,'val_threshold':thr,**m})
    sel_df=pd.DataFrame(sel_rows); sel_df.to_csv(OUT/'multiseed_selective_60.csv',index=False)
    sel_agg=aggregate_rows(sel_df,['coverage','selective_accuracy','selective_macro_f1','melanoma_coverage','melanoma_selective_recall','melanoma_fn_escape_rate']); sel_agg.to_csv(OUT/'multiseed_selective_60_mean_sd.csv',index=False)

    # Deterministic 5-member deep ensemble
    import torch
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE|'+str(device),flush=True)
    val_members=[]; test_members=[]; ensemble_status='complete'
    det_errors={}
    for s,d in seeds.items():
        try:
            if d['ckpt'] is None: raise FileNotFoundError('checkpoint missing')
            vp=deterministic_member_probs(d,device,'val'); tp=deterministic_member_probs(d,device,'test')
            val_members.append(vp); test_members.append(tp); print(f'DEEP_MEMBER_DONE|seed={s}',flush=True)
        except Exception as e:
            ensemble_status='failed'; det_errors[str(s)]=repr(e); print(f'DEEP_MEMBER_ERROR|seed={s}|{repr(e)}',flush=True); break
    ensemble_results={'status':ensemble_status,'errors':det_errors}
    calibration_rows=[]
    if ensemble_status=='complete' and len(val_members)==5:
        vmem=np.stack(val_members,1); tmem=np.stack(test_members,1)
        ven=vmem.mean(1); ten=tmem.mean(1)
        ensemble_results['deterministic_deep_ensemble']=basic_metrics(ten,ref['test_y'])
        es,_=uncertainty_scores(tmem); epred=ten.argmax(1)
        ensemble_results['uncertainty']={name:{'aurc':aurc_from_score(sc,epred,ref['test_y']),'error_auroc':error_detection(sc,epred,ref['test_y'])[0],'error_auprc':error_detection(sc,epred,ref['test_y'])[1]} for name,sc in es.items()}
        Tdeep,_=fit_temperature(vmem,ref['val_y']); tcal=power_temp_samples(tmem,Tdeep).mean(1)
        ensemble_results['temperature']=Tdeep; ensemble_results['calibrated']=basic_metrics(tcal,ref['test_y'])
        calibration_rows += [
            {'method':'deep_ensemble','calibrated':False,'temperature':1.0,**basic_metrics(ten,ref['test_y'])},
            {'method':'deep_ensemble','calibrated':True,'temperature':Tdeep,**basic_metrics(tcal,ref['test_y'])}
        ]
        np.savez_compressed(OUT/'deep_ensemble_predictions.npz',val_member_probs=vmem.astype(np.float32),test_member_probs=tmem.astype(np.float32),val_labels=ref['val_y'],test_labels=ref['test_y'],val_image_ids=ref['val_ids'],test_image_ids=ref['test_ids'])

    # MC Dropout temperature scaling, seed 2026, protocol T=30
    mc_v=ref['val_mc'][:,:30,:]; mc_t=ref['test_mc'][:,:30,:]
    Tmc,_=fit_temperature(mc_v,ref['val_y']); mc_unc=mc_t.mean(1); mc_cal_samples=power_temp_samples(mc_t,Tmc); mc_cal=mc_cal_samples.mean(1)
    calibration_rows += [
      {'method':'mc_dropout_seed2026_T30','calibrated':False,'temperature':1.0,**basic_metrics(mc_unc,ref['test_y'])},
      {'method':'mc_dropout_seed2026_T30','calibrated':True,'temperature':Tmc,**basic_metrics(mc_cal,ref['test_y'])}
    ]
    pd.DataFrame(calibration_rows).to_csv(OUT/'calibration_comparison.csv',index=False)
    calibration_detail={'mc_dropout_shared_temperature':Tmc,'deep_ensemble_shared_temperature':ensemble_results.get('temperature',None)}

    # T sensitivity: append 20 independent stochastic passes to the saved T=30 seed-2026 samples.
    extra_v=extra_mc_passes(ref,device,'val',20); extra_t=extra_mc_passes(ref,device,'test',20)
    v50=np.concatenate([ref['val_mc'][:,:30,:],extra_v],axis=1); t50=np.concatenate([ref['test_mc'][:,:30,:],extra_t],axis=1)
    np.savez_compressed(OUT/'seed2026_mc_T50_predictions.npz',val_mc_probs=v50.astype(np.float32),test_mc_probs=t50.astype(np.float32),val_labels=ref['val_y'],test_labels=ref['test_y'],val_image_ids=ref['val_ids'],test_image_ids=ref['test_ids'])
    trows=[]
    for T in [5,10,20,30,50]:
        vv=v50[:,:T,:]; tt=t50[:,:T,:]; vs,vp=uncertainty_scores(vv); ts,tp=uncertainty_scores(tt)
        row={'T':T,**basic_metrics(tp,ref['test_y'])}
        for name in ['mutual_information','one_minus_msp','predictive_entropy']:
            row[f'{name}_aurc']=aurc_from_score(ts[name],tp.argmax(1),ref['test_y'])
            row[f'{name}_error_auroc']=error_detection(ts[name],tp.argmax(1),ref['test_y'])[0]
        thr=threshold_for_coverage(vs[primary],0.60); sm=selective_metrics(ts[primary],tp,ref['test_y'],thr)
        row.update({f'sel_{k}':v for k,v in sm.items() if k in ['coverage','selective_accuracy','selective_macro_f1','melanoma_coverage','melanoma_fn_escape_rate']})
        trows.append(row)
    pd.DataFrame(trows).to_csv(OUT/'mc_T_sensitivity.csv',index=False)

    # Risk-coverage figure for MC seed2026 and deterministic deep ensemble if available.
    try:
        import matplotlib.pyplot as plt
        s_mc,_=uncertainty_scores(mc_t); c,r=risk_curve(s_mc[primary],mc_unc,ref['test_y'])
        fig=plt.figure(figsize=(6.4,4.4)); plt.plot(c,r,label=f'MC Dropout ({primary})')
        if ensemble_status=='complete':
            s_de,_=uncertainty_scores(tmem); c2,r2=risk_curve(s_de['mutual_information'],ten,ref['test_y']); plt.plot(c2,r2,label='Deep ensemble (member MI)')
        plt.xlabel('Coverage'); plt.ylabel('Selective risk (error rate)'); plt.title('Risk–coverage comparison'); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); fig.savefig(OUT/'risk_coverage_final.png',dpi=220); plt.close(fig)
    except Exception as e: print('FIGURE_WARNING|'+repr(e),flush=True)

    with open(OUT/'model_inventory.json','w') as f: json.dump(inventory,f,indent=2)
    with open(OUT/'ensemble_results.json','w') as f: json.dump(ensemble_results,f,indent=2)
    with open(OUT/'calibration_parameters.json','w') as f: json.dump(calibration_detail,f,indent=2)

    summary={
      'seeds':SEEDS,
      'primary_score_selected_by_lowest_mean_validation_aurc':primary,
      'mean_validation_aurc_by_score':{k:float(v) for k,v in score_mean.items()},
      'multiseed_mean_sd':agg.to_dict(orient='records'),
      'selective_60_mean_sd':sel_agg.to_dict(orient='records'),
      'deep_ensemble':ensemble_results,
      'temperature_scaling':calibration_detail,
      'mc_T_sensitivity':trows
    }
    with open(OUT/'final_stage_summary.json','w') as f: json.dump(summary,f,indent=2)

    archive=BASE/'Paper2_FINAL_STAGE_RESULTS.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): tf.add(p,arcname=str(Path('final_stage')/p.relative_to(OUT)))
    sha=sha256_file(archive)
    (BASE/'Paper2_FINAL_STAGE_RESULTS.sha256').write_text(f'{sha}  {archive.name}\n')
    print('FINAL_STAGE_SUMMARY_BEGIN',flush=True); print(json.dumps(summary,sort_keys=True),flush=True); print('FINAL_STAGE_SUMMARY_END',flush=True)
    print(f'FINAL_STAGE_ARCHIVE|{archive}|size={archive.stat().st_size}|sha256={sha}',flush=True)
    print('PAPER2_FINAL_STAGE_DONE',flush=True)

if __name__=='__main__':
    main()
