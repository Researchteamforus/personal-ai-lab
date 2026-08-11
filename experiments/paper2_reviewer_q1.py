import json,re,tarfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
BASE=Path('/kaggle/working/paper2_data'); OUT=BASE/'reviewer_q1_revision'; OUT.mkdir(parents=True,exist_ok=True)
FINAL=BASE/'final_stage'; SPLIT=BASE/'splits'/'ham10000_lesion_group_split_seed2026.csv'; SEEDS=[11,29,47,71,2026]
CLS=['akiec','bcc','bkl','df','mel','nv','vasc']; MEL=4; EPS=1e-12; B=500; COV=.60
ARCH=BASE/'Paper2_REVIEWER_Q1_RESULTS.tar.gz'; SHAF=BASE/'Paper2_REVIEWER_Q1_RESULTS.sha256'
def norm(p):
 p=np.clip(np.asarray(p,float),EPS,1); return p/p.sum(-1,keepdims=True)
def ent(p): p=np.clip(p,EPS,1); return -(p*np.log(p)).sum(-1)
def pick(z,names):
 for n in names:
  if n in z.files:return z[n]
def orient(a,n):
 a=np.asarray(a)
 if a.ndim==2:return a[:,None,:]
 if a.shape[0]==n:return a
 if a.shape[1]==n:return a.transpose(1,0,2)
 raise ValueError(a.shape)
def seed_dir(s):
 if s==2026:return BASE/'grouped_mc_seed2026'
 root=BASE/'multiseed_fixed_split'
 for d in [root]+[x for x in root.rglob('*') if x.is_dir()]:
  if re.search(rf'(?<!\d){s}(?!\d)',str(d)) and list(d.glob('*.npz')):return d
 raise FileNotFoundError(s)
def load_seed(s):
 d=seed_dir(s); fs=list(d.glob('mc_predictions.npz'))+list(d.glob('paper2_grouped_mc_predictions.npz'))+list(d.glob('*predictions*.npz'))
 z=np.load(fs[0],allow_pickle=False); vy=pick(z,['val_labels','validation_labels']); ty=pick(z,['id_labels','test_labels']); vi=pick(z,['val_image_ids']); ti=pick(z,['id_image_ids','test_image_ids']); vp=pick(z,['val_mc_probs']); tp=pick(z,['id_mc_probs','test_mc_probs'])
 vy=np.asarray(vy,int);ty=np.asarray(ty,int);return {'s':s,'vy':vy,'ty':ty,'vi':np.asarray(vi).astype(str),'ti':np.asarray(ti).astype(str),'v':orient(vp,len(vy)),'t':orient(tp,len(ty))}
def align(ref,ids,a):
 if np.array_equal(ref,ids):return a
 pos={x:i for i,x in enumerate(ids.tolist())}; return a[np.asarray([pos[x] for x in ref])]
def mc_scores(mc):
 mc=norm(mc); p=mc.mean(1); pe=ent(p);ee=ent(mc).mean(1);mi=np.maximum(pe-ee,0);votes=mc.argmax(2);cnt=np.stack([(votes==c).sum(1) for c in range(7)],1);vr=1-cnt.max(1)/mc.shape[1]
 return p,{'pe':pe,'ee':ee,'mi':mi,'msp':1-p.max(1),'vr':vr}
def det_scores(p):p=norm(p);return {'msp':1-p.max(1),'pe':ent(p)}
def aurc_ord(c):
 c=np.asarray(c,float);k=np.arange(1,len(c)+1);risk=1-np.cumsum(c)/k;cov=k/len(c);return float(np.trapezoid(np.r_[risk[0],risk],np.r_[0,cov]))
def aurc(sc,pred,y):o=np.argsort(sc,kind='mergesort');return aurc_ord(pred[o]==y[o])
def eaurc(sc,pred,y):return aurc(sc,pred,y)-aurc_ord(np.sort((pred==y).astype(int))[::-1])
def ece(p,y,bins=15):
 p=norm(p);pr=p.argmax(1);cf=p.max(1);ok=(pr==y);ed=np.linspace(0,1,bins+1);v=0
 for i in range(bins):
  m=(cf>=ed[i])&((cf<ed[i+1]) if i<bins-1 else (cf<=ed[i+1]));
  if m.any():v+=m.mean()*abs(ok[m].mean()-cf[m].mean())
 return float(v)
def nll(p,y):p=norm(p);return float(-np.log(p[np.arange(len(y)),y]).mean())
def brier(p,y):p=norm(p);return float(np.mean(np.sum((p-np.eye(7)[y])**2,1)))
def threshold(sc,c=.60):s=np.sort(sc,kind='mergesort');return float(s[max(0,min(len(s)-1,int(round(c*len(s)))-1))])
def safety(p,y,sc,th):
 p=norm(p);pr=p.argmax(1);k=sc<=th;mel=y==MEL;mk=k&mel;ff=mel&(pr!=MEL);rf=ff&k;clin=np.isin(y,[0,1,4]);clinfn=clin&~np.isin(pr,[0,1,4]);clinrf=clinfn&k; rn=int(mk.sum()); rfn=int(rf.sum()); mn=int(mel.sum())
 return {'coverage':float(k.mean()),'selective_accuracy':float(accuracy_score(y[k],pr[k])),'selective_macro_f1':float(f1_score(y[k],pr[k],average='macro',zero_division=0)),'melanoma_coverage':float(rn/mn),'retained_melanoma_n':rn,'retained_melanoma_tp':int((mk&(pr==MEL)).sum()),'retained_melanoma_fn':rfn,'retained_melanoma_sensitivity':float((mk&(pr==MEL)).sum()/rn) if rn else np.nan,'retained_melanoma_fnr':float(rfn/rn) if rn else np.nan,'automatic_melanoma_miss_rate':float(rfn/mn),'melanoma_fn_escape_rate':float(rfn/ff.sum()) if ff.sum() else 0.0,'clinically_important_auto_miss_rate':float(clinrf.sum()/clin.sum())}
def metrics(p,y,sc,th=None):
 p=norm(p);pr=p.argmax(1);err=(pr!=y).astype(int);d={'accuracy':float(accuracy_score(y,pr)),'macro_f1':float(f1_score(y,pr,average='macro',zero_division=0)),'weighted_f1':float(f1_score(y,pr,average='weighted',zero_division=0)),'melanoma_recall':float(np.mean(pr[y==MEL]==MEL)),'nll':nll(p,y),'brier':brier(p,y),'ece15':ece(p,y),'aurc':aurc(sc,pr,y),'eaurc':eaurc(sc,pr,y),'error_auroc':float(roc_auc_score(err,sc)),'error_auprc':float(average_precision_score(err,sc))}
 if th is not None:d.update({'sel_'+k:v for k,v in safety(p,y,sc,th).items()})
 return d
def lesion_ids(ids,df,split):
 vals=['validation','val'] if split=='validation' else ['test']; f=df[df['split'].astype(str).str.lower().isin(vals)];mp=dict(zip(f.image_id.astype(str),f.lesion_id.astype(str)));return np.asarray([mp[x] for x in ids])
def ci(v):a=np.asarray(v,float);a=a[np.isfinite(a)];return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]
def boot_pair(A,D,y,les,Bn=300,seed=1):
 u=np.unique(les);im={g:np.where(les==g)[0] for g in u};rng=np.random.default_rng(seed);pa,sa,ta=A;pd,sd,td=D;keys=['accuracy','macro_f1','melanoma_recall','aurc','eaurc','nll','brier','ece15','sel_selective_accuracy','sel_melanoma_coverage','sel_automatic_melanoma_miss_rate','sel_melanoma_fn_escape_rate'];vals={k:[] for k in keys};a0=metrics(pa,y,sa,ta);d0=metrics(pd,y,sd,td)
 for _ in range(Bn):
  samp=rng.choice(u,len(u),replace=True);ix=np.concatenate([im[g] for g in samp]);a=metrics(pa[ix],y[ix],sa[ix],ta);d=metrics(pd[ix],y[ix],sd[ix],td)
  for k in keys:vals[k].append(a[k]-d[k])
 return [{'metric':k,'delta_A_minus_B':a0[k]-d0[k],'ci_low':ci(vals[k])[0],'ci_high':ci(vals[k])[1]} for k in keys]
def boot_ci(p,y,sc,th,les,Bn=400,seed=7):
 u=np.unique(les);im={g:np.where(les==g)[0] for g in u};rng=np.random.default_rng(seed);vals={}
 for _ in range(Bn):
  samp=rng.choice(u,len(u),replace=True);ix=np.concatenate([im[g] for g in samp]);d=metrics(p[ix],y[ix],sc[ix],th)
  for k,v in d.items():
   if isinstance(v,(int,float,np.floating)) and np.isfinite(v):vals.setdefault(k,[]).append(v)
 pt=metrics(p,y,sc,th);return [{'metric':k,'point':pt.get(k,np.nan),'ci_low':ci(v)[0],'ci_high':ci(v)[1]} for k,v in vals.items()]
def fitT(p,y):
 from scipy.optimize import minimize_scalar
 p=norm(p)
 def f(t):return nll(applyT(p,t),y)
 return float(minimize_scalar(f,bounds=(.1,8),method='bounded').x)
def applyT(p,t):p=norm(p);z=np.log(p)/t;z-=z.max(1,keepdims=True);z=np.exp(z);return z/z.sum(1,keepdims=True)
def constrained(p,y,sc,maxmiss):
 p=norm(p);pr=p.argmax(1);mel=y==MEL;o=np.argsort(sc,kind='mergesort');best=None
 for k in range(1,len(y)+1):
  keep=np.zeros(len(y),bool);keep[o[:k]]=1;rate=float((keep&mel&(pr!=MEL)).sum()/mel.sum())
  if rate<=maxmiss+1e-12:best=(float(sc[o[k-1]]),k/len(y),rate)
 return best
def main():
 print('Q1_REVIEWER_START',flush=True);df=pd.read_csv(SPLIT);ss=[load_seed(s) for s in SEEDS];rv=ss[0]['vi'];rt=ss[0]['ti'];vy=ss[0]['vy'];ty=ss[0]['ty']
 for d in ss[1:]:d['v']=align(rv,d['vi'],d['v']);d['t']=align(rt,d['ti'],d['t']);d['vy']=align(rv,d['vi'],d['vy']);d['ty']=align(rt,d['ti'],d['ty']);d['vi']=rv;d['ti']=rt
 vl=lesion_ids(rv,df,'validation');tl=lesion_ids(rt,df,'test');z=np.load(FINAL/'deep_ensemble_predictions.npz',allow_pickle=False);ev=align(rv,z['val_image_ids'].astype(str),z['val_member_probs']);et=align(rt,z['test_image_ids'].astype(str),z['test_member_probs'])
 rows=[];pairs=[];mc=[]
 for j,d in enumerate(ss):
  vp,vs=mc_scores(d['v']);tp,ts=mc_scores(d['t']);mc.append((vp,vs,tp,ts));dv=norm(ev[:,j]);dt=norm(et[:,j]);dsv=det_scores(dv);dst=det_scores(dt);thd=threshold(dsv['msp']);thm=threshold(vs['pe']);md=metrics(dt,ty,dst['msp'],thd);mm=metrics(tp,ty,ts['pe'],thm);row={'seed':d['s']};row.update({'det_'+k:v for k,v in md.items()});row.update({'mc_'+k:v for k,v in mm.items()});rows.append(row);bp=boot_pair((dt,dst['msp'],thd),(tp,ts['pe'],thm),ty,tl,300,202600+d['s']);
  for r in bp:r['seed']=d['s'];pairs.append(r)
 pd.DataFrame(rows).to_csv(OUT/'deterministic_vs_mc_by_seed.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_det_msp_vs_mc_pe.csv',index=False);r=pd.DataFrame(rows);ag=[]
 for m in ['det','mc']:
  for k in ['accuracy','macro_f1','melanoma_recall','aurc','eaurc','error_auroc','error_auprc','nll','brier','ece15','sel_coverage','sel_selective_accuracy','sel_melanoma_coverage','sel_automatic_melanoma_miss_rate','sel_melanoma_fn_escape_rate']:
   x=r[f'{m}_{k}'];ag.append({'method':m,'metric':k,'mean':x.mean(),'sd':x.std(ddof=1)})
 pd.DataFrame(ag).to_csv(OUT/'deterministic_vs_mc_mean_sd.csv',index=False)
 ensv=norm(ev.mean(1));enst=norm(et.mean(1));esv=det_scores(ensv);est=det_scores(enst);the=threshold(esv['msp']);ens=metrics(enst,ty,est['msp'],the);T=fitT(ensv,vy);cv=applyT(ensv,T);ct=applyT(enst,T);csv=det_scores(cv);cst=det_scores(ct);thc=threshold(csv['msp']);cal=metrics(ct,ty,cst['msp'],thc);json.dump({'post_ensemble_temperature':T,'uncalibrated':ens,'post_ensemble_calibrated':cal},open(OUT/'ensemble_calibration.json','w'),indent=2)
 evm=[]
 for j,d in enumerate(ss):
  vp,vs,tp,ts=mc[j];thm=threshold(vs['pe']);bp=boot_pair((enst,est['msp'],the),(tp,ts['pe'],thm),ty,tl,300,207000+j)
  for q0 in bp:q0['mc_seed']=d['s'];evm.append(q0)
 pd.DataFrame(evm).to_csv(OUT/'paired_ensemble_vs_mc.csv',index=False)
 j=4;vp,vs,tp,ts=mc[j];dv=norm(ev[:,j]);dt=norm(et[:,j]);dsv=det_scores(dv);dst=det_scores(dt);methods={'Deterministic MSP seed2026':(dt,dst['msp'],threshold(dsv['msp'])),'MC Dropout PE seed2026':(tp,ts['pe'],threshold(vs['pe'])),'Deep Ensemble MSP':(enst,est['msp'],the),'Post-calibrated Ensemble MSP':(ct,cst['msp'],thc)};safe=[];cis=[]
 for name,(p,s,t) in methods.items():safe.append({'method':name,**safety(p,ty,s,t)});x=boot_ci(p,ty,s,t,tl,400,208000);[q0.update({'method':name}) for q0 in x];cis+=x
 pd.DataFrame(safe).to_csv(OUT/'clinical_safety_accounting.csv',index=False);pd.DataFrame(cis).to_csv(OUT/'lesion_cluster_ci_major_metrics.csv',index=False)
 sc=[]
 for name,(pv,pt,sv,st) in {'Deterministic MSP seed2026':(dv,dt,dsv['msp'],dst['msp']),'MC Dropout PE seed2026':(vp,tp,vs['pe'],ts['pe']),'Deep Ensemble MSP':(ensv,enst,esv['msp'],est['msp'])}.items():
  for c in [.05,.10,.15,.20]:
   q0=constrained(pv,vy,sv,c)
   if q0:th,vc,vm=q0;sc.append({'method':name,'validation_max_auto_miss':c,'threshold':th,'validation_coverage':vc,'validation_auto_miss':vm,**{'test_'+k:v for k,v in safety(pt,ty,st,th).items()}})
 pd.DataFrame(sc).to_csv(OUT/'safety_constrained_operating_points.csv',index=False)
 # T sensitivity on validation only
 f=FINAL/'seed2026_mc_T50_predictions.npz';trs=[]
 if f.exists():
  zz=np.load(f,allow_pickle=False);m50=align(rv,zz['val_image_ids'].astype(str),zz['val_mc_probs'])
  for Tn in [5,10,20,30,50]:
   p0,s0=mc_scores(m50[:,:Tn]);pr=p0.argmax(1)
   for sn in ['pe','mi','msp']:trs.append({'T':Tn,'score':sn,'validation_accuracy':accuracy_score(vy,pr),'validation_aurc':aurc(s0[sn],pr,vy),'validation_eaurc':eaurc(s0[sn],pr,vy)})
 pd.DataFrame(trs).to_csv(OUT/'mc_T_validation_sensitivity.csv',index=False)
 # variation ratio tie-aware random tie averaging
 tie=[];rng=np.random.default_rng(20260811)
 for j,d in enumerate(ss):
  _,_,tp,ts=mc[j];pr=tp.argmax(1);vals=[];uv=np.sort(np.unique(ts['vr']))
  for _ in range(300):
   order=[]
   for v in uv:
    ix=np.where(ts['vr']==v)[0].copy();rng.shuffle(ix);order+=ix.tolist()
   vals.append(aurc_ord((pr[np.asarray(order)]==ty[np.asarray(order)])))
  tie.append({'seed':d['s'],'n_unique_scores':len(uv),'stable_aurc':aurc(ts['vr'],pr,ty),'tie_averaged_aurc':np.mean(vals),'tie_sd':np.std(vals,ddof=1),'ci_low':ci(vals)[0],'ci_high':ci(vals)[1]})
 pd.DataFrame(tie).to_csv(OUT/'variation_ratio_tie_aware.csv',index=False)
 # split validation roles sensitivity (A score selection, B temperature, C threshold)
 rng=np.random.default_rng(20260811);u=np.unique(vl);rng.shuffle(u);a,b,c=np.array_split(u,3);ia=np.isin(vl,a);ib=np.isin(vl,b);ic=np.isin(vl,c);means={}
 for sn in ['msp','pe','ee','mi']:
  vals=[]
  for j in range(5):vp,vs,_,_=mc[j];vals.append(aurc(vs[sn][ia],vp.argmax(1)[ia],vy[ia]))
  means[sn]=float(np.mean(vals))
 sel=min(means,key=means.get);Tsep=fitT(ensv[ib],vy[ib]);vp,vs,tp,ts=mc[4];ths=threshold(vs[sel][ic]);sep={'validation_lesions_A_B_C':[len(a),len(b),len(c)],'score_selected_on_A':sel,'mean_AURC_A':means,'ensemble_temperature_on_B':Tsep,'threshold_on_C':ths,'test_selected_MC':metrics(tp,ty,ts[sel],ths),'test_ensemble_calibrated_from_B':metrics(applyT(enst,Tsep),ty,det_scores(applyT(enst,Tsep))['msp'],threshold(det_scores(applyT(ensv,Tsep))['msp']))};json.dump(sep,open(OUT/'separate_validation_roles_sensitivity.json','w'),indent=2)
 summary={'det_mc_mean_sd':ag,'ensemble':ens,'post_calibrated_ensemble':cal,'safety':safe,'validation_roles_sensitivity':sep,'notes':{'MC_configuration':'head-level MC Dropout','primary_MC_score':'predictive entropy selected by five-seed mean validation AURC','bootstrap_unit':'lesion_id'}};json.dump(summary,open(OUT/'reviewer_q1_summary.json','w'),indent=2)
 with tarfile.open(ARCH,'w:gz') as tf:tf.add(OUT,arcname='reviewer_q1_revision')
 h=hashlib.sha256(ARCH.read_bytes()).hexdigest();SHAF.write_text(f'{h}  {ARCH.name}\n');print('Q1_SUMMARY|'+json.dumps({'archive':str(ARCH),'size':ARCH.stat().st_size,'sha256':h,'ensemble':ens,'calibrated_ensemble':cal,'score_split_sensitivity':sel}),flush=True);print('PAPER2_Q1_REVIEWER_DONE',flush=True)
if __name__=='__main__':main()
