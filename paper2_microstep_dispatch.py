import json, os, re, subprocess, time
from pathlib import Path
import requests

BASE='https://miomiomiomizan-personal-ai-lab.hf.space'
PLAN={'Node2-GPU-T4-x2':([0,1,2],False),'Node10-GPU-T4-x2':([3,4],True)}
STATUS=Path('results/paper2_microsteps_status.json')
OUT=Path('results/paper2_microsteps'); OUT.mkdir(parents=True,exist_ok=True)
POLL=int(os.getenv('MICRO_POLL_SECONDS','300')); LIMIT=int(os.getenv('MICRO_MAX_SECONDS','18000'))


def req(method,path,**kw):
    r=requests.request(method,BASE+path,timeout=120,**kw); r.raise_for_status(); return r.json()

def task(code,targets,timeout=180):
    d=req('POST','/run_code',json={'code':code,'targets':targets}); tid=d.get('task_id')
    if not tid: raise RuntimeError(d)
    end=time.time()+timeout
    while time.time()<end:
        x=req('GET','/get_task_result/'+tid); st=x.get('status'); rp=x.get('responses',{})
        if st=='completed':
            if any(t not in rp for t in targets): raise RuntimeError('missing response '+repr(x))
            return rp
        if st in {'failed','error','cancelled','overwritten'}: raise RuntimeError(repr(x))
        time.sleep(1)
    raise TimeoutError(tid)

def boot(target,folds,aug,src):
    runner=f'''import importlib.util,traceback\nfrom pathlib import Path\np=Path('/kaggle/working/paper2_focused_revision_20260820.py')\ns=importlib.util.spec_from_file_location('focused',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\ntry:\n print('MICRO_NODE_START|{target}',flush=True);m.download_ham();df=m.metadata()\n for f in {folds!r}:\n  print('MICRO_FOLD_START|'+str(f),flush=True);m.run_crossfit_outer(df,f);m.archive_component('crossfit_outer_'+str(f),m.OUTROOT/'crossfit'/('outer_'+str(f)));print('MICRO_FOLD_DONE|'+str(f),flush=True)\n if {aug!r}:\n  print('MICRO_AUG_START',flush=True);m.run_augmentation_pair(df);m.archive_component('augmentation_sensitivity',m.OUTROOT/'augmentation_sensitivity');print('MICRO_AUG_DONE',flush=True)\n print('MICRO_NODE_DONE|{target}',flush=True)\nexcept Exception:\n print('MICRO_NODE_ERROR|'+traceback.format_exc(),flush=True);raise\n'''
    code=f'''import os,subprocess,sys\nfrom pathlib import Path\nr=Path('/kaggle/working/paper2_microsteps');r.mkdir(parents=True,exist_ok=True);pidf=r/'worker.pid';alive=False\nif pidf.exists():\n try:\n  pid=int(pidf.read_text());os.kill(pid,0);alive=True\n except Exception: alive=False\nPath('/kaggle/working/paper2_focused_revision_20260820.py').write_text({src!r},encoding='utf-8')\nPath('/kaggle/working/paper2_micro_runner.py').write_text({runner!r},encoding='utf-8')\nif alive: print('MICRO_ALREADY_RUNNING|'+str(pid),flush=True)\nelse:\n log=open(r/'worker.log','ab',buffering=0);p=subprocess.Popen([sys.executable,'/kaggle/working/paper2_micro_runner.py'],stdout=log,stderr=subprocess.STDOUT,cwd='/kaggle/working',start_new_session=True);pidf.write_text(str(p.pid));print('MICRO_STARTED|'+str(p.pid),flush=True)\n'''
    z=task(code,[target],120)[target]; print((z.get('output') or z.get('stdout') or ''),flush=True)
    if z.get('error'): raise RuntimeError(str(z.get('error')))

def query(targets):
    code=r'''import json,os\nfrom pathlib import Path\nr=Path('/kaggle/working/paper2_microsteps');pid=None;alive=False\ntry: pid=int((r/'worker.pid').read_text());os.kill(pid,0);alive=True\nexcept Exception: pass\no={'pid':pid,'alive':alive,'log':(r/'worker.log').read_text(encoding='utf-8',errors='replace') if (r/'worker.log').exists() else '','summaries':{},'augmentation':None}\nfor f in range(5):\n p=Path(f'/kaggle/working/paper2_data/focused_revision_20260820/crossfit/outer_{f}/summary.json')\n if p.exists():\n  try:o['summaries'][str(f)]=json.loads(p.read_text())\n  except Exception:pass\np=Path('/kaggle/working/paper2_data/focused_revision_20260820/augmentation_sensitivity/paired_summary.json')\nif p.exists():\n try:o['augmentation']=json.loads(p.read_text())\n except Exception:pass\nprint('<<<S>>>');print(json.dumps(o,default=str));print('<<<E>>>')'''
    rp=task(code,targets,120); out={}
    for t,z in rp.items():
        s=z.get('output') or z.get('stdout') or ''; a=s.find('<<<S>>>'); b=s.find('<<<E>>>',a+7)
        out[t]=json.loads(s[a+7:b].strip()) if a>=0 and b>=0 else {'alive':False,'log':'QUERY_ERROR '+s[-2000:],'summaries':{}}
    return out

def parse(log,folds):
    fu={str(f):0 for f in folds}; cf=ci=None; phase='startup'; base=0; latest=None
    for line in log.splitlines():
        m=re.search(r'CROSSFIT_INNER\|outer=(\d+)\|inner=(\d+)',line)
        if m: cf=int(m.group(1));ci=int(m.group(2));base=ci*50;phase='inner_train';fu[str(cf)]=max(fu.get(str(cf),0),base);continue
        if line.startswith('TRAIN|') and cf is not None and phase.startswith('inner'):
            try: latest=json.loads(line.split('|',1)[1]); ep=int(latest.get('epoch',0));fu[str(cf)]=max(fu[str(cf)],base+min(ep,20));phase='inner_train'
            except Exception: pass
            continue
        if line.startswith('EARLY_STOP|') and cf is not None and phase=='inner_train': fu[str(cf)]=max(fu[str(cf)],base+20);phase='inner_mc';continue
        m=re.search(r'CROSSFIT_FINAL\|outer=(\d+)\|epochs=(\d+)',line)
        if m: cf=int(m.group(1));ci=None;base=150;phase='final_train';fu[str(cf)]=max(fu.get(str(cf),0),150);continue
        if line.startswith('FINAL_TRAIN|') and cf is not None:
            try: latest=json.loads(line.split('|',1)[1]);ep=int(latest.get('epoch',0));fu[str(cf)]=max(fu[str(cf)],150+min(ep,20));phase='final_train'
            except Exception: pass
            continue
        m=re.search(r'MC\|(\d+)/(\d+)',line)
        if m and cf is not None:
            q=min(30,round(30*int(m.group(1))/max(1,int(m.group(2)))))
            if phase=='inner_train': phase='inner_mc'
            if phase=='inner_mc': fu[str(cf)]=max(fu[str(cf)],base+20+q)
            elif phase=='final_train': phase='final_mc';fu[str(cf)]=max(fu[str(cf)],170+q)
            elif phase=='final_mc': fu[str(cf)]=max(fu[str(cf)],170+q)
            continue
        m=re.search(r'MICRO_FOLD_DONE\|(\d+)',line)
        if m: f=m.group(1);fu[f]=200;phase='fold_done'
        if line.startswith('MICRO_AUG_START'): phase='augmentation'
        if line.startswith('MICRO_AUG_DONE'): phase='augmentation_done'
    status='error' if 'MICRO_NODE_ERROR|' in log else ('completed' if 'MICRO_NODE_DONE|' in log else 'running')
    return {'status':status,'phase':phase,'fold_units':fu,'completed_core_units':sum(fu.values()),'current_fold':cf,'current_inner':ci,'latest_metrics':latest,'last_line':log.splitlines()[-1][-1500:] if log.splitlines() else None}

def persist(nodes,status):
    subprocess.run(['git','pull','--rebase','origin','main'],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    STATUS.parent.mkdir(parents=True,exist_ok=True);STATUS.write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
    for t,d in nodes.items():
        for f,s in d.get('summaries',{}).items(): (OUT/f'outer_{f}_summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
        if isinstance(d.get('augmentation'),dict): (OUT/'augmentation_paired_summary.json').write_text(json.dumps(d['augmentation'],indent=2,sort_keys=True)+'\n')
    subprocess.run(['git','add',str(STATUS),str(OUT)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode==0:return
    subprocess.run(['git','config','user.name','github-actions[bot]'],check=True);subprocess.run(['git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
    subprocess.run(['git','commit','-m',f"Update Paper 2 microsteps {status['core_completed_steps']}/1000"],check=True);subprocess.run(['git','push','origin','HEAD:main'],check=True)

def main():
    src=Path('experiments/paper2_focused_revision_20260820.py').read_text()
    st=req('GET','/get_state');online={n.get('node_id') for n in st.get('nodes',[]) if n.get('status')=='online'};missing=[t for t in PLAN if t not in online]
    if missing: raise RuntimeError('offline '+repr(missing)+' online='+repr(sorted(online)))
    if (st.get('execution') or {}).get('status')=='running': raise RuntimeError('foreground task already running: '+str((st.get('execution') or {}).get('current_task_id')))
    for t,(folds,aug) in PLAN.items(): boot(t,folds,aug,src)
    end=time.time()+LIMIT
    while time.time()<end:
        nodes=query(list(PLAN)); ns={};done=0;err=False
        for t,(folds,aug) in PLAN.items():
            p=parse(nodes[t].get('log',''),folds);p['process_alive']=nodes[t].get('alive');p['pid']=nodes[t].get('pid');p['augmentation_assigned']=aug;ns[t]=p;done+=p['completed_core_units'];err|=p['status']=='error'
        done=max(0,min(1000,done));status={'schema':1,'checked_at':time.time(),'core_total_steps':1000,'core_completed_steps':done,'core_percent':round(done/10,1),'nodes':ns,'all_core_done':done>=1000,'has_error':err}
        print('MICRO_PROGRESS|'+json.dumps(status,sort_keys=True),flush=True);persist(nodes,status)
        if done>=1000:return 0
        if err:raise RuntimeError('worker error')
        time.sleep(POLL)
    raise TimeoutError('microstep monitor timeout')

if __name__=='__main__': raise SystemExit(main())
