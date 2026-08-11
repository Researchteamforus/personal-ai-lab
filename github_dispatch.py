import sys,time,hashlib,json,os
from pathlib import Path
import requests

COORDINATOR_URL='https://miomiomiomizan-personal-ai-lab.hf.space'
TARGET='Node2-GPU-T4-x2'
REPO='Researchteamforus/personal-ai-lab'
BACKUP_ARTIFACT=9087344257
FINAL_ARTIFACT=9087590772
EXPECTED_COMPLETE_SHA='1d7f864d4bbf5881b2b4350a271ebb34bb49b816fcd050c01bdfcaa2913aae3a'
SCRIPT_PATH=Path('experiments/paper2_external_hiba.py')
REMOTE_ARCHIVE='/kaggle/working/paper2_data/Paper2_HIBA_EXTERNAL_RESULTS.tar.gz'
REMOTE_SHA='/kaggle/working/paper2_data/Paper2_HIBA_EXTERNAL_RESULTS.sha256'
LOCAL=Path('paper2_artifacts')

def req(method,url,**kw):
    r=requests.request(method,url,timeout=90,**kw); r.raise_for_status(); return r.json()

def signed_artifact(aid):
    tok=os.environ.get('GITHUB_TOKEN')
    if not tok: raise RuntimeError('GITHUB_TOKEN unavailable')
    r=requests.get(f'https://api.github.com/repos/{REPO}/actions/artifacts/{aid}/zip',headers={'Authorization':f'Bearer {tok}','Accept':'application/vnd.github+json'},allow_redirects=False,timeout=60)
    if r.status_code not in (301,302,303,307,308): raise RuntimeError(f'artifact redirect failed {aid}: {r.status_code}')
    return r.headers['Location']

def submit(code):
    d=req('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':[TARGET]})
    tid=d.get('task_id')
    if not tid: raise RuntimeError('No task id')
    return tid

def wait(tid,mx=4800):
    end=time.time()+mx
    while time.time()<end:
        r=req('GET',f'{COORDINATOR_URL}/get_task_result/{tid}'); st=r.get('status'); resp=r.get('responses',{})
        print(f'POLL|{tid}|{st}|responses={len(resp)}',flush=True)
        if st=='completed':
            d=resp.get(TARGET)
            if not d: raise RuntimeError('missing target response')
            out=d.get('output') or d.get('stdout') or ''
            print(out,flush=True)
            if d.get('error'): raise RuntimeError(str(d.get('error')))
            return out
        if st in {'failed','error','cancelled','overwritten'}: raise RuntimeError(st)
        time.sleep(5)
    raise TimeoutError(tid)

def remote(code,mx=300): return wait(submit(code),mx)

def between(t,a,b):
    i=t.find(a); j=t.find(b,i+len(a))
    if i<0 or j<0: raise ValueError((a,b))
    return t[i+len(a):j].strip()

def transfer(rp,lp):
    raw=remote(f"from pathlib import Path;import hashlib,json;p=Path({rp!r});print('<<<META>>>');print(json.dumps({{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}));print('<<<ENDMETA>>>')")
    meta=json.loads(between(raw,'<<<META>>>','<<<ENDMETA>>>')); total=int(meta['size']); expected=meta['sha256']; off=0; chunk=16384
    lp.parent.mkdir(exist_ok=True)
    with lp.open('wb') as f:
        while off<total:
            want=min(chunk,total-off)
            raw=remote(f"from pathlib import Path;p=Path({rp!r});h=open(p,'rb');h.seek({off});b=h.read({want});print('<<<HEX>>>');print(b.hex());print('<<<ENDHEX>>>')")
            b=bytes.fromhex(between(raw,'<<<HEX>>>','<<<ENDHEX>>>'))
            if len(b)!=want: raise RuntimeError('chunk mismatch')
            f.write(b); off+=len(b); print(f'TRANSFER|{off}/{total}',flush=True)
    got=hashlib.sha256(lp.read_bytes()).hexdigest()
    if got!=expected: raise RuntimeError('checksum mismatch')
    return meta

def main():
    state=req('GET',f'{COORDINATOR_URL}/get_state'); online={n['node_id'] for n in state.get('nodes',[]) if n.get('status')=='online'}
    print('ONLINE|'+','.join(sorted(online)),flush=True)
    if TARGET not in online: return 31
    backup_url=signed_artifact(BACKUP_ARTIFACT); final_url=signed_artifact(FINAL_ARTIFACT); src=SCRIPT_PATH.read_text(encoding='utf-8')
    loader="""import traceback,urllib.request,tarfile,zipfile,hashlib,shutil\nfrom pathlib import Path\nprint('HIBA_NODE2_LOADER_START',flush=True)\ntry:\n base=Path('/kaggle/working/paper2_data');base.mkdir(parents=True,exist_ok=True)\n split=base/'splits/ham10000_lesion_group_split_seed2026.csv';final_pred=base/'final_stage/deep_ensemble_predictions.npz'\n print('RESTORE_PREFLIGHT|split='+str(split.exists())+'|final='+str(final_pred.exists()),flush=True)\n if not split.exists():\n  z=Path('/kaggle/working/paper2_verified_backup.zip');print('DOWNLOAD_BACKUP_START',flush=True);urllib.request.urlretrieve(BACKUP_URL,z);print('DOWNLOAD_BACKUP_DONE|'+str(z.stat().st_size),flush=True)\n  d=Path('/kaggle/working/paper2_verified_backup');d.mkdir(exist_ok=True);zipfile.ZipFile(z).extractall(d);complete=d/'Paper2_COMPLETE_BACKUP.tar'\n  h=hashlib.sha256(complete.read_bytes()).hexdigest();print('COMPLETE_SHA|'+h,flush=True)\n  if h!=EXPECTED_SHA:raise RuntimeError('complete backup checksum mismatch')\n  tmp=Path('/kaggle/working/paper2_restore');tmp.mkdir(exist_ok=True)\n  with tarfile.open(complete,'r') as tf:tf.extractall(tmp,filter='data')\n  for name in ['grouped_mc_seed2026.tar','multiseed_fixed_split.tar','splits.tar']:\n   p=tmp/name;print('RESTORE_INNER|'+name+'|'+str(p.exists()),flush=True)\n   if p.exists():\n    with tarfile.open(p,'r') as tf:tf.extractall(base,filter='data')\n if not final_pred.exists():\n  z2=Path('/kaggle/working/paper2_final_artifact.zip');print('DOWNLOAD_FINAL_START',flush=True);urllib.request.urlretrieve(FINAL_URL,z2);print('DOWNLOAD_FINAL_DONE|'+str(z2.stat().st_size),flush=True)\n  d2=Path('/kaggle/working/paper2_final_artifact');d2.mkdir(exist_ok=True);zipfile.ZipFile(z2).extractall(d2);fa=d2/'Paper2_FINAL_STAGE_RESULTS.tar.gz'\n  if not fa.exists():raise FileNotFoundError(fa)\n  shutil.copy2(fa,base/fa.name)\n  with tarfile.open(fa,'r:gz') as tf:tf.extractall(base,filter='data')\n print('RESTORE_POST|split='+str(split.exists())+'|final='+str(final_pred.exists()),flush=True)\n src=SCRIPT_TEXT\n print('HIBA_SCRIPT_BYTES|'+str(len(src)),flush=True)\n exec(compile(src,'paper2_external_hiba.py','exec'),{'__name__':'__main__'})\nexcept Exception:\n traceback.print_exc();print('HIBA_EXTERNAL_EXCEPTION',flush=True)\n""".replace('BACKUP_URL',repr(backup_url)).replace('FINAL_URL',repr(final_url)).replace('EXPECTED_SHA',repr(EXPECTED_COMPLETE_SHA)).replace('SCRIPT_TEXT',repr(src))
    out=wait(submit(loader),4800); LOCAL.mkdir(exist_ok=True); (LOCAL/'hiba_external_remote_output.txt').write_text(out,encoding='utf-8')
    if 'PAPER2_HIBA_EXTERNAL_DONE' not in out: raise RuntimeError('HIBA completion marker missing')
    meta=transfer(REMOTE_ARCHIVE,LOCAL/'Paper2_HIBA_EXTERNAL_RESULTS.tar.gz'); sha=remote(f"from pathlib import Path;print(Path({REMOTE_SHA!r}).read_text())").strip(); (LOCAL/'Paper2_HIBA_EXTERNAL_RESULTS.sha256').write_text(sha+'\n',encoding='utf-8')
    print('HIBA_TRANSFER_VERIFIED|'+json.dumps(meta,sort_keys=True),flush=True); print('PAPER2_HIBA_GITHUB_READY',flush=True); return 0

if __name__=='__main__': sys.exit(main())
