import sys,time,base64,hashlib,json
from pathlib import Path
import requests
COORDINATOR_URL='https://miomiomiomizan-personal-ai-lab.hf.space';TARGET='Node10-GPU-T4-x2'
REMOTE_SCRIPT_URL='https://raw.githubusercontent.com/Researchteamforus/personal-ai-lab/main/experiments/paper2_reviewer_q1.py'
REMOTE_ARCHIVE='/kaggle/working/paper2_data/Paper2_REVIEWER_Q1_RESULTS.tar.gz';REMOTE_SHA='/kaggle/working/paper2_data/Paper2_REVIEWER_Q1_RESULTS.sha256'
LOCAL_DIR=Path('paper2_artifacts');POLL_SECONDS=10;MAX_WAIT=4800
def req(method,url,**kw):
 r=requests.request(method,url,timeout=90,**kw);r.raise_for_status();return r.json()
def submit(code):
 d=req('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':[TARGET]});tid=d.get('task_id')
 if not tid:raise RuntimeError('No task_id')
 return tid
def wait(tid,mx):
 end=time.time()+mx
 while time.time()<end:
  r=req('GET',f'{COORDINATOR_URL}/get_task_result/{tid}');st=r.get('status');resp=r.get('responses',{});print(f'POLL|{tid}|{st}|responses={len(resp)}',flush=True)
  if st=='completed':
   d=resp.get(TARGET)
   if not d:raise RuntimeError('Missing Node10 response')
   out=d.get('output') or d.get('stdout') or ''
   if d.get('error'):raise RuntimeError(str(d.get('error'))+'\n'+out)
   return out
  if st in {'failed','error','cancelled','overwritten'}:raise RuntimeError(st)
  time.sleep(POLL_SECONDS)
 raise TimeoutError(tid)
def remote(code,mx=300):return wait(submit(code),mx)
def extract(t,a,b):
 i=t.find(a);j=t.find(b)
 if i<0 or j<0 or j<=i:raise ValueError((a,b))
 return t[i+len(a):j].strip()
def transfer(rp,lp):
 meta=json.loads(extract(remote(f"from pathlib import Path;import hashlib,json;p=Path({rp!r});h=hashlib.sha256(p.read_bytes()).hexdigest();print('M0');print(json.dumps({{'size':p.stat().st_size,'sha256':h}}));print('M1')"),'M0','M1'));total=int(meta['size']);expected=meta['sha256'];lp.parent.mkdir(parents=True,exist_ok=True);off=0;chunk=4*1024*1024
 with lp.open('wb') as f:
  while off<total:
   want=min(chunk,total-off);raw=remote(f"from pathlib import Path;import base64;p=Path({rp!r});f=open(p,'rb');f.seek({off});b=f.read({want});print('C0');print(base64.b64encode(b).decode());print('C1')");b=base64.b64decode(extract(raw,'C0','C1'));f.write(b);off+=len(b);print(f'TRANSFER|{off}/{total}',flush=True)
 got=hashlib.sha256(lp.read_bytes()).hexdigest()
 if got!=expected or lp.stat().st_size!=total:raise RuntimeError('transfer verify failed')
 return meta
def main():
 state=req('GET',f'{COORDINATOR_URL}/get_state');online={n.get('node_id') for n in state.get('nodes',[]) if n.get('status')=='online'};print('ONLINE|'+','.join(sorted(online)),flush=True)
 if TARGET not in online:return 31
 loader=f'''import traceback,urllib.request,tarfile\nfrom pathlib import Path\nprint("Q1_LOADER_START",flush=True)\ntry:\n base=Path("/kaggle/working/paper2_data"); split=base/"splits/ham10000_lesion_group_split_seed2026.csv"; complete=Path("/kaggle/working/Paper2_COMPLETE_BACKUP.tar")\n print("RESTORE_PREFLIGHT|complete="+str(complete.exists())+"|split="+str(split.exists()),flush=True)\n if not split.exists() and complete.exists():\n  tmp=Path("/kaggle/working/paper2_restore");tmp.mkdir(parents=True,exist_ok=True)\n  with tarfile.open(complete,"r") as tf: tf.extractall(tmp,filter="data")\n  for name in ["grouped_mc_seed2026.tar","multiseed_fixed_split.tar","splits.tar"]:\n   p=tmp/name\n   if p.exists():\n    print("RESTORE_INNER|"+name,flush=True)\n    with tarfile.open(p,"r") as tf: tf.extractall(base,filter="data")\n final_archive=base/"Paper2_FINAL_STAGE_RESULTS.tar.gz"; final_pred=base/"final_stage/deep_ensemble_predictions.npz"\n if not final_pred.exists() and final_archive.exists():\n  print("RESTORE_FINAL_STAGE",flush=True)\n  with tarfile.open(final_archive,"r:gz") as tf: tf.extractall(base,filter="data")\n print("RESTORE_POST|split="+str(split.exists())+"|final="+str(final_pred.exists()),flush=True)\n src=urllib.request.urlopen({REMOTE_SCRIPT_URL!r},timeout=30).read().decode("utf-8")\n print("Q1_SCRIPT_BYTES|"+str(len(src)),flush=True)\n exec(compile(src,"paper2_reviewer_q1.py","exec"),{{"__name__":"__main__"}})\nexcept Exception:\n traceback.print_exc();print("Q1_REVIEWER_EXCEPTION",flush=True)\n'''
 tid=submit(loader);print('Q1_TASK|'+tid,flush=True);out=wait(tid,MAX_WAIT);LOCAL_DIR.mkdir(exist_ok=True);(LOCAL_DIR/'q1_reviewer_remote_output.txt').write_text(out);print(out,flush=True)
 if 'PAPER2_Q1_REVIEWER_DONE' not in out:raise RuntimeError('completion marker missing')
 meta=transfer(REMOTE_ARCHIVE,LOCAL_DIR/'Paper2_REVIEWER_Q1_RESULTS.tar.gz');sha=remote(f"from pathlib import Path;print(Path({REMOTE_SHA!r}).read_text())").strip();(LOCAL_DIR/'Paper2_REVIEWER_Q1_RESULTS.sha256').write_text(sha+'\n');print('Q1_TRANSFER_VERIFIED|'+json.dumps(meta),flush=True);return 0
if __name__=='__main__':sys.exit(main())
