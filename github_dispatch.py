import sys,time,hashlib,json
from pathlib import Path
import requests
COORDINATOR_URL='https://miomiomiomizan-personal-ai-lab.hf.space';TARGET='Node10-GPU-T4-x2'
SCRIPT_PATH=Path('experiments/paper2_external_hiba.py')
REMOTE_ARCHIVE='/kaggle/working/paper2_data/Paper2_HIBA_EXTERNAL_RESULTS.tar.gz';REMOTE_SHA='/kaggle/working/paper2_data/Paper2_HIBA_EXTERNAL_RESULTS.sha256';LOCAL=Path('paper2_artifacts')
def req(method,url,**kw):
 r=requests.request(method,url,timeout=90,**kw);r.raise_for_status();return r.json()
def submit(code):
 d=req('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':[TARGET]});return d['task_id']
def wait(tid,mx=4800):
 end=time.time()+mx
 while time.time()<end:
  r=req('GET',f'{COORDINATOR_URL}/get_task_result/{tid}');st=r.get('status');resp=r.get('responses',{});print(f'POLL|{tid}|{st}|responses={len(resp)}',flush=True)
  if st=='completed':
   d=resp.get(TARGET)
   if not d:raise RuntimeError('missing Node10 response')
   out=d.get('output') or d.get('stdout') or '';print(out,flush=True)
   if d.get('error'):raise RuntimeError(str(d.get('error')))
   return out
  if st in {'failed','error','cancelled','overwritten'}:raise RuntimeError(st)
  time.sleep(5)
 raise TimeoutError(tid)
def remote(code,mx=300):return wait(submit(code),mx)
def between(t,a,b):
 i=t.find(a);j=t.find(b,i+len(a))
 if i<0 or j<0:raise ValueError((a,b))
 return t[i+len(a):j].strip()
def transfer(rp,lp):
 raw=remote(f"from pathlib import Path;import hashlib,json;p=Path({rp!r});print('<<<META>>>');print(json.dumps({{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}));print('<<<ENDMETA>>>')")
 meta=json.loads(between(raw,'<<<META>>>','<<<ENDMETA>>>'));total=int(meta['size']);expected=meta['sha256'];off=0;chunk=16384;lp.parent.mkdir(exist_ok=True)
 with lp.open('wb') as f:
  while off<total:
   want=min(chunk,total-off);raw=remote(f"from pathlib import Path;p=Path({rp!r});h=open(p,'rb');h.seek({off});b=h.read({want});print('<<<HEX>>>');print(b.hex());print('<<<ENDHEX>>>')");b=bytes.fromhex(between(raw,'<<<HEX>>>','<<<ENDHEX>>>'))
   if len(b)!=want:raise RuntimeError('chunk mismatch')
   f.write(b);off+=len(b);print(f'TRANSFER|{off}/{total}',flush=True)
 got=hashlib.sha256(lp.read_bytes()).hexdigest()
 if got!=expected:raise RuntimeError('checksum mismatch')
 return meta
def main():
 state=req('GET',f'{COORDINATOR_URL}/get_state');online={n['node_id'] for n in state.get('nodes',[]) if n.get('status')=='online'};print('ONLINE|'+','.join(sorted(online)),flush=True)
 if TARGET not in online:return 31
 src=SCRIPT_PATH.read_text(encoding='utf-8')
 loader="import traceback\nprint('HIBA_LOADER_START',flush=True)\ntry:\n src="+repr(src)+"\n print('HIBA_SCRIPT_BYTES|'+str(len(src)),flush=True)\n exec(compile(src,'paper2_external_hiba.py','exec'),{'__name__':'__main__'})\nexcept Exception:\n traceback.print_exc();print('HIBA_EXTERNAL_EXCEPTION',flush=True)\n"
 out=wait(submit(loader),4800);LOCAL.mkdir(exist_ok=True);(LOCAL/'hiba_external_remote_output.txt').write_text(out,encoding='utf-8')
 if 'PAPER2_HIBA_EXTERNAL_DONE' not in out:raise RuntimeError('HIBA completion marker missing')
 meta=transfer(REMOTE_ARCHIVE,LOCAL/'Paper2_HIBA_EXTERNAL_RESULTS.tar.gz');sha=remote(f"from pathlib import Path;print(Path({REMOTE_SHA!r}).read_text())").strip();(LOCAL/'Paper2_HIBA_EXTERNAL_RESULTS.sha256').write_text(sha+'\n',encoding='utf-8');print('HIBA_TRANSFER_VERIFIED|'+json.dumps(meta,sort_keys=True),flush=True);print('PAPER2_HIBA_GITHUB_READY',flush=True);return 0
if __name__=='__main__':sys.exit(main())
