import sys,time,hashlib,json
from pathlib import Path
import requests

COORDINATOR_URL='https://miomiomiomizan-personal-ai-lab.hf.space'
TARGET='Node10-GPU-T4-x2'
REMOTE_ARCHIVE='/kaggle/working/paper2_data/Paper2_REVIEWER_Q1_RESULTS.tar.gz'
REMOTE_SHA='/kaggle/working/paper2_data/Paper2_REVIEWER_Q1_RESULTS.sha256'
LOCAL_DIR=Path('paper2_artifacts')
POLL_SECONDS=2
MAX_WAIT=600

def req(method,url,**kw):
    r=requests.request(method,url,timeout=90,**kw); r.raise_for_status(); return r.json()

def submit(code):
    d=req('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':[TARGET]})
    tid=d.get('task_id')
    if not tid: raise RuntimeError('No task_id')
    return tid

def wait(tid,mx=120):
    end=time.time()+mx
    while time.time()<end:
        r=req('GET',f'{COORDINATOR_URL}/get_task_result/{tid}')
        st=r.get('status'); resp=r.get('responses',{})
        print(f'POLL|{tid}|{st}|responses={len(resp)}',flush=True)
        if st=='completed':
            d=resp.get(TARGET)
            if not d: raise RuntimeError('Missing Node10 response')
            out=d.get('output') or d.get('stdout') or ''
            if d.get('error'): raise RuntimeError(str(d.get('error'))+'\n'+out)
            return out
        if st in {'failed','error','cancelled','overwritten'}: raise RuntimeError(st)
        time.sleep(POLL_SECONDS)
    raise TimeoutError(tid)

def remote(code,mx=120): return wait(submit(code),mx)

def between(text,start,end):
    i=text.find(start); j=text.find(end,i+len(start))
    if i<0 or j<0: raise ValueError(f'markers missing: {start}, {end}')
    return text[i+len(start):j].strip()

def transfer_hex(remote_path,local_path):
    meta_raw=remote(f"from pathlib import Path;import hashlib,json;p=Path({remote_path!r});print('<<<META>>>');print(json.dumps({{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}));print('<<<ENDMETA>>>')")
    meta=json.loads(between(meta_raw,'<<<META>>>','<<<ENDMETA>>>'))
    total=int(meta['size']); expected=meta['sha256']; chunk=4096; off=0
    local_path.parent.mkdir(parents=True,exist_ok=True)
    with local_path.open('wb') as f:
        while off<total:
            want=min(chunk,total-off)
            raw=remote(f"from pathlib import Path;p=Path({remote_path!r});fh=open(p,'rb');fh.seek({off});b=fh.read({want});print('<<<HEX>>>');print(b.hex());print('<<<ENDHEX>>>')")
            b=bytes.fromhex(between(raw,'<<<HEX>>>','<<<ENDHEX>>>'))
            if len(b)!=want: raise RuntimeError(f'chunk size mismatch {len(b)} != {want}')
            f.write(b); off+=len(b); print(f'TRANSFER|{off}/{total}',flush=True)
    got=hashlib.sha256(local_path.read_bytes()).hexdigest()
    if got!=expected or local_path.stat().st_size!=total: raise RuntimeError('transfer verification failed')
    return meta

def main():
    state=req('GET',f'{COORDINATOR_URL}/get_state')
    online={n.get('node_id') for n in state.get('nodes',[]) if n.get('status')=='online'}
    print('ONLINE|'+','.join(sorted(online)),flush=True)
    if TARGET not in online: return 31
    check=remote(f"from pathlib import Path;p=Path({REMOTE_ARCHIVE!r});print('Q1_EXISTS|'+str(p.exists())+'|'+(str(p.stat().st_size) if p.exists() else '0'))")
    print(check,flush=True)
    if 'Q1_EXISTS|True' not in check: raise RuntimeError('Q1 archive missing on Node10')
    LOCAL_DIR.mkdir(parents=True,exist_ok=True)
    meta=transfer_hex(REMOTE_ARCHIVE,LOCAL_DIR/'Paper2_REVIEWER_Q1_RESULTS.tar.gz')
    sha=remote(f"from pathlib import Path;print(Path({REMOTE_SHA!r}).read_text())").strip()
    (LOCAL_DIR/'Paper2_REVIEWER_Q1_RESULTS.sha256').write_text(sha+'\n',encoding='utf-8')
    print('Q1_TRANSFER_VERIFIED|'+json.dumps(meta,sort_keys=True),flush=True)
    print('PAPER2_Q1_GITHUB_ARTIFACT_READY',flush=True)
    return 0

if __name__=='__main__': sys.exit(main())
