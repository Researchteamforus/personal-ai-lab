import sys, time, base64, hashlib, json
from pathlib import Path
import requests

COORDINATOR_URL = 'https://miomiomiomizan-personal-ai-lab.hf.space'
TARGET = 'Node10-GPU-T4-x2'
REMOTE_SCRIPT_URL = 'https://raw.githubusercontent.com/Researchteamforus/personal-ai-lab/main/paper2_final_stage.py'
REMOTE_ARCHIVE = '/kaggle/working/paper2_data/Paper2_FINAL_STAGE_RESULTS.tar.gz'
REMOTE_SHA = '/kaggle/working/paper2_data/Paper2_FINAL_STAGE_RESULTS.sha256'
LOCAL_DIR = Path('paper2_artifacts')
POLL_SECONDS = 10
MAX_EXPERIMENT_WAIT = 4800


def request_json(method,url,**kwargs):
    r=requests.request(method,url,timeout=90,**kwargs); r.raise_for_status(); return r.json()


def submit(code):
    d=request_json('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':[TARGET]})
    tid=d.get('task_id')
    if not tid: raise RuntimeError('No task_id')
    return tid


def wait_task(tid,max_wait):
    deadline=time.time()+max_wait
    while time.time()<deadline:
        r=request_json('GET',f'{COORDINATOR_URL}/get_task_result/{tid}')
        st=r.get('status'); responses=r.get('responses',{})
        print(f'POLL|{tid}|{st}|responses={len(responses)}',flush=True)
        if st=='completed':
            data=responses.get(TARGET)
            if not data: raise RuntimeError('Missing Node10 response')
            output=data.get('output') or data.get('stdout') or ''
            if data.get('error'): raise RuntimeError('Remote error: '+str(data.get('error'))+'\nOUTPUT:\n'+str(output))
            if not output and data:
                output=json.dumps(data,default=str)
            return output
        if st in {'failed','error','cancelled','overwritten'}: raise RuntimeError(f'Task failed: {st}')
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f'Task {tid} timed out')


def run_remote(code,max_wait=300): return wait_task(submit(code),max_wait)


def extract(text,a,b):
    i=text.find(a); j=text.find(b)
    if i<0 or j<0 or j<=i: raise ValueError(f'Missing markers {a} {b}')
    return text[i+len(a):j].strip()


def transfer(remote_path,local_path):
    meta_code=f'''\nfrom pathlib import Path\nimport hashlib,json\np=Path({remote_path!r})\nif not p.exists(): raise FileNotFoundError(str(p))\nh=hashlib.sha256()\nwith p.open('rb') as f:\n    while True:\n        b=f.read(8*1024*1024)\n        if not b: break\n        h.update(b)\nprint('META_BEGIN'); print(json.dumps({{'size':p.stat().st_size,'sha256':h.hexdigest()}})); print('META_END')\n'''
    meta=json.loads(extract(run_remote(meta_code), 'META_BEGIN','META_END'))
    total=int(meta['size']); expected=meta['sha256']; print(f'TRANSFER_SOURCE|{remote_path}|size={total}|sha256={expected}',flush=True)
    local_path.parent.mkdir(parents=True,exist_ok=True)
    chunk=4*1024*1024; off=0
    with local_path.open('wb') as out:
        while off<total:
            want=min(chunk,total-off)
            code=f'''\nfrom pathlib import Path\nimport base64\np=Path({remote_path!r})\nwith p.open('rb') as f:\n f.seek({off}); b=f.read({want})\nprint('CHUNK_BEGIN'); print(base64.b64encode(b).decode('ascii')); print('CHUNK_END')\n'''
            raw=run_remote(code); data=base64.b64decode(extract(raw,'CHUNK_BEGIN','CHUNK_END'),validate=True)
            if len(data)!=want: raise RuntimeError(f'Chunk mismatch {len(data)} != {want}')
            out.write(data); off+=len(data); print(f'TRANSFER|{off}/{total}|{100*off/total:.1f}%',flush=True)
    h=hashlib.sha256()
    with local_path.open('rb') as f:
        while True:
            b=f.read(8*1024*1024)
            if not b: break
            h.update(b)
    got=h.hexdigest()
    if local_path.stat().st_size!=total or got!=expected: raise RuntimeError('Transfer verification failed')
    return {'size':total,'sha256':got}


def main():
    state=request_json('GET',f'{COORDINATOR_URL}/get_state')
    online={n.get('node_id') for n in state.get('nodes',[]) if n.get('status')=='online'}
    print('ONLINE|'+','.join(sorted(online)),flush=True)
    if TARGET not in online: return 31
    loader=f'''\nimport traceback, urllib.request\nprint("REMOTE_LOADER_START", flush=True)\ntry:\n    src=urllib.request.urlopen({REMOTE_SCRIPT_URL!r}, timeout=30).read().decode("utf-8")\n    print("REMOTE_SCRIPT_BYTES|"+str(len(src)), flush=True)\n    exec(compile(src, "paper2_final_stage.py", "exec"), {{"__name__":"__main__"}})\nexcept Exception:\n    traceback.print_exc()\n    print("PAPER2_FINAL_STAGE_EXCEPTION", flush=True)\n'''
    tid=submit(loader); print(f'FINAL_STAGE_TASK|{tid}',flush=True)
    output=wait_task(tid,MAX_EXPERIMENT_WAIT)
    LOCAL_DIR.mkdir(parents=True,exist_ok=True)
    (LOCAL_DIR/'final_stage_remote_output.txt').write_text(output,encoding='utf-8')
    print(output,flush=True)
    if 'PAPER2_FINAL_STAGE_DONE' not in output: raise RuntimeError('Final stage completion marker missing')
    meta=transfer(REMOTE_ARCHIVE,LOCAL_DIR/'Paper2_FINAL_STAGE_RESULTS.tar.gz')
    sha_text=run_remote(f"from pathlib import Path; print(Path({REMOTE_SHA!r}).read_text())").strip()
    (LOCAL_DIR/'Paper2_FINAL_STAGE_RESULTS.sha256').write_text(sha_text+'\n',encoding='utf-8')
    print('FINAL_STAGE_TRANSFER_VERIFIED|'+json.dumps(meta,sort_keys=True),flush=True)
    print('PAPER2_GITHUB_FINAL_ARTIFACT_READY',flush=True)
    return 0

if __name__=='__main__': sys.exit(main())
