import sys, time
from pathlib import Path
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
SCRIPT_PATH = Path("paper2_multiseed_runner.py")
TARGET = "Node10-GPU-T4-x2"
SEEDS = [11, 29]
POLL_SECONDS = 10
MAX_WAIT_SECONDS = 7200

def request_json(method,url,**kwargs):
    r=requests.request(method,url,timeout=30,**kwargs); r.raise_for_status(); return r.json()

def main():
    if not SCRIPT_PATH.exists(): return 2
    base=SCRIPT_PATH.read_text(encoding='utf-8')
    code=f'TRAIN_SEEDS = {SEEDS!r}\n'+base
    state=request_json('GET',f'{COORDINATOR_URL}/get_state')
    online=[n.get('node_id') for n in state.get('nodes',[]) if n.get('status')=='online']
    print('ONLINE|'+','.join(sorted(online)),flush=True)
    if TARGET not in online:
        print('TARGET_OFFLINE|'+TARGET,flush=True); return 31
    d=request_json('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':[TARGET]})
    tid=d.get('task_id'); print(f'RECOVERY_TASK|{tid}|target={TARGET}|seeds={SEEDS}',flush=True)
    if not tid: return 32
    deadline=time.time()+MAX_WAIT_SECONDS
    while time.time()<deadline:
        result=request_json('GET',f'{COORDINATOR_URL}/get_task_result/{tid}')
        status=result.get('status'); responses=result.get('responses',{})
        print(f'POLL|{status}|responses={len(responses)}',flush=True)
        if status=='completed':
            data=responses.get(TARGET) or (next(iter(responses.values())) if responses else None)
            if not data or data.get('error'):
                print('ERROR|'+str(None if not data else data.get('error')),flush=True); return 40
            print(data.get('output',''),flush=True); return 0
        if status in {'failed','error','cancelled','overwritten'}:
            print('TASK_FAILED|'+str(result),flush=True); return 41
        time.sleep(POLL_SECONDS)
    print('TIMEOUT',flush=True); return 50

if __name__=='__main__': sys.exit(main())
