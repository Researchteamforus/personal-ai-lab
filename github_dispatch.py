import sys, time
from pathlib import Path
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
SCRIPT_PATH = Path("paper2_backup_outputs.py")
TARGETS = ["Node2-GPU-T4-x2", "Node10-GPU-T4-x2"]
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 1800

def request_json(method,url,**kwargs):
    r=requests.request(method,url,timeout=30,**kwargs); r.raise_for_status(); return r.json()

def main():
    if not SCRIPT_PATH.exists(): return 2
    code=SCRIPT_PATH.read_text(encoding='utf-8')
    state=request_json('GET',f'{COORDINATOR_URL}/get_state')
    online=[n.get('node_id') for n in state.get('nodes',[]) if n.get('status')=='online']
    print('ONLINE|'+','.join(sorted(online)),flush=True)
    targets=[n for n in TARGETS if n in online]
    if not targets:
        print('NO_TARGETS_ONLINE',flush=True); return 31
    d=request_json('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':targets})
    tid=d.get('task_id'); print(f'BACKUP_TASK|{tid}|targets={targets}',flush=True)
    if not tid: return 32
    deadline=time.time()+MAX_WAIT_SECONDS
    while time.time()<deadline:
        result=request_json('GET',f'{COORDINATOR_URL}/get_task_result/{tid}')
        status=result.get('status'); responses=result.get('responses',{})
        print(f'POLL|{status}|responses={len(responses)}',flush=True)
        if status=='completed':
            had_error=False
            for node in targets:
                data=responses.get(node)
                print(f'===== {node} =====',flush=True)
                if not data:
                    had_error=True; print('ERROR|missing response',flush=True)
                elif data.get('error'):
                    had_error=True; print('ERROR|'+str(data.get('error')),flush=True)
                else:
                    print(data.get('output',''),flush=True)
            return 40 if had_error else 0
        if status in {'failed','error','cancelled','overwritten'}:
            print('TASK_FAILED|'+json.dumps(result),flush=True); return 41
        time.sleep(POLL_SECONDS)
    print('TIMEOUT',flush=True); return 50

if __name__=='__main__': sys.exit(main())
