import json,time,socket,os
from pathlib import Path
import requests

BASE='https://miomiomiomizan-personal-ai-lab.hf.space'
TARGETS=['Node2-GPU-T4-x2','Node10-GPU-T4-x2','Nabila-GPU-T4-x2']
OUT=Path('paper2_artifacts'); OUT.mkdir(exist_ok=True)

def req(method,url,**kw):
    r=requests.request(method,url,timeout=90,**kw); r.raise_for_status(); return r.json()

state=req('GET',BASE+'/get_state')
online=[n['node_id'] for n in state.get('nodes',[]) if n.get('status')=='online']
print('ONLINE|'+','.join(online),flush=True)
targets=[t for t in TARGETS if t in online]
code="""import os,socket,json,torch
from pathlib import Path
print('HOST_DIAG|'+json.dumps({'hostname':socket.gethostname(),'env':{k:v for k,v in os.environ.items() if 'NODE' in k.upper() or 'KAGGLE' in k.upper()},'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'paper2_data':Path('/kaggle/working/paper2_data').exists(),'ham_images':len(list(Path('/kaggle/working/paper2_data/HAM10000_images').glob('*.jpg'))) if Path('/kaggle/working/paper2_data/HAM10000_images').exists() else 0},default=str),flush=True)
"""
d=req('POST',BASE+'/run_code',json={'code':code,'targets':targets}); tid=d['task_id']; print('TASK|'+tid,flush=True)
for _ in range(120):
    r=req('GET',BASE+'/get_task_result/'+tid); print('POLL|'+str(r.get('status'))+'|'+str(len(r.get('responses',{}))),flush=True)
    if r.get('status')=='completed':
        txt=json.dumps(r,indent=2,default=str); print(txt,flush=True); (OUT/'worker_identity.json').write_text(txt,encoding='utf-8'); break
    if r.get('status') in {'failed','error','cancelled','overwritten'}: raise RuntimeError(r)
    time.sleep(2)
else: raise TimeoutError(tid)
