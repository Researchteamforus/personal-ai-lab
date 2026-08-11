import sys,time,json
from pathlib import Path
import requests
COORDINATOR_URL='https://miomiomiomizan-personal-ai-lab.hf.space'; TARGET='Node10-GPU-T4-x2'; OUT=Path('paper2_artifacts')
def req(method,url,**kw):
 r=requests.request(method,url,timeout=90,**kw);r.raise_for_status();return r.json()
def submit(code):
 d=req('POST',f'{COORDINATOR_URL}/run_code',json={'code':code,'targets':[TARGET]});return d['task_id']
def wait(tid,mx=300):
 end=time.time()+mx
 while time.time()<end:
  r=req('GET',f'{COORDINATOR_URL}/get_task_result/{tid}');st=r.get('status');resp=r.get('responses',{});print(f'POLL|{tid}|{st}|responses={len(resp)}',flush=True)
  if st=='completed':
   d=resp[TARGET];out=d.get('output') or '';print(out,flush=True)
   if d.get('error'):raise RuntimeError(str(d['error']))
   return out
  time.sleep(2)
 raise TimeoutError(tid)
def main():
 state=req('GET',f'{COORDINATOR_URL}/get_state');print('ONLINE|'+','.join(n['node_id'] for n in state.get('nodes',[]) if n.get('status')=='online'),flush=True)
 code=r'''import pandas as pd,urllib.request,json\nfrom pathlib import Path\nu='https://isic-archive.s3.amazonaws.com/dois/10.34970-559884/hiba-skin-lesions.csv'\np=Path('/kaggle/working/hiba-skin-lesions.csv');urllib.request.urlretrieve(u,p)\ndf=pd.read_csv(p)\nprint('HIBA_META_SIZE|'+str(p.stat().st_size))\nprint('HIBA_COLUMNS|'+json.dumps(df.columns.tolist()))\nprint('HIBA_SHAPE|'+str(df.shape))\nfor c in df.columns:\n s=df[c]\n if s.nunique(dropna=False)<=30 or any(k in c.lower() for k in ['diagn','dx','lesion','image','patient','type']):\n  print('HIBA_COL|'+c+'|NUNIQUE='+str(s.nunique(dropna=False))+'|VALUES='+json.dumps(s.value_counts(dropna=False).head(40).astype(int).to_dict(),default=str))\nprint('HIBA_PROBE_DONE')\n'''
 out=wait(submit(code));OUT.mkdir(exist_ok=True);(OUT/'hiba_metadata_probe.txt').write_text(out,encoding='utf-8');return 0
if __name__=='__main__':sys.exit(main())
