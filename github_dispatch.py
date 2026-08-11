import base64,hashlib,json,os,sys,time
from pathlib import Path
import requests

BASE='https://miomiomiomizan-personal-ai-lab.hf.space'
REPO='Researchteamforus/personal-ai-lab'
TARGETS=['Node2-GPU-T4-x2','Node10-GPU-T4-x2','Nabila-GPU-T4-x2']
ROLES={
 'Node2-GPU-T4-x2':'node2_explain_tta_focal_backbone',
 'Node10-GPU-T4-x2':'node10_cv_0_1_2',
 'Nabila-GPU-T4-x2':'nabila_cv_3_4_deepdrop',
}
BACKUP_ARTIFACT=9087344257
EXPECTED_COMPLETE_SHA='1d7f864d4bbf5881b2b4350a271ebb34bb49b816fcd050c01bdfcaa2913aae3a'
SCRIPT=Path('experiments/paper2_top_tier_expansion.py')
OUT=Path('paper2_artifacts'); OUT.mkdir(exist_ok=True)

def req(method,url,**kw):
    r=requests.request(method,url,timeout=120,**kw); r.raise_for_status(); return r.json()

def signed_artifact(aid):
    tok=os.environ.get('GITHUB_TOKEN')
    if not tok: raise RuntimeError('GITHUB_TOKEN unavailable')
    r=requests.get(f'https://api.github.com/repos/{REPO}/actions/artifacts/{aid}/zip',headers={'Authorization':f'Bearer {tok}','Accept':'application/vnd.github+json'},allow_redirects=False,timeout=60)
    if r.status_code not in (301,302,303,307,308): raise RuntimeError(f'artifact redirect failed: {r.status_code} {r.text[:300]}')
    return r.headers['Location']

def submit(code,targets):
    d=req('POST',BASE+'/run_code',json={'code':code,'targets':targets}); tid=d.get('task_id')
    if not tid: raise RuntimeError('No task id: '+repr(d))
    return tid

def wait(tid,targets,seconds=10800,poll=5):
    end=time.time()+seconds
    while time.time()<end:
        r=req('GET',BASE+'/get_task_result/'+tid); st=r.get('status'); resp=r.get('responses',{})
        print(f'POLL|{tid}|{st}|responses={len(resp)}/{len(targets)}',flush=True)
        if st=='completed':
            missing=[t for t in targets if t not in resp]
            if missing: raise RuntimeError('missing responses '+repr(missing))
            for t in targets:
                d=resp[t]; out=d.get('output') or d.get('stdout') or ''
                (OUT/f"remote_{ROLES.get(t,t)}.txt").write_text(out,encoding='utf-8')
                print(f'REMOTE_TAIL|{t}|'+out[-12000:],flush=True)
                if d.get('error'): raise RuntimeError(f'{t}: {d.get("error")}\n{out[-4000:]}')
                if 'TOP_TIER_EXPANSION_DONE|' not in out: raise RuntimeError(f'{t}: completion marker missing')
            return resp
        if st in {'failed','error','cancelled','overwritten'}: raise RuntimeError(f'{tid}: {st}')
        time.sleep(poll)
    raise TimeoutError(tid)

def remote(target,code,seconds=300):
    tid=submit(code,[target]); end=time.time()+seconds
    while time.time()<end:
        r=req('GET',BASE+'/get_task_result/'+tid); st=r.get('status'); resp=r.get('responses',{})
        if st=='completed':
            d=resp.get(target)
            if not d: raise RuntimeError('missing '+target)
            out=d.get('output') or d.get('stdout') or ''
            if d.get('error'): raise RuntimeError(str(d.get('error'))+'\n'+out)
            return out
        if st in {'failed','error','cancelled','overwritten'}: raise RuntimeError(st)
        time.sleep(.6)
    raise TimeoutError(tid)

def between(t,a,b):
    i=t.find(a); j=t.find(b,i+len(a))
    if i<0 or j<0: raise ValueError((a,b,t[:300]))
    return t[i+len(a):j].strip()

def transfer(target,remote_path,local_path):
    raw=remote(target,f"from pathlib import Path;import hashlib,json;p=Path({remote_path!r});print('<<<META>>>');print(json.dumps({{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}));print('<<<ENDMETA>>>')")
    meta=json.loads(between(raw,'<<<META>>>','<<<ENDMETA>>>')); total=int(meta['size']); expected=meta['sha256']; local_path.parent.mkdir(exist_ok=True); off=0; chunk=1024*1024
    with local_path.open('wb') as f:
        while off<total:
            want=min(chunk,total-off)
            code=f"from pathlib import Path;import base64;p=Path({remote_path!r});h=open(p,'rb');h.seek({off});b=h.read({want});print('<<<B64>>>');print(base64.b64encode(b).decode());print('<<<ENDB64>>>')"
            s=remote(target,code,seconds=120); b=base64.b64decode(between(s,'<<<B64>>>','<<<ENDB64>>>'))
            if len(b)!=want: raise RuntimeError(f'{target} chunk {off}: {len(b)} != {want}')
            f.write(b); off+=len(b)
            if off==total or off%(16*1024*1024)<chunk: print(f'TRANSFER|{target}|{off}/{total}',flush=True)
    got=hashlib.sha256(local_path.read_bytes()).hexdigest()
    if got!=expected: raise RuntimeError(f'{target} checksum {got} != {expected}')
    print('TRANSFER_VERIFIED|'+target+'|'+json.dumps(meta,sort_keys=True),flush=True)
    return meta

def main():
    state=req('GET',BASE+'/get_state'); online={n['node_id'] for n in state.get('nodes',[]) if n.get('status')=='online'}; targets=[t for t in TARGETS if t in online]
    print('ONLINE|'+','.join(sorted(online)),flush=True)
    if len(targets)<3: raise RuntimeError('All three T4 workers are required; online='+repr(sorted(online)))
    backup_url=signed_artifact(BACKUP_ARTIFACT); src=SCRIPT.read_text(encoding='utf-8')
    wrapper="""import hashlib,tarfile,urllib.request,zipfile\nfrom pathlib import Path\nbase=Path('/kaggle/working/paper2_data');base.mkdir(parents=True,exist_ok=True)\ncp=base/'grouped_mc_seed2026/model_best_val_macro_f1.pt';split=base/'splits/ham10000_lesion_group_split_seed2026.csv'\nprint('EXPANSION_PREFLIGHT|cp='+str(cp.exists())+'|split='+str(split.exists()),flush=True)\nif not cp.exists() or not split.exists():\n z=Path('/kaggle/working/paper2_verified_backup.zip'); print('BACKUP_DOWNLOAD_START',flush=True); urllib.request.urlretrieve(BACKUP_URL,z); print('BACKUP_DOWNLOAD_DONE|'+str(z.stat().st_size),flush=True)\n d=Path('/kaggle/working/paper2_verified_backup'); d.mkdir(exist_ok=True); zipfile.ZipFile(z).extractall(d); complete=d/'Paper2_COMPLETE_BACKUP.tar'; h=hashlib.sha256(complete.read_bytes()).hexdigest(); print('COMPLETE_SHA|'+h,flush=True)\n if h!=EXPECTED_SHA: raise RuntimeError('backup checksum mismatch')\n tmp=Path('/kaggle/working/paper2_restore'); tmp.mkdir(exist_ok=True)\n with tarfile.open(complete,'r') as tf: tf.extractall(tmp,filter='data')\n for name in ['grouped_mc_seed2026.tar','splits.tar']:\n  p=tmp/name; print('RESTORE_INNER|'+name+'|'+str(p.exists()),flush=True)\n  if p.exists():\n   with tarfile.open(p,'r') as tf: tf.extractall(base,filter='data')\nprint('EXPANSION_POSTRESTORE|cp='+str(cp.exists())+'|split='+str(split.exists()),flush=True)\nsrc=SCRIPT_TEXT\nexec(compile(src,'paper2_top_tier_expansion.py','exec'),{'__name__':'__main__'})\n""".replace('BACKUP_URL',repr(backup_url)).replace('EXPECTED_SHA',repr(EXPECTED_COMPLETE_SHA)).replace('SCRIPT_TEXT',repr(src))
    tid=submit(wrapper,targets); print('EXPANSION_TASK|'+tid,flush=True); wait(tid,targets,seconds=10800,poll=10)
    manifest={}
    for target in targets:
        role=ROLES[target]; rp=f'/kaggle/working/paper2_data/Paper2_TOP_TIER_{role}.tar.gz'; lp=OUT/f'Paper2_TOP_TIER_{role}.tar.gz'; meta=transfer(target,rp,lp); manifest[target]={'role':role,**meta}
        sha_text=remote(target,f"from pathlib import Path;print(Path({(rp+'.sha256')!r}).read_text())").strip(); (OUT/f'Paper2_TOP_TIER_{role}.tar.gz.sha256').write_text(sha_text+'\n',encoding='utf-8')
    (OUT/'top_tier_expansion_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    print('TOP_TIER_EXPANSION_GITHUB_READY|'+json.dumps(manifest,sort_keys=True),flush=True)
    return 0

if __name__=='__main__': sys.exit(main())
