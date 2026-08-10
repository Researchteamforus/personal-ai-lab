"""Package all Paper 2 model/checkpoint/prediction/result artifacts before lab shutdown."""
from __future__ import annotations
import hashlib, json, os, tarfile, time
from pathlib import Path

BASE = Path('/kaggle/working/paper2_data')
EXPORT = Path('/kaggle/working/paper2_exports')
KEEP_SUFFIXES = {'.pt','.pth','.ckpt','.npz','.npy','.json','.csv','.tsv','.txt','.png','.jpg','.jpeg','.pdf','.svg','.log','.yaml','.yml'}
EXCLUDE_DIR_NAMES = {'HAM10000_images','__pycache__'}


def sha256_file(path: Path, chunk_size: int=8*1024*1024)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()


def wanted(path: Path)->bool:
    return path.is_file() and not any(part in EXCLUDE_DIR_NAMES for part in path.parts) and path.suffix.lower() in KEEP_SUFFIXES


def safe_name(rel: Path)->str:
    s='__'.join(rel.parts) if rel.parts else 'root'
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)


def build_manifest(files):
    rows=[]; total=0
    for p in files:
        size=p.stat().st_size; total+=size
        rows.append({'relative_path':str(p.relative_to(BASE)),'size_bytes':int(size),'sha256':sha256_file(p),'modified_unix':float(p.stat().st_mtime)})
    return {'base':str(BASE),'created_unix':time.time(),'hostname':os.uname().nodename,'file_count':len(rows),'total_bytes':int(total),'files':rows}


def group_files(files):
    groups={}
    for p in files:
        rel=p.relative_to(BASE); key=rel.parts[0] if len(rel.parts)>1 else 'root_files'; groups.setdefault(key,[]).append(p)
    return groups


def make_tar(name,files):
    out=EXPORT/f'{safe_name(Path(name))}.tar'
    with tarfile.open(out,'w') as tf:
        for p in files: tf.add(p,arcname=str(p.relative_to(BASE)),recursive=False)
    return out


def seed_status():
    status={}
    for seed in [11,29,47,71]:
        d=BASE/'multiseed_fixed_split'/f'seed_{seed}'
        status[str(seed)]={'directory':d.exists(),'best_pt':(d/'best.pt').exists(),'mc_predictions_npz':(d/'mc_predictions.npz').exists(),'summary_json':(d/'summary.json').exists()}
    d=BASE/'grouped_mc_seed2026'
    status['2026']={'directory':d.exists(),'best_pt':(d/'model_best_val_macro_f1.pt').exists(),'mc_predictions_npz':(d/'paper2_grouped_mc_predictions.npz').exists(),'summary_json':(d/'mc_summary.json').exists(),'statistical_revision_json':(d/'paper2_statistical_revision.json').exists()}
    return status


def critical_seed_files_complete(status):
    for seed in ['11','29','47','71']:
        row=status[seed]
        if not all(row[k] for k in ['directory','best_pt','mc_predictions_npz','summary_json']):
            return False
    row=status['2026']
    return all(row[k] for k in ['directory','best_pt','mc_predictions_npz','summary_json','statistical_revision_json'])


def main():
    if not BASE.exists():
        print('PAPER2_NO_SOURCE_DATA|'+str(BASE), flush=True)
        return
    EXPORT.mkdir(parents=True, exist_ok=True)
    files=sorted(p for p in BASE.rglob('*') if wanted(p))
    if not files: raise RuntimeError(f'No Paper 2 artifacts found under {BASE}')
    manifest=build_manifest(files)
    manifest_path=EXPORT/'paper2_artifact_manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    archives=[]
    for group,paths in sorted(group_files(files).items()):
        a=make_tar(group,paths); row={'group':group,'path':str(a),'size_bytes':a.stat().st_size,'sha256':sha256_file(a),'file_count':len(paths)}; archives.append(row); print('BACKUP_ARCHIVE|'+json.dumps(row,sort_keys=True),flush=True)
    status=seed_status()
    index={'manifest_path':str(manifest_path),'manifest_sha256':sha256_file(manifest_path),'archives':archives,'source_file_count':manifest['file_count'],'source_total_bytes':manifest['total_bytes'],'seed_status':status,'critical_seed_files_complete':critical_seed_files_complete(status)}
    index_path=EXPORT/'paper2_backup_index.json'; index_path.write_text(json.dumps(index,indent=2,sort_keys=True),encoding='utf-8')
    verification={Path(a['path']).name:(sha256_file(Path(a['path']))==a['sha256']) for a in archives}
    index['archive_verification']=verification
    index['all_archive_checksums_verified']=all(verification.values())
    index_path.write_text(json.dumps(index,indent=2,sort_keys=True),encoding='utf-8')
    print('PAPER2_BACKUP_INDEX_BEGIN',flush=True); print(json.dumps(index,indent=2,sort_keys=True),flush=True); print('PAPER2_BACKUP_INDEX_END',flush=True)
    print('PAPER2_BACKUP_VERIFY|'+json.dumps(verification,sort_keys=True),flush=True)
    if not index['critical_seed_files_complete']:
        raise RuntimeError('Critical Paper 2 seed artifacts are incomplete')
    if not index['all_archive_checksums_verified']:
        raise RuntimeError('One or more backup archive checksums failed verification')
    print('PAPER2_BACKUP_COMPLETE',flush=True)


main()
