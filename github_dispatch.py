import sys, time, json
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
TARGETS = ["Node10-GPU-T4-x2"]
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 900

REMOTE_CODE = r'''
from pathlib import Path
import hashlib, tarfile, json

EXPORT_DIR = Path('/kaggle/working/paper2_exports')
FINAL_ARCHIVE = Path('/kaggle/working/Paper2_COMPLETE_BACKUP.tar')
NAMES = [
    'grouped_mc_seed2026.tar',
    'multiseed_fixed_split.tar',
    'splits.tar',
    'paper2_artifact_manifest.json',
    'paper2_backup_index.json',
]

def sha256_file(path, chunk_size=8*1024*1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

rows = []
for name in NAMES:
    p = EXPORT_DIR / name
    if not p.exists():
        raise FileNotFoundError(str(p))
    rows.append({
        'name': name,
        'path': str(p),
        'size_bytes': p.stat().st_size,
        'size_mb': round(p.stat().st_size / (1024**2), 3),
        'sha256': sha256_file(p),
    })

print('PAPER2_SOURCE_FILES_BEGIN', flush=True)
print(json.dumps(rows, indent=2), flush=True)
print('PAPER2_SOURCE_FILES_END', flush=True)

with tarfile.open(FINAL_ARCHIVE, 'w') as tf:
    for name in NAMES:
        tf.add(EXPORT_DIR / name, arcname=name, recursive=False)

final = {
    'path': str(FINAL_ARCHIVE),
    'size_bytes': FINAL_ARCHIVE.stat().st_size,
    'size_mb': round(FINAL_ARCHIVE.stat().st_size / (1024**2), 3),
    'sha256': sha256_file(FINAL_ARCHIVE),
    'contains': NAMES,
}
print('PAPER2_COMPLETE_BACKUP_BEGIN', flush=True)
print(json.dumps(final, indent=2), flush=True)
print('PAPER2_COMPLETE_BACKUP_END', flush=True)
print('PAPER2_COMPLETE_BACKUP_READY', flush=True)
'''


def request_json(method, url, **kwargs):
    r = requests.request(method, url, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def main():
    state = request_json('GET', f'{COORDINATOR_URL}/get_state')
    online = {n.get('node_id') for n in state.get('nodes', []) if n.get('status') == 'online'}
    print('ONLINE|' + ','.join(sorted(online)), flush=True)
    available = [t for t in TARGETS if t in online]
    if not available:
        print('TARGET_OFFLINE|Node10-GPU-T4-x2', flush=True)
        return 31

    d = request_json('POST', f'{COORDINATOR_URL}/run_code', json={'code': REMOTE_CODE, 'targets': available})
    tid = d.get('task_id')
    print(f'TASK|{tid}|targets={available}', flush=True)
    if not tid:
        return 32

    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        result = request_json('GET', f'{COORDINATOR_URL}/get_task_result/{tid}')
        status = result.get('status')
        responses = result.get('responses', {})
        print(f'POLL|{status}|responses={len(responses)}', flush=True)
        if status == 'completed':
            data = responses.get(available[0])
            if not data:
                print('MISSING_RESPONSE', flush=True)
                return 40
            if data.get('error'):
                print('REMOTE_ERROR|' + str(data.get('error')), flush=True)
                return 41
            out = data.get('output', '')
            print(out, flush=True)
            return 0 if 'PAPER2_COMPLETE_BACKUP_READY' in out else 42
        if status in {'failed','error','cancelled','overwritten'}:
            print('TASK_FAILED|' + str(result), flush=True)
            return 43
        time.sleep(POLL_SECONDS)
    print('TIMEOUT', flush=True)
    return 50

if __name__ == '__main__':
    sys.exit(main())
