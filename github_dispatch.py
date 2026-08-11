import sys, time
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
TARGETS = ["Node10-GPU-T4-x2"]
POLL_SECONDS = 3
MAX_WAIT_SECONDS = 300

REMOTE_CODE = r'''
from pathlib import Path
from IPython.display import FileLink

p = Path('/kaggle/working/Paper2_COMPLETE_BACKUP.tar')
if not p.exists():
    raise FileNotFoundError(str(p))

link = FileLink(str(p), result_html_prefix='<b>Download complete Paper 2 backup:</b> ')
print('PAPER2_DOWNLOAD_FILE|' + str(p), flush=True)
print('PAPER2_DOWNLOAD_SIZE|' + str(p.stat().st_size), flush=True)
try:
    html = link._repr_html_()
except Exception:
    html = repr(link)
print('PAPER2_FILELINK_HTML_BEGIN', flush=True)
print(html, flush=True)
print('PAPER2_FILELINK_HTML_END', flush=True)
print('PAPER2_FILELINK_READY', flush=True)
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
            return 0 if 'PAPER2_FILELINK_READY' in out else 42
        if status in {'failed','error','cancelled','overwritten'}:
            print('TASK_FAILED|' + str(result), flush=True)
            return 43
        time.sleep(POLL_SECONDS)
    print('TIMEOUT', flush=True)
    return 50

if __name__ == '__main__':
    sys.exit(main())
