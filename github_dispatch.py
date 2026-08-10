import sys
import time
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
POLL_SECONDS = 3
MAX_WAIT_SECONDS = 180

TEST_CODE = r'''
from pathlib import Path

root = Path('/kaggle/input/datasets')
print('=== KAGGLE DATASET TREE ===')
print('ROOT_EXISTS|', root.exists())
if root.exists():
    count = 0
    stack = [(root, 0)]
    while stack and count < 500:
        current, depth = stack.pop(0)
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except Exception as exc:
            print('ERR|', current, '|', repr(exc))
            continue
        for e in entries:
            rel = e.relative_to(root)
            kind = 'DIR' if e.is_dir() else 'FILE'
            print(f'TREE|{depth}|{kind}|{rel}')
            count += 1
            if count >= 500:
                break
            if e.is_dir() and depth < 3:
                stack.append((e, depth + 1))
print('DATASET_TREE_DONE')
'''


def request_json(method, url, **kwargs):
    response = requests.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    state = request_json('GET', f'{COORDINATOR_URL}/get_state')
    online = [n for n in state.get('nodes', []) if n.get('status') == 'online']
    print('Online worker nodes:', len(online))
    if not online:
        return 20
    dispatch = request_json('POST', f'{COORDINATOR_URL}/run_code', json={'code': TEST_CODE, 'targets': []})
    task_id = dispatch.get('task_id')
    if not task_id:
        print('No task_id:', dispatch)
        return 31
    print('Task ID:', task_id)
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        result = request_json('GET', f'{COORDINATOR_URL}/get_task_result/{task_id}')
        status = result.get('status')
        responses = result.get('responses', {})
        print(f'Status: {status}; responses: {len(responses)}')
        if status == 'completed':
            for node_id, data in responses.items():
                print(f'\n===== {node_id} =====')
                if data.get('error'):
                    print(data.get('error'))
                else:
                    print(data.get('output', ''))
            return 0
        if status in {'failed', 'error', 'cancelled'}:
            print(result)
            return 41
        time.sleep(POLL_SECONDS)
    return 50


if __name__ == '__main__':
    sys.exit(main())
