import sys
import time
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
POLL_SECONDS = 3
MAX_WAIT_SECONDS = 180

TEST_CODE = r'''
import os
from pathlib import Path

print("=== KAGGLE PAPER 2 ASSET CHECK ===")

roots = [Path('/kaggle/input'), Path('/kaggle/working')]
keywords = (
    'ham10000', 'skin', 'lesion', 'resnet', 'dropout', 'uncertainty',
    'paper2', 'paper_2', 'prediction', 'ood', 'metadata', 'checkpoint', 'model'
)
interesting_ext = {
    '.py', '.ipynb', '.pt', '.pth', '.ckpt', '.h5', '.keras', '.onnx',
    '.npy', '.npz', '.csv', '.json', '.parquet', '.pkl', '.joblib'
}

for root in roots:
    print('ROOT|', root, '|exists=', root.exists())
    if not root.exists():
        continue
    try:
        top = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except Exception as exc:
        print('ROOT_ERROR|', root, '|', repr(exc))
        continue

    print('TOP_LEVEL_COUNT|', root, '|', len(top))
    for entry in top[:200]:
        try:
            kind = 'DIR' if entry.is_dir() else 'FILE'
            size = entry.stat().st_size / (1024*1024) if entry.is_file() else 0.0
        except Exception:
            kind, size = 'UNKNOWN', 0.0
        print(f'TOP|{kind}|{size:.3f} MB|{entry}')

    matches = []
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth
        if depth >= 5:
            dirnames[:] = []
        for name in filenames:
            p = current / name
            low = str(p).lower()
            if p.suffix.lower() in interesting_ext and any(k in low for k in keywords):
                try:
                    size = p.stat().st_size / (1024*1024)
                except Exception:
                    size = -1
                matches.append((str(p), size))
                if len(matches) >= 250:
                    break
        if len(matches) >= 250:
            break

    print('MATCH_COUNT|', root, '|', len(matches))
    for path, size in sorted(matches):
        print(f'ASSET|{size:.3f} MB|{path}')

print('KAGGLE_ASSET_CHECK_DONE')
'''


def request_json(method, url, **kwargs):
    response = requests.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    state = request_json("GET", f"{COORDINATOR_URL}/get_state")
    online = [n for n in state.get("nodes", []) if n.get("status") == "online"]
    print(f"Online worker nodes: {len(online)}")
    if not online:
        return 20

    dispatch = request_json(
        "POST", f"{COORDINATOR_URL}/run_code",
        json={"code": TEST_CODE, "targets": []},
    )
    task_id = dispatch.get("task_id")
    if not task_id:
        print("No task_id:", dispatch)
        return 31

    print("Task ID:", task_id)
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        result = request_json("GET", f"{COORDINATOR_URL}/get_task_result/{task_id}")
        status = result.get("status")
        responses = result.get("responses", {})
        print(f"Status: {status}; responses: {len(responses)}")
        if status == "completed":
            had_error = False
            for node_id, data in responses.items():
                print(f"\n===== {node_id} =====")
                if data.get("error"):
                    had_error = True
                    print(data.get("error"))
                else:
                    print(data.get("output", ""))
            return 40 if had_error else 0
        if status in {"failed", "error", "cancelled"}:
            print(result)
            return 41
        time.sleep(POLL_SECONDS)
    return 50


if __name__ == "__main__":
    sys.exit(main())
