# GitHub-to-lab Paper 2 asset inventory. Lists relevant paths only; does not read file contents.
import sys
import time
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
POLL_SECONDS = 3
MAX_WAIT_SECONDS = 180

TEST_CODE = r'''
import os
from pathlib import Path

print("=== PAPER 2 ASSET INVENTORY ===")
print("cwd:", os.getcwd())
print("home:", str(Path.home()))

roots = []
for p in [Path.cwd(), Path.home(), Path('/workspace'), Path('/content'), Path('/mnt/data')]:
    try:
        rp = p.resolve()
    except Exception:
        continue
    if rp.exists() and rp.is_dir() and rp not in roots:
        roots.append(rp)

keywords = (
    'ham10000', 'ham_10000', 'skin', 'lesion', 'resnet', 'dropout',
    'mc_dropout', 'uncertainty', 'paper2', 'paper_2', 'prediction',
    'ood', 'metadata', 'lesion_id', 'checkpoint', 'model'
)
interesting_ext = {
    '.py', '.ipynb', '.pt', '.pth', '.ckpt', '.h5', '.keras', '.onnx',
    '.npy', '.npz', '.csv', '.json', '.parquet', '.pkl', '.joblib'
}
skip_dirs = {
    '.git', '.cache', '__pycache__', 'node_modules', 'site-packages',
    '.local', '.npm', '.conda', 'anaconda3', 'miniconda3', 'venv', '.venv'
}

matches = []
seen = set()
max_files = 300
max_depth = 6

for root in roots:
    print("SCAN_ROOT:", root)
    root_parts = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        depth = len(current.parts) - root_parts
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith('.cache')]
        if depth >= max_depth:
            dirnames[:] = []

        for name in filenames:
            p = current / name
            low = str(p).lower()
            if p.suffix.lower() not in interesting_ext:
                continue
            if not any(k in low for k in keywords):
                continue
            try:
                rp = str(p.resolve())
                if rp in seen:
                    continue
                seen.add(rp)
                size_mb = p.stat().st_size / (1024 * 1024)
            except Exception:
                continue
            matches.append((rp, size_mb))
            if len(matches) >= max_files:
                break
        if len(matches) >= max_files:
            break
    if len(matches) >= max_files:
        break

print("MATCH_COUNT:", len(matches))
for path, size_mb in sorted(matches):
    print(f"ASSET|{size_mb:.3f} MB|{path}")

# Lightweight dataset/model directory hints without recursive content reads.
for root in roots:
    try:
        entries = list(root.iterdir())[:200]
    except Exception:
        continue
    for entry in entries:
        low = entry.name.lower()
        if entry.is_dir() and any(k in low for k in keywords):
            print("DIR_HINT|", str(entry))

print("PAPER2_ASSET_INVENTORY_DONE")
'''


def request_json(method, url, **kwargs):
    response = requests.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    print(f"Coordinator: {COORDINATOR_URL}")
    try:
        state = request_json("GET", f"{COORDINATOR_URL}/get_state")
    except Exception as exc:
        print(f"ERROR: Could not reach coordinator/get_state: {exc}")
        return 10

    nodes = state.get("nodes", [])
    online = [n for n in nodes if n.get("status") == "online"]
    print(f"Online worker nodes: {len(online)}")
    for node in online:
        print(" -", node.get("node_id"), "| GPU:", node.get("gpu_name"))

    if not online:
        print("NO_ONLINE_WORKERS")
        return 20

    try:
        dispatch = request_json(
            "POST", f"{COORDINATOR_URL}/run_code",
            json={"code": TEST_CODE, "targets": []},
        )
    except Exception as exc:
        print(f"ERROR: /run_code request failed: {exc}")
        return 30

    task_id = dispatch.get("task_id")
    if not task_id:
        print("ERROR: no task_id", dispatch)
        return 31

    print("Task ID:", task_id)
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        try:
            result = request_json("GET", f"{COORDINATOR_URL}/get_task_result/{task_id}")
        except Exception as exc:
            print("Polling error:", exc)
            time.sleep(POLL_SECONDS)
            continue

        status = result.get("status")
        responses = result.get("responses", {})
        print(f"Status: {status}; responses: {len(responses)}")
        if status == "completed":
            had_error = False
            for node_id, data in responses.items():
                print(f"\n===== {node_id} =====")
                if data.get("error"):
                    had_error = True
                    print("ERROR:")
                    print(data.get("error"))
                else:
                    print("OUTPUT:")
                    print(data.get("output", ""))
            return 40 if had_error else 0
        if status in {"failed", "error", "cancelled"}:
            print("Task ended unsuccessfully:", result)
            return 41
        time.sleep(POLL_SECONDS)

    print("TIMEOUT")
    return 50


if __name__ == "__main__":
    sys.exit(main())
