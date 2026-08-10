import sys
import time
from pathlib import Path
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
SCRIPT_PATH = Path("paper2_backup_outputs.py")
TARGETS = ["Node2-GPU-T4-x2", "Node10-GPU-T4-x2"]
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 1800


def request_json(method, url, **kwargs):
    r = requests.request(method, url, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def main():
    if not SCRIPT_PATH.exists():
        print(f"Missing script: {SCRIPT_PATH}", flush=True)
        return 2
    code = SCRIPT_PATH.read_text(encoding="utf-8")
    state = request_json("GET", f"{COORDINATOR_URL}/get_state")
    online = {n.get("node_id"): n for n in state.get("nodes", []) if n.get("status") == "online"}
    print("Online worker nodes:", sorted(online), flush=True)
    tasks = {}
    for node in TARGETS:
        if node not in online:
            print(f"BACKUP_NODE_OFFLINE|{node}", flush=True)
            continue
        d = request_json("POST", f"{COORDINATOR_URL}/run_code", json={"code": code, "targets": [node]})
        tid = d.get("task_id")
        if tid:
            tasks[node] = tid
            print(f"BACKUP_DISPATCHED|{node}|{tid}", flush=True)
    if not tasks:
        return 31
    deadline = time.time() + MAX_WAIT_SECONDS
    done = set(); had_error = False
    while time.time() < deadline and len(done) < len(tasks):
        for node, tid in tasks.items():
            if node in done: continue
            result = request_json("GET", f"{COORDINATOR_URL}/get_task_result/{tid}")
            status = result.get("status")
            responses = result.get("responses", {})
            print(f"POLL|{node}|{status}|responses={len(responses)}", flush=True)
            if status == "completed":
                done.add(node)
                data = responses.get(node) or (next(iter(responses.values())) if responses else None)
                print(f"===== {node} BACKUP COMPLETE =====", flush=True)
                if not data or data.get("error"):
                    had_error = True
                    print("ERROR|", None if not data else data.get("error"), flush=True)
                else:
                    print(data.get("output", ""), flush=True)
            elif status in {"failed","error","cancelled","overwritten"}:
                done.add(node); had_error=True
                print(f"BACKUP_FAILED|{node}|{result}", flush=True)
        if len(done) < len(tasks): time.sleep(POLL_SECONDS)
    if len(done) < len(tasks):
        print("BACKUP_TIMEOUT|", sorted(set(tasks)-done), flush=True)
        return 50
    return 40 if had_error else 0

if __name__ == "__main__":
    sys.exit(main())
