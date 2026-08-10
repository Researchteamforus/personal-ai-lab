import sys
import time
from pathlib import Path
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
EXPERIMENT_PATH = Path("paper2_multiseed_runner.py")
ASSIGNMENTS = {
    "Node2-GPU-T4-x2": [11, 29],
    "Node10-GPU-T4-x2": [47, 71],
}
POLL_SECONDS = 10
MAX_WAIT_SECONDS = 7200


def request_json(method, url, **kwargs):
    response = requests.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    if not EXPERIMENT_PATH.exists():
        print(f"Missing experiment file: {EXPERIMENT_PATH}", flush=True)
        return 2
    base_code = EXPERIMENT_PATH.read_text(encoding="utf-8")
    state = request_json("GET", f"{COORDINATOR_URL}/get_state")
    online = {n.get("node_id"): n for n in state.get("nodes", []) if n.get("status") == "online"}
    print("Experiment:", EXPERIMENT_PATH, flush=True)
    print("Online worker nodes:", sorted(online), flush=True)

    tasks = {}
    for node_id, seeds in ASSIGNMENTS.items():
        if node_id not in online:
            print(f"ASSIGNMENT_ERROR|{node_id}|offline|seeds={seeds}", flush=True)
            return 31
        code = f"TRAIN_SEEDS = {seeds!r}\n" + base_code
        dispatch = request_json("POST", f"{COORDINATOR_URL}/run_code", json={"code": code, "targets": [node_id]})
        task_id = dispatch.get("task_id")
        if not task_id:
            print(f"NO_TASK_ID|{node_id}|{dispatch}", flush=True)
            return 32
        tasks[node_id] = task_id
        print(f"DISPATCHED|{node_id}|seeds={seeds}|task_id={task_id}", flush=True)

    deadline = time.time() + MAX_WAIT_SECONDS
    done = set()
    had_error = False
    while time.time() < deadline and len(done) < len(tasks):
        for node_id, task_id in tasks.items():
            if node_id in done:
                continue
            result = request_json("GET", f"{COORDINATOR_URL}/get_task_result/{task_id}")
            status = result.get("status")
            responses = result.get("responses", {})
            print(f"POLL|{node_id}|status={status}|responses={len(responses)}", flush=True)
            if status == "completed":
                done.add(node_id)
                data = responses.get(node_id)
                if data is None and responses:
                    data = next(iter(responses.values()))
                print(f"\n===== {node_id} COMPLETE =====", flush=True)
                if not data:
                    had_error = True
                    print("ERROR|missing response payload", flush=True)
                elif data.get("error"):
                    had_error = True
                    print("ERROR|", data.get("error"), flush=True)
                else:
                    print(data.get("output", ""), flush=True)
            elif status in {"failed", "error", "cancelled"}:
                done.add(node_id); had_error = True
                print(f"TASK_FAILED|{node_id}|{result}", flush=True)
        if len(done) < len(tasks):
            time.sleep(POLL_SECONDS)

    if len(done) < len(tasks):
        print("TIMEOUT|unfinished=", sorted(set(tasks)-done), flush=True)
        return 50
    print("MULTISEED_DISPATCH_COMPLETE", flush=True)
    return 40 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
