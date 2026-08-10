import sys
import time
from pathlib import Path
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
EXPERIMENT_PATH = Path("experiments/paper2_statistical_revision.py")
TARGETS = ["Node10-GPU-T4-x2"]
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 1800


def request_json(method, url, **kwargs):
    response = requests.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    if not EXPERIMENT_PATH.exists():
        print(f"Missing experiment file: {EXPERIMENT_PATH}")
        return 2
    code = EXPERIMENT_PATH.read_text(encoding="utf-8")
    print(f"Experiment: {EXPERIMENT_PATH}")
    state = request_json("GET", f"{COORDINATOR_URL}/get_state")
    online = [n for n in state.get("nodes", []) if n.get("status") == "online"]
    print(f"Online worker nodes: {len(online)}")
    for n in online:
        print(" -", n.get("node_id"), "| GPU:", n.get("gpu_name"))
    dispatch = request_json(
        "POST", f"{COORDINATOR_URL}/run_code",
        json={"code": code, "targets": TARGETS},
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
                    print("ERROR:")
                    print(data.get("error"))
                else:
                    print(data.get("output", ""))
            return 40 if had_error else 0
        if status in {"failed", "error", "cancelled"}:
            print(result)
            return 41
        time.sleep(POLL_SECONDS)
    print("TIMEOUT")
    return 50


if __name__ == "__main__":
    sys.exit(main())
