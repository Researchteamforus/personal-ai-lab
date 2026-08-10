import sys
import time
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
POLL_SECONDS = 3
MAX_WAIT_SECONDS = 180

TEST_CODE = r'''
import platform
print("=== GITHUB -> PERSONAL AI LAB CONNECTION TEST ===")
print("Python:", platform.python_version())
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.randn(512, 512, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("GPU tensor test sum:", float(y.sum().item()))
    else:
        print("Worker is running on CPU")
except Exception as exc:
    print("Torch/GPU test error:", repr(exc))
print("LAB_CONNECTION_TEST_SUCCESS")
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
        print(
            " -",
            node.get("node_id"),
            "| GPU:", node.get("gpu_name"),
            "| VRAM:", node.get("gpu_vram"),
            "| RAM:", node.get("total_ram"),
        )

    if not online:
        print("NO_ONLINE_WORKERS: Start at least one worker and rerun the workflow.")
        return 20

    try:
        dispatch = request_json(
            "POST",
            f"{COORDINATOR_URL}/run_code",
            json={"code": TEST_CODE, "targets": []},
        )
    except Exception as exc:
        print(f"ERROR: /run_code request failed: {exc}")
        return 30

    task_id = dispatch.get("task_id")
    if not task_id:
        print("ERROR: Coordinator returned no task_id:", dispatch)
        return 31

    print("Task ID:", task_id)
    deadline = time.time() + MAX_WAIT_SECONDS

    while time.time() < deadline:
        try:
            result = request_json(
                "GET", f"{COORDINATOR_URL}/get_task_result/{task_id}"
            )
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

    print(f"TIMEOUT: No completed result after {MAX_WAIT_SECONDS} seconds")
    return 50


if __name__ == "__main__":
    sys.exit(main())
