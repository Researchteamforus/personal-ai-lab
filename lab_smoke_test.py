import json
import sys
import time

import requests

BASE = "https://miomiomiomizan-personal-ai-lab.hf.space"
TARGET = "Node2-GPU-T4-x2"


def req(method, path, **kwargs):
    r = requests.request(method, BASE + path, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def main():
    state = req("GET", "/get_state")
    nodes = {n.get("node_id"): n for n in state.get("nodes", [])}
    node = nodes.get(TARGET)
    if not node or node.get("status") != "online":
        raise RuntimeError(f"Target not online: {TARGET}; node={node!r}")

    execution = state.get("execution", {})
    if execution.get("status") == "running":
        raise RuntimeError(
            "Coordinator already reports a running task; refusing to overwrite it: "
            + str(execution.get("current_task_id"))
        )

    code = r'''
import json, socket, time
import torch

info = {
    "hostname": socket.gethostname(),
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
}

if not torch.cuda.is_available():
    print("SMOKE_RESULT|" + json.dumps(info, sort_keys=True), flush=True)
    raise RuntimeError("CUDA is not available")

idx = torch.cuda.current_device()
info["device_index"] = int(idx)
info["device_name"] = torch.cuda.get_device_name(idx)
info["total_memory_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 3)

torch.cuda.synchronize()
t0 = time.time()
a = torch.randn((512, 512), device="cuda", dtype=torch.float32)
b = torch.randn((512, 512), device="cuda", dtype=torch.float32)
c = a @ b
torch.cuda.synchronize()

info["matmul_seconds"] = round(time.time() - t0, 6)
info["checksum"] = round(float(c[0, 0].item()), 6)
info["allocated_mb"] = round(torch.cuda.memory_allocated(idx) / (1024**2), 3)
info["reserved_mb"] = round(torch.cuda.memory_reserved(idx) / (1024**2), 3)
print("SMOKE_RESULT|" + json.dumps(info, sort_keys=True), flush=True)

del a, b, c
torch.cuda.empty_cache()
print("SMOKE_DONE", flush=True)
'''

    submitted = req("POST", "/run_code", json={"code": code, "targets": [TARGET]})
    task_id = submitted.get("task_id")
    if not task_id:
        raise RuntimeError("No task_id returned: " + repr(submitted))

    print(f"SMOKE_TASK|{task_id}|{TARGET}", flush=True)
    deadline = time.time() + 120

    while time.time() < deadline:
        result = req("GET", f"/get_task_result/{task_id}")
        status = result.get("status")
        responses = result.get("responses", {})
        print(f"SMOKE_POLL|{status}|responses={len(responses)}", flush=True)

        if status == "completed":
            response = responses.get(TARGET)
            if not response:
                raise RuntimeError("Task completed without target response")
            output = response.get("output") or response.get("stdout") or ""
            print(output, flush=True)
            if response.get("error"):
                raise RuntimeError(str(response.get("error")))
            if "SMOKE_DONE" not in output:
                raise RuntimeError("Smoke-test completion marker missing")
            print("LAB_SMOKE_TEST_OK", flush=True)
            return 0

        if status in {"failed", "error", "cancelled", "overwritten"}:
            raise RuntimeError(f"Task ended with status={status}: {json.dumps(result, default=str)}")

        time.sleep(1)

    raise TimeoutError(task_id)


if __name__ == "__main__":
    sys.exit(main())
