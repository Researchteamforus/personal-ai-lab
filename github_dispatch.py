import sys, time
from pathlib import Path
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
SCRIPT_PATH = Path("paper2_backup_outputs.py")
TARGETS = ["Node10-GPU-T4-x2", "Node2-GPU-T4-x2"]
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 900


def request_json(method, url, **kwargs):
    r = requests.request(method, url, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def main():
    if not SCRIPT_PATH.exists():
        print(f"MISSING_SCRIPT|{SCRIPT_PATH}", flush=True)
        return 2
    backup_code = SCRIPT_PATH.read_text(encoding="utf-8")
    state = request_json("GET", f"{COORDINATOR_URL}/get_state")
    online = {n.get("node_id") for n in state.get("nodes", []) if n.get("status") == "online"}
    print("ONLINE|" + ",".join(sorted(online)), flush=True)
    available = [t for t in TARGETS if t in online]
    missing = [t for t in TARGETS if t not in online]
    for t in missing:
        print(f"TARGET_OFFLINE|{t}", flush=True)
    if not available:
        return 31

    # One task across all currently-online targets. Each node runs the same canonical
    # paper2_backup_outputs.py and reports either a verified backup or an explicit
    # no-data/error state. Originals are untouched.
    code = """
from pathlib import Path
try:
    exec(compile(BACKUP_SOURCE, 'paper2_backup_outputs.py', 'exec'), globals(), globals())
except FileNotFoundError as e:
    print('PAPER2_BACKUP_NO_DATA|' + str(e), flush=True)
except Exception as e:
    print('PAPER2_BACKUP_ERROR|' + repr(e), flush=True)
"""
    code = "BACKUP_SOURCE = " + repr(backup_code) + "\n" + code
    d = request_json("POST", f"{COORDINATOR_URL}/run_code", json={"code": code, "targets": available})
    tid = d.get("task_id")
    print(f"BACKUP_TASK|{tid}|targets={available}", flush=True)
    if not tid:
        return 32

    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        result = request_json("GET", f"{COORDINATOR_URL}/get_task_result/{tid}")
        status = result.get("status")
        responses = result.get("responses", {})
        print(f"POLL|{status}|responses={len(responses)}", flush=True)
        if status == "completed":
            bad = False
            for target in available:
                data = responses.get(target)
                print(f"===== {target} BACKUP RESPONSE =====", flush=True)
                if not data:
                    print("MISSING_RESPONSE", flush=True); bad = True; continue
                if data.get("error"):
                    print("REMOTE_ERROR|" + str(data.get("error")), flush=True); bad = True
                out = data.get("output", "")
                print(out, flush=True)
                if "PAPER2_BACKUP_COMPLETE" not in out and "PAPER2_BACKUP_NO_DATA" not in out:
                    bad = True
            return 40 if bad else 0
        if status in {"failed", "error", "cancelled", "overwritten"}:
            print("TASK_FAILED|" + str(result), flush=True)
            return 41
        time.sleep(POLL_SECONDS)
    print("TIMEOUT", flush=True)
    return 50


if __name__ == "__main__":
    sys.exit(main())
