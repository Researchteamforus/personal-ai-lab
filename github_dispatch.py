import sys, time, base64, hashlib, json
from pathlib import Path
import requests

COORDINATOR_URL = "https://miomiomiomizan-personal-ai-lab.hf.space"
TARGET = "Node10-GPU-T4-x2"
REMOTE_PATH = "/kaggle/working/Paper2_COMPLETE_BACKUP.tar"
LOCAL_PATH = Path("paper2_artifacts/Paper2_COMPLETE_BACKUP.tar")
POLL_SECONDS = 1
MAX_REMOTE_WAIT = 180


def request_json(method, url, **kwargs):
    r = requests.request(method, url, timeout=60, **kwargs)
    r.raise_for_status()
    return r.json()


def run_remote(code):
    d = request_json("POST", f"{COORDINATOR_URL}/run_code", json={"code": code, "targets": [TARGET]})
    tid = d.get("task_id")
    if not tid:
        raise RuntimeError("Coordinator did not return task_id")
    deadline = time.time() + MAX_REMOTE_WAIT
    while time.time() < deadline:
        result = request_json("GET", f"{COORDINATOR_URL}/get_task_result/{tid}")
        status = result.get("status")
        if status == "completed":
            data = result.get("responses", {}).get(TARGET)
            if not data:
                raise RuntimeError(f"Missing response for {TARGET}")
            if data.get("error"):
                raise RuntimeError(f"Remote error: {data.get('error')}")
            return data.get("output", "")
        if status in {"failed", "error", "cancelled", "overwritten"}:
            raise RuntimeError(f"Remote task failed: {status}")
        time.sleep(POLL_SECONDS)
    raise TimeoutError("Remote task timed out")


def extract_between(text, begin, end):
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0 or j <= i:
        raise ValueError(f"Missing markers {begin}/{end}")
    return text[i + len(begin):j].strip()


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    state = request_json("GET", f"{COORDINATOR_URL}/get_state")
    online = {n.get("node_id") for n in state.get("nodes", []) if n.get("status") == "online"}
    print("ONLINE|" + ",".join(sorted(online)), flush=True)
    if TARGET not in online:
        print("TARGET_OFFLINE|" + TARGET, flush=True)
        return 31

    meta_code = f'''\nfrom pathlib import Path\nimport hashlib, json\np=Path({REMOTE_PATH!r})\nif not p.exists(): raise FileNotFoundError(str(p))\nh=hashlib.sha256()\nwith p.open("rb") as f:\n    while True:\n        b=f.read(8*1024*1024)\n        if not b: break\n        h.update(b)\nprint("META_BEGIN")\nprint(json.dumps({{"size":p.stat().st_size,"sha256":h.hexdigest()}}))\nprint("META_END")\n'''
    meta_out = run_remote(meta_code)
    meta = json.loads(extract_between(meta_out, "META_BEGIN", "META_END"))
    total = int(meta["size"])
    expected_sha = meta["sha256"]
    print(f"SOURCE|size={total}|sha256={expected_sha}", flush=True)

    chunk_size = 4 * 1024 * 1024
    offset = 0
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_PATH.open("wb") as out_f:
        chunk_index = 0
        while offset < total:
            want = min(chunk_size, total - offset)
            code = f'''\nfrom pathlib import Path\nimport base64\np=Path({REMOTE_PATH!r})\nwith p.open("rb") as f:\n    f.seek({offset})\n    b=f.read({want})\nprint("CHUNK_BEGIN")\nprint(base64.b64encode(b).decode("ascii"))\nprint("CHUNK_END")\n'''
            try:
                raw = run_remote(code)
                payload = extract_between(raw, "CHUNK_BEGIN", "CHUNK_END")
                data = base64.b64decode(payload, validate=True)
                if len(data) != want:
                    raise ValueError(f"chunk length {len(data)} != {want}")
            except Exception as e:
                if chunk_size > 1024 * 1024:
                    chunk_size //= 2
                    print(f"CHUNK_RETRY|offset={offset}|new_chunk={chunk_size}|reason={type(e).__name__}", flush=True)
                    continue
                raise
            out_f.write(data)
            offset += len(data)
            chunk_index += 1
            if chunk_index % 10 == 0 or offset == total:
                print(f"TRANSFER|{offset}/{total}|{100.0*offset/total:.1f}%|chunk={chunk_size}", flush=True)

    local_sha = sha256_file(LOCAL_PATH)
    if LOCAL_PATH.stat().st_size != total or local_sha != expected_sha:
        raise RuntimeError(f"Transfer verification failed: size={LOCAL_PATH.stat().st_size}/{total} sha={local_sha}/{expected_sha}")

    checksum_path = LOCAL_PATH.parent / "Paper2_COMPLETE_BACKUP.sha256"
    checksum_path.write_text(f"{local_sha}  {LOCAL_PATH.name}\n", encoding="utf-8")
    print(f"TRANSFER_VERIFIED|path={LOCAL_PATH}|size={total}|sha256={local_sha}", flush=True)
    print("PAPER2_GITHUB_ARTIFACT_READY", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
