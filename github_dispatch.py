import base64, hashlib, json, sys, time
from pathlib import Path
import requests

BASE = 'https://miomiomiomizan-personal-ai-lab.hf.space'
TARGETS = ['Node2-GPU-T4-x2', 'Node10-GPU-T4-x2', 'Nabila-GPU-T4-x2']
ROLES = {
    'Node2-GPU-T4-x2': 'crossfit_outer_0_1',
    'Node10-GPU-T4-x2': 'crossfit_outer_2_3',
    'Nabila-GPU-T4-x2': 'crossfit_outer_4_augmentation',
}
SCRIPT = Path('experiments/paper2_focused_revision_20260820.py')
OUT = Path('paper2_artifacts')
OUT.mkdir(exist_ok=True)


def req(method, url, **kw):
    r = requests.request(method, url, timeout=120, **kw)
    r.raise_for_status()
    return r.json()


def submit(code, targets):
    d = req('POST', BASE + '/run_code', json={'code': code, 'targets': targets})
    tid = d.get('task_id')
    if not tid:
        raise RuntimeError('No task id: ' + repr(d))
    return tid


def wait(tid, targets, seconds=16200, poll=10):
    end = time.time() + seconds
    while time.time() < end:
        r = req('GET', BASE + '/get_task_result/' + tid)
        st = r.get('status')
        resp = r.get('responses', {})
        print(f'POLL|{tid}|{st}|responses={len(resp)}/{len(targets)}', flush=True)
        if st == 'completed':
            missing = [t for t in targets if t not in resp]
            if missing:
                raise RuntimeError('missing responses ' + repr(missing))
            for t in targets:
                d = resp[t]
                out = d.get('output') or d.get('stdout') or ''
                (OUT / f"remote_{ROLES[t]}.txt").write_text(out, encoding='utf-8')
                print(f'REMOTE_TAIL|{t}|' + out[-14000:], flush=True)
                if d.get('error'):
                    raise RuntimeError(f'{t}: {d.get("error")}\n{out[-5000:]}')
                if 'FOCUSED_REVISION_DONE|' not in out:
                    raise RuntimeError(f'{t}: completion marker missing')
            return resp
        if st in {'failed', 'error', 'cancelled', 'overwritten'}:
            raise RuntimeError(f'{tid}: {st}')
        time.sleep(poll)
    raise TimeoutError(tid)


def remote(target, code, seconds=300):
    tid = submit(code, [target])
    end = time.time() + seconds
    while time.time() < end:
        r = req('GET', BASE + '/get_task_result/' + tid)
        st = r.get('status')
        resp = r.get('responses', {})
        if st == 'completed':
            d = resp.get(target)
            if not d:
                raise RuntimeError('missing ' + target)
            out = d.get('output') or d.get('stdout') or ''
            if d.get('error'):
                raise RuntimeError(str(d.get('error')) + '\n' + out)
            return out
        if st in {'failed', 'error', 'cancelled', 'overwritten'}:
            raise RuntimeError(st)
        time.sleep(.6)
    raise TimeoutError(tid)


def between(t, a, b):
    i = t.find(a)
    j = t.find(b, i + len(a))
    if i < 0 or j < 0:
        raise ValueError((a, b, t[:300]))
    return t[i+len(a):j].strip()


def transfer(target, remote_path, local_path):
    raw = remote(
        target,
        f"from pathlib import Path;import hashlib,json;p=Path({remote_path!r});print('<<<META>>>');print(json.dumps({{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}));print('<<<ENDMETA>>>')"
    )
    meta = json.loads(between(raw, '<<<META>>>', '<<<ENDMETA>>>'))
    total = int(meta['size'])
    expected = meta['sha256']
    local_path.parent.mkdir(exist_ok=True)
    off = 0
    chunk = 1024 * 1024
    with local_path.open('wb') as f:
        while off < total:
            want = min(chunk, total - off)
            code = f"from pathlib import Path;import base64;p=Path({remote_path!r});h=open(p,'rb');h.seek({off});b=h.read({want});print('<<<B64>>>');print(base64.b64encode(b).decode());print('<<<ENDB64>>>')"
            s = remote(target, code, seconds=120)
            b = base64.b64decode(between(s, '<<<B64>>>', '<<<ENDB64>>>'))
            if len(b) != want:
                raise RuntimeError(f'{target} chunk {off}: {len(b)} != {want}')
            f.write(b)
            off += len(b)
            if off == total or off % (16 * 1024 * 1024) < chunk:
                print(f'TRANSFER|{target}|{off}/{total}', flush=True)
    got = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if got != expected:
        raise RuntimeError(f'{target} checksum {got} != {expected}')
    print('TRANSFER_VERIFIED|' + target + '|' + json.dumps(meta, sort_keys=True), flush=True)
    return meta


def main():
    state = req('GET', BASE + '/get_state')
    online = {n['node_id'] for n in state.get('nodes', []) if n.get('status') == 'online'}
    targets = [t for t in TARGETS if t in online]
    print('ONLINE|' + ','.join(sorted(online)), flush=True)
    if len(targets) < 3:
        raise RuntimeError('All three T4 workers are required; online=' + repr(sorted(online)))
    execution = state.get('execution', {})
    if execution.get('status') == 'running':
        raise RuntimeError('Coordinator already has a running task: ' + str(execution.get('current_task_id')))

    src = SCRIPT.read_text(encoding='utf-8')
    wrapper = "src=SCRIPT_TEXT\nexec(compile(src,'paper2_focused_revision_20260820.py','exec'),{'__name__':'__main__'})\n".replace('SCRIPT_TEXT', repr(src))
    tid = submit(wrapper, targets)
    print('FOCUSED_TASK|' + tid, flush=True)
    wait(tid, targets)

    manifest = {}
    for target in targets:
        role = ROLES[target]
        rp = f'/kaggle/working/paper2_data/Paper2_FOCUSED_{role}.tar.gz'
        lp = OUT / f'Paper2_FOCUSED_{role}.tar.gz'
        meta = transfer(target, rp, lp)
        manifest[target] = {'role': role, **meta}
        sha_text = remote(target, f"from pathlib import Path;print(Path({(rp + '.sha256')!r}).read_text())").strip()
        (OUT / f'Paper2_FOCUSED_{role}.tar.gz.sha256').write_text(sha_text + '\n', encoding='utf-8')

    (OUT / 'focused_revision_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print('FOCUSED_REVISION_GITHUB_READY|' + json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
