import json
import time

import paper2_microstep_dispatch as d


def wait_task(task_id, timeout=60):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = d.req('GET', '/get_task_result/' + task_id)
        status = last.get('status')
        if status == 'completed':
            return last
        if status in {'failed', 'error', 'cancelled'}:
            raise RuntimeError(f'recovery task ended {status}: {last!r}')
        time.sleep(1)
    raise TimeoutError(f'recovery task timeout {task_id}: {last!r}')


def clear_stale_foreground():
    state = d.req('GET', '/get_state')
    execution = state.get('execution') or {}
    if execution.get('status') != 'running':
        print('RECOVERY|coordinator already idle', flush=True)
        return

    old_id = execution.get('current_task_id')
    targets = [t for t in execution.get('node_targets', []) if t in d.PLAN]
    if not targets:
        targets = ['Node10-GPU-T4-x2']

    old = None
    if old_id:
        try:
            old = d.req('GET', '/get_task_result/' + old_id)
        except Exception as exc:
            print('RECOVERY|old task lookup failed|' + repr(exc), flush=True)

    responses = (old or {}).get('responses') or {}
    print('RECOVERY|stale candidate|' + json.dumps({
        'task_id': old_id,
        'task_status': (old or {}).get('status'),
        'responses': list(responses),
        'targets': targets,
    }, sort_keys=True), flush=True)

    new = d.req('POST', '/run_code', json={
        'code': "print('MICRO_RECOVERY_RESET_OK', flush=True)",
        'targets': [targets[0]],
    })
    new_id = new.get('task_id')
    if not new_id:
        raise RuntimeError('recovery reset returned no task_id: ' + repr(new))
    print(f'RECOVERY|reset task|{new_id}|{targets[0]}', flush=True)
    result = wait_task(new_id, 90)
    print('RECOVERY|reset result|' + json.dumps({
        'status': result.get('status'),
        'responses': list((result.get('responses') or {}).keys()),
    }, sort_keys=True), flush=True)

    end = time.time() + 30
    while time.time() < end:
        state = d.req('GET', '/get_state')
        if (state.get('execution') or {}).get('status') != 'running':
            print('RECOVERY|coordinator idle', flush=True)
            return
        time.sleep(1)
    raise RuntimeError('coordinator did not return idle after recovery reset')


def safe_boot(target, folds, aug, src):
    runner = f'''import importlib.util
import traceback
from pathlib import Path

p = Path('/kaggle/working/paper2_focused_revision_20260820.py')
spec = importlib.util.spec_from_file_location('focused', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

try:
    print('MICRO_NODE_START|{target}', flush=True)
    m.download_ham()
    df = m.metadata()
    for f in {folds!r}:
        print('MICRO_FOLD_START|' + str(f), flush=True)
        m.run_crossfit_outer(df, f)
        m.archive_component('crossfit_outer_' + str(f), m.OUTROOT / 'crossfit' / ('outer_' + str(f)))
        print('MICRO_FOLD_DONE|' + str(f), flush=True)
    if {aug!r}:
        print('MICRO_AUG_START', flush=True)
        m.run_augmentation_pair(df)
        m.archive_component('augmentation_sensitivity', m.OUTROOT / 'augmentation_sensitivity')
        print('MICRO_AUG_DONE', flush=True)
    print('MICRO_NODE_DONE|{target}', flush=True)
except Exception:
    print('MICRO_NODE_ERROR|' + traceback.format_exc(), flush=True)
    raise
'''

    code = f'''import os
import subprocess
import sys
import time
from pathlib import Path

root = Path('/kaggle/working/paper2_microsteps')
root.mkdir(parents=True, exist_ok=True)
pidf = root / 'worker.pid'
alive = False
pid = None
if pidf.exists():
    try:
        pid = int(pidf.read_text())
        os.kill(pid, 0)
        alive = True
    except Exception:
        alive = False

Path('/kaggle/working/paper2_focused_revision_20260820.py').write_text({src!r}, encoding='utf-8')
Path('/kaggle/working/paper2_micro_runner.py').write_text({runner!r}, encoding='utf-8')

if alive:
    print('MICRO_ALREADY_RUNNING|' + str(pid), flush=True)
else:
    log = open(root / 'worker.log', 'ab', buffering=0)
    proc = subprocess.Popen(
        [sys.executable, '/kaggle/working/paper2_micro_runner.py'],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd='/kaggle/working',
        start_new_session=True,
        close_fds=True,
    )
    pidf.write_text(str(proc.pid))
    log.close()
    time.sleep(0.25)
    print('MICRO_STARTED|' + str(proc.pid) + '|poll=' + str(proc.poll()), flush=True)
'''

    response = d.task(code, [target], 120)[target]
    output = response.get('output') or response.get('stdout') or ''
    print(output, flush=True)
    if response.get('error'):
        raise RuntimeError(str(response.get('error')) + '\n' + output[-4000:])


def safe_query(targets):
    code = '''import json
import os
from pathlib import Path

root = Path('/kaggle/working/paper2_microsteps')
pid = None
alive = False
try:
    pid = int((root / 'worker.pid').read_text())
    os.kill(pid, 0)
    alive = True
except Exception:
    pass

out = {
    'pid': pid,
    'alive': alive,
    'log': (root / 'worker.log').read_text(encoding='utf-8', errors='replace') if (root / 'worker.log').exists() else '',
    'summaries': {},
    'augmentation': None,
}
for fold in range(5):
    p = Path(f'/kaggle/working/paper2_data/focused_revision_20260820/crossfit/outer_{fold}/summary.json')
    if p.exists():
        try:
            out['summaries'][str(fold)] = json.loads(p.read_text())
        except Exception:
            pass
p = Path('/kaggle/working/paper2_data/focused_revision_20260820/augmentation_sensitivity/paired_summary.json')
if p.exists():
    try:
        out['augmentation'] = json.loads(p.read_text())
    except Exception:
        pass
print('<<<S>>>')
print(json.dumps(out, default=str))
print('<<<E>>>')
'''

    responses = d.task(code, targets, 120)
    parsed = {}
    for target, response in responses.items():
        text = response.get('output') or response.get('stdout') or ''
        if response.get('error'):
            parsed[target] = {
                'alive': False,
                'log': 'QUERY_REMOTE_ERROR|' + str(response.get('error')) + '\n' + text[-3000:],
                'summaries': {},
            }
            continue
        a = text.find('<<<S>>>')
        b = text.find('<<<E>>>', a + 7)
        if a >= 0 and b >= 0:
            parsed[target] = json.loads(text[a + 7:b].strip())
        else:
            parsed[target] = {
                'alive': False,
                'log': 'QUERY_PARSE_ERROR|' + text[-3000:],
                'summaries': {},
            }
    return parsed


def main():
    clear_stale_foreground()
    d.boot = safe_boot
    d.query = safe_query
    return d.main()


if __name__ == '__main__':
    raise SystemExit(main())
