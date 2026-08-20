import json
import time
from pathlib import Path

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
    old_status = (old or {}).get('status')
    print('RECOVERY|stale candidate|' + json.dumps({
        'task_id': old_id,
        'task_status': old_status,
        'responses': list(responses),
        'targets': targets,
    }, sort_keys=True), flush=True)

    # The persisted state has been stuck for hours with zero responses. Supersede it
    # with a tiny task so the coordinator can return to an idle state.
    code = "print('MICRO_RECOVERY_RESET_OK', flush=True)"
    new = d.req('POST', '/run_code', json={'code': code, 'targets': [targets[0]]})
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
        execution = state.get('execution') or {}
        if execution.get('status') != 'running':
            print('RECOVERY|coordinator idle', flush=True)
            return
        time.sleep(1)
    raise RuntimeError('coordinator did not return idle after recovery reset')


def safe_boot(target, folds, aug, src):
    runner = f'''import importlib.util,traceback\nfrom pathlib import Path\np=Path('/kaggle/working/paper2_focused_revision_20260820.py')\ns=importlib.util.spec_from_file_location('focused',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\ntry:\n print('MICRO_NODE_START|{target}',flush=True);m.download_ham();df=m.metadata()\n for f in {folds!r}:\n  print('MICRO_FOLD_START|'+str(f),flush=True);m.run_crossfit_outer(df,f);m.archive_component('crossfit_outer_'+str(f),m.OUTROOT/'crossfit'/('outer_'+str(f)));print('MICRO_FOLD_DONE|'+str(f),flush=True)\n if {aug!r}:\n  print('MICRO_AUG_START',flush=True);m.run_augmentation_pair(df);m.archive_component('augmentation_sensitivity',m.OUTROOT/'augmentation_sensitivity');print('MICRO_AUG_DONE',flush=True)\n print('MICRO_NODE_DONE|{target}',flush=True)\nexcept Exception:\n print('MICRO_NODE_ERROR|'+traceback.format_exc(),flush=True);raise\n'''
    code = f'''import os,subprocess,sys,time\nfrom pathlib import Path\nr=Path('/kaggle/working/paper2_microsteps');r.mkdir(parents=True,exist_ok=True);pidf=r/'worker.pid';alive=False\nif pidf.exists():\n try:\n  pid=int(pidf.read_text());os.kill(pid,0);alive=True\n except Exception: alive=False\nPath('/kaggle/working/paper2_focused_revision_20260820.py').write_text({src!r},encoding='utf-8')\nPath('/kaggle/working/paper2_micro_runner.py').write_text({runner!r},encoding='utf-8')\nif alive:\n print('MICRO_ALREADY_RUNNING|'+str(pid),flush=True)\nelse:\n log=open(r/'worker.log','ab',buffering=0)\n p=subprocess.Popen([sys.executable,'/kaggle/working/paper2_micro_runner.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd='/kaggle/working',start_new_session=True,close_fds=True)\n pidf.write_text(str(p.pid));log.close();time.sleep(0.25);print('MICRO_STARTED|'+str(p.pid)+'|poll='+str(p.poll()),flush=True)\n'''
    response = d.task(code, [target], 120)[target]
    output = response.get('output') or response.get('stdout') or ''
    print(output, flush=True)
    if response.get('error'):
        raise RuntimeError(str(response.get('error')))


def main():
    clear_stale_foreground()
    d.boot = safe_boot
    return d.main()


if __name__ == '__main__':
    raise SystemExit(main())
