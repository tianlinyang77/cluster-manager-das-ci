# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Observe real training evidence and target one owned worker precisely.

Importing this module performs no process, container, or network operation.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from lib.evidence import STError


_LOCAL_SAVE = re.compile(r"\bsuccessfully saved local checkpoint from iteration\s+(\d+)\b")
_GLOBAL_SAVE = re.compile(r"\bsuccessfully saved checkpoint from iteration\s+(\d+)\b")
_SAVE_START = re.compile(r"\bsaving checkpoint at iteration\s+(\d+)\s+to\s+.+?\s+in\s+(\w+)\s+format\b")
_LOAD = re.compile(
    r"\bsuccessfully loaded checkpoint from\s+(.+?)\s+"
    r"(?:\[[^\r\n]*\]\s+)?at iteration\s+(\d+)\b"
)
_ITERATION = re.compile(
    r"\biteration\s+(\d+)\s*/\s*\d+\b.*\belapsed time per iteration\b"
)


def parse_training_log(text: str) -> dict:
    """Extract completed saves/loads and actual Megatron progress, in log order."""
    result = dict(local_saves=[], global_saves=[], loads=[], iterations=[], save_formats={}, load_progress=[])
    for line in text.splitlines():
        if match := _SAVE_START.search(line):
            result["save_formats"][int(match.group(1))] = match.group(2)
        if match := _LOCAL_SAVE.search(line):
            result["local_saves"].append(int(match.group(1)))
        elif match := _GLOBAL_SAVE.search(line):
            result["global_saves"].append(int(match.group(1)))
        if match := _LOAD.search(line):
            result["loads"].append(dict(path=match.group(1).strip(), iteration=int(match.group(2))))
            result["load_progress"] = []
        if match := _ITERATION.search(line):
            iteration = int(match.group(1))
            result["iterations"].append(iteration)
            if result["loads"]:
                result["load_progress"].append(iteration)
    return result


def checkpoint_files(directory: str | Path) -> list[dict]:
    """List nonempty checkpoint data files, without deserializing large weights.

    Tracker/config files alone do not prove that checkpoint data was written.
    An unknown layout is deliberately not treated as successful evidence.
    """
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        return []
    suffixes = {".pt", ".pth", ".distcp", ".bin", ".safetensors", ".ckpt"}
    metadata = {"common.pt", "train_state.pt", "latest_train_state.pt", "metadata.pt"}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.name.lower() in metadata:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue  # Retention may remove an older local checkpoint while listing.
        if size > 0:
            files.append(dict(path=str(path), size=size))
    return files


_PROCESS_HELPERS = r'''
import json, os, socket, sys, time

def proc(pid):
    base = '/proc/' + str(pid)
    try:
        with open(base + '/environ', 'rb') as f:
            env = dict(part.decode('utf-8', 'replace').split('=', 1)
                       for part in f.read().split(b'\0') if b'=' in part)
        with open(base + '/cmdline', 'rb') as f:
            cmd = [part.decode('utf-8', 'replace') for part in f.read().split(b'\0') if part]
        with open(base + '/stat') as f:
            stat = f.read().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'ppid': int(stat[1]), 'starttime': stat[19],
                'cmdline': cmd, 'env': env, 'state': stat[0]}
    except (OSError, ValueError, IndexError):
        return None

def is_agent(cmd):
    return any(os.path.basename(arg) == 'torchrun' or arg == 'torch.distributed.run'
               for arg in cmd)

def owned(run_id, pattern):
    records = []
    for value in os.listdir('/proc'):
        if not value.isdigit():
            continue
        item = proc(value)
        if not item or item['state'] == 'Z':
            continue
        env = item['env']
        if env.get('HCU_ST_RUN_ID') != run_id or pattern not in ' '.join(item['cmdline']):
            continue
        try:
            rank = int(env['RANK'])
            local_rank = int(env['LOCAL_RANK'])
            world_size = int(env['WORLD_SIZE'])
        except (ValueError, KeyError):
            continue
        parent = proc(item['ppid'])
        item.update(rank=rank, local_rank=local_rank, world_size=world_size,
                    max_restarts=env.get('TORCHELASTIC_MAX_RESTARTS'),
                    attempt_id=env.get('HCU_ST_ATTEMPT_ID', ''),
                    parent_cmdline=parent['cmdline'] if parent else [],
                    is_worker=bool(parent and is_agent(parent['cmdline'])))
        del item['env']
        records.append(item)
    return sorted(records, key=lambda value: (value['rank'], value['pid']))
'''

_INSPECT = _PROCESS_HELPERS + r'''
records = owned(sys.argv[1], sys.argv[2])
print(json.dumps({'host': socket.gethostname(), 'fqdn': socket.getfqdn(),
                  'workers': [p for p in records if p['is_worker']],
                  'owned_processes': records}))
'''

_KILL = _PROCESS_HELPERS + r'''
import signal
target = json.loads(sys.argv[3])
records = [p for p in owned(sys.argv[1], sys.argv[2])
           if p['is_worker'] and p['local_rank'] == 1]
if len(records) != 1:
    raise RuntimeError('rank 1 must uniquely identify one owned training worker')
live = records[0]
if live['pid'] != target.get('pid') or live['starttime'] != str(target.get('starttime')):
    raise RuntimeError('worker identity changed before injection')
if live['ppid'] != target.get('ppid') or live['attempt_id'] != target.get('attempt_id'):
    raise RuntimeError('worker parent or attempt changed before injection')
if live['world_size'] != int(sys.argv[4]) or live['max_restarts'] != '0':
    raise RuntimeError('expected the configured world size with inner restarts disabled')
pidfd = None
try:
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        pidfd = os.pidfd_open(live['pid'])
        latest = proc(live['pid'])
        if not latest or latest['starttime'] != live['starttime']:
            raise RuntimeError('worker vanished or PID was reused')
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
    else:
        latest = proc(live['pid'])
        if not latest or latest['starttime'] != live['starttime']:
            raise RuntimeError('worker vanished or PID was reused')
        os.kill(live['pid'], signal.SIGKILL)
finally:
    if pidfd is not None:
        os.close(pidfd)
print(json.dumps({'pid': live['pid'], 'starttime': live['starttime'],
                  'local_rank': 1, 'signal': 'SIGKILL', 'sent_epoch': time.time(),
                  'attempt_id': live['attempt_id'], 'host': socket.gethostname()}))
'''


def _container_python(remote, host, cid, target_python, code, args, timeout):
    if not re.fullmatch(r"[0-9a-f]{64}", cid):
        raise STError("test-framework", "INVALID_CONTAINER_ID", "A full Docker container ID is required")
    script = 'set -eu\nexec docker exec -u 0 "$1" "$2" -c "$3" "${@:4}"\n'
    result = remote.script(host, script, [cid, target_python, code, *args], timeout=timeout)
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise STError("environment", "INVALID_PROCESS_EVIDENCE", str(error)) from error


def inspect_workers(remote, host, cid, run_id, worker_pattern, target_python="python3", timeout=30):
    """Inspect only this run's rank processes, inside its container PID space."""
    if not run_id or not worker_pattern:
        raise STError("test-framework", "INVALID_WORKER_SELECTOR", "Run ID and worker pattern are required")
    return _container_python(remote, host, cid, target_python, _INSPECT,
                             [run_id, worker_pattern], timeout)


def kill_worker(remote, host, cid, worker, run_id, worker_pattern, target_python="python3", timeout=30, *, world_size):
    """Recheck identity and issue one SIGKILL to the owned local-rank-1 worker."""
    if (worker.get("local_rank") != 1 or not run_id or not worker_pattern
            or isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 2):
        raise STError("test-framework", "INVALID_INJECTION_TARGET", "Only owned local rank 1 may be injected")
    return _container_python(remote, host, cid, target_python, _KILL,
                             [run_id, worker_pattern, json.dumps(worker), str(world_size)], timeout)
