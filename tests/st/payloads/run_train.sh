#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# One invocation = one training attempt. Container lifecycle/recovery belongs to reliability.
set -euo pipefail
umask 022

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for context in "$SCRIPT_DIR/context.env" "$SCRIPT_DIR/../context.env"; do
    if [[ -f "$context" ]]; then
        source "$context"
        break
    fi
done
: "${HCU_ST_CASE_ROOT:?ST case root is required}"
: "${HCU_ST_RUN_ID:?ST run ID is required}"
: "${HCU_ST_TRAIN_SCRIPT:?Frozen training script is required}"
HCU_ST_PYTHON=${HCU_ST_PYTHON:-python3}
readonly HCU_ST_CASE_ROOT HCU_ST_RUN_ID HCU_ST_TRAIN_SCRIPT
export HCU_ST_CASE_ROOT HCU_ST_RUN_ID HCU_ST_TRAIN_SCRIPT HCU_ST_PYTHON

mkdir -p -- "$HCU_ST_CASE_ROOT/attempts" "$HCU_ST_CASE_ROOT/monitor"
HCU_ST_ATTEMPT_ID="$(date +%Y%m%dT%H%M%S)-$$-${RANDOM}${RANDOM}"
HCU_ST_ATTEMPT_DIR="$HCU_ST_CASE_ROOT/attempts/$HCU_ST_ATTEMPT_ID"
mkdir -- "$HCU_ST_ATTEMPT_DIR"
export HCU_ST_ATTEMPT_ID HCU_ST_ATTEMPT_DIR
ATTEMPT_LOG="$HCU_ST_ATTEMPT_DIR/train.log"
: > "$ATTEMPT_LOG"

write_metadata() {
    "$HCU_ST_PYTHON" - "$1" "${2:-}" <<'PY'
import datetime, json, os, socket, sys, time
from pathlib import Path
directory = Path(os.environ['HCU_ST_ATTEMPT_DIR'])
path = directory / 'attempt.json'
new = not path.exists()
data = json.loads(path.read_text(encoding='utf-8')) if not new else {
    'attempt_id': os.environ['HCU_ST_ATTEMPT_ID'], 'run_id': os.environ['HCU_ST_RUN_ID'],
    'host': socket.gethostname(), 'fqdn': socket.getfqdn(),
    'started_epoch': time.time(), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'script': os.environ['HCU_ST_TRAIN_SCRIPT'], 'log': str(directory / 'train.log'),
    'slots_file': os.environ.get('HCU_ST_SLOTS_FILE', ''), 'slots': [], 'exit_code': None,
}
slots = data['slots_file']
if new and slots and Path(slots).is_file():
    data['slots'] = Path(slots).read_text(encoding='utf-8').splitlines()
data['status'] = sys.argv[1]
if os.environ.get('HCU_ST_MONITOR_LOG'):
    data['monitor_log'] = os.environ['HCU_ST_MONITOR_LOG']
if sys.argv[2]:
    data.update(exit_code=int(sys.argv[2]), ended_epoch=time.time(),
                ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
temporary.replace(path)
PY
}

run_attempt() {
    set -e
    [[ $# == 1 && -f "$1" ]] || { echo 'ST bootstrap: one readable slots hostfile is required' >&2; return 64; }
    HCU_ST_SLOTS_FILE=$(readlink -f -- "$1")
    export HCU_ST_SLOTS_FILE
    write_metadata STARTING
    [[ -f "$HCU_ST_CASE_ROOT/training.env" ]] || { echo 'ST bootstrap: training.env is missing' >&2; return 66; }
    source "$HCU_ST_CASE_ROOT/training.env"
    command -v flock >/dev/null || { echo 'ST bootstrap: flock is required' >&2; return 127; }
    exec 9> "$HCU_ST_CASE_ROOT/attempt.lock"
    flock -n 9 || { echo 'ST bootstrap: another training attempt is still active' >&2; return 75; }
    [[ -f "$HCU_ST_TRAIN_SCRIPT" ]] || { echo 'ST bootstrap: training script is missing' >&2; return 66; }
    : "${MEGATRON_BRIDGE_PATH:?Explicit Bridge root is required}"

    HCU_ST_MONITOR_LOG="$HCU_ST_CASE_ROOT/monitor/log-$(date +%F-%H%M)-$HCU_ST_ATTEMPT_ID.log"
    export HCU_ST_MONITOR_LOG
    "$HCU_ST_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['HCU_ST_CASE_ROOT'])
monitor = root / 'monitor'
current = root / 'current-attempt.json'
if current.exists():
    previous = json.loads(current.read_text(encoding='utf-8'))
    if previous.get('run_id') != os.environ['HCU_ST_RUN_ID']:
        raise RuntimeError('current-attempt marker belongs to another run')
    old = Path(previous['monitor_log'])
    source = Path(previous['log'])
    if old.parent.resolve() != monitor.resolve() or not source.resolve().is_relative_to((root / 'attempts').resolve()):
        raise RuntimeError('refusing to rotate an unowned training log')
    if old.exists():
        if old.is_symlink() or not source.is_file() or not old.samefile(source):
            raise RuntimeError('registered monitor log identity changed')
        old.unlink()  # The original inode remains at attempts/<id>/train.log.
if list(monitor.glob('log*.log')):
    raise RuntimeError('unregistered logs exist in this private monitor directory')
log = Path(os.environ['HCU_ST_ATTEMPT_DIR']) / 'train.log'
active = Path(os.environ['HCU_ST_MONITOR_LOG'])
os.link(log, active)  # Same case filesystem; failure is a real bootstrap error.
data = dict(run_id=os.environ['HCU_ST_RUN_ID'], attempt_id=os.environ['HCU_ST_ATTEMPT_ID'],
            log=str(log), monitor_log=str(active))
temporary = current.with_suffix('.json.tmp')
temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
temporary.replace(current)
PY
    export CKPT_DIR="$HCU_ST_CASE_ROOT/checkpoints/global"
    export LOCAL_CKPT_DIR="$HCU_ST_CASE_ROOT/checkpoints/local"
    # The existing train_st.sh has its own tee. Keep its file away from our sole monitor writer.
    export LOG_DIR="$HCU_ST_ATTEMPT_DIR/script_logs"
    export LOG_NAME="train.log"
    mkdir -p -- "$CKPT_DIR" "$LOCAL_CKPT_DIR" "$LOG_DIR"
    # Inner mpirun must pass the same dynamic attempt identity to every node.
    # Retain the site's existing fabric/environment entries; append ST identity.
    export OMPI_MCA_mca_base_env_list="${OMPI_MCA_mca_base_env_list:+${OMPI_MCA_mca_base_env_list};}HCU_ST_RUN_ID=$HCU_ST_RUN_ID;HCU_ST_ATTEMPT_ID=$HCU_ST_ATTEMPT_ID"
    export PRTE_MCA_mca_base_env_list="${PRTE_MCA_mca_base_env_list:+${PRTE_MCA_mca_base_env_list};}HCU_ST_RUN_ID=$HCU_ST_RUN_ID;HCU_ST_ATTEMPT_ID=$HCU_ST_ATTEMPT_ID"
    # Bridge compiles on LOCAL_RANK=0 of each host. Its source is shared here,
    # so finish one build before starting any distributed workers.
    if [[ ${HCU_ST_PREPARE_DATASET_HELPERS:-0} == 1 ]]; then
        (
            source "${TRAIN_ENV_SCRIPT:?Training environment script is required}" || exit $?
            dataset_helpers="$MEGATRON_BRIDGE_PATH/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets"
            exec 8> "$dataset_helpers/.hcu-st-build.lock" || exit $?
            flock 8 || exit $?
            printf '[ST-DEPENDENCY] building dataset helpers before distributed launch\n'
            make -C "$dataset_helpers" || exit $?
            printf '[ST-DEPENDENCY] dataset helpers ready\n'
        ) || return $?
    fi
    write_metadata RUNNING
    printf '[ST-ATTEMPT] id=%s host=%s slots=%s\n' "$HCU_ST_ATTEMPT_ID" "$(hostname)" "$HCU_ST_SLOTS_FILE"
    bash "$HCU_ST_TRAIN_SCRIPT" "$HCU_ST_SLOTS_FILE"
}

# Capture bootstrap errors and the unmodified training stdout/stderr. Preserve the real pipeline code.
set +e
run_attempt "$@" 2>&1 | tee "$ATTEMPT_LOG"
pipeline_status=("${PIPESTATUS[@]}")
code=${pipeline_status[0]}
if (( pipeline_status[1] != 0 )); then
    code=74
fi
set -e
write_metadata EXITED "$code" || { echo 'ST bootstrap: could not finalize attempt metadata' >&2; exit 74; }
exit "$code"
