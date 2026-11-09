#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Based on Bridge 4430df5388a84b0ae88c8633ebc8fe5488b8385c GLM53 example.
# $1: reliability's current allocation hostfile. No HF assets or pretrained weights.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MEGATRON_BRIDGE_PATH:?Bridge source root is required}"
: "${CKPT_DIR:?Shared torch_dist checkpoint root is required}"
: "${LOCAL_CKPT_DIR:?Shared local checkpoint root is required}"
: "${LOG_DIR:?Training log directory is required}"
HOSTFILE=${1:?Current allocation hostfile is required}
[[ -f "$HOSTFILE" ]] || { echo "ERROR: hostfile not found: $HOSTFILE" >&2; exit 64; }
HOSTFILE=$(cd -- "$(dirname -- "$HOSTFILE")" && pwd)/$(basename -- "$HOSTFILE")
MODE=${MODE:-train}
[[ "$MODE" == train || "$MODE" == dryrun ]] || { echo 'ERROR: MODE must be train or dryrun' >&2; exit 64; }
[[ -f "$MEGATRON_BRIDGE_PATH/run_recipe.py" && -f "$SCRIPT_DIR/st_run_recipe.py" && -f "$SCRIPT_DIR/st_glm53_recipe.py" ]] || {
    echo 'ERROR: Bridge runner or ST recipe payload is missing' >&2; exit 66;
}
GPUS=${GPUS:-4}
AVAILABLE_NODES=$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$HOSTFILE")
NNODES=${NNODES:-$AVAILABLE_NODES}
for value in "$GPUS" "$NNODES"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: GPUS and NNODES must be positive integers' >&2; exit 64; }
done
(( NNODES <= AVAILABLE_NODES )) || { echo 'ERROR: allocation contains too few nodes' >&2; exit 64; }
MASTER_ADDR=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$HOSTFILE")
MASTER_PORT=${MASTER_PORT:-29501}
PORT=${PORT:-36000}
for value in "$PORT" "$MASTER_PORT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( value <= 65535 )) || { echo 'ERROR: invalid SSH/rendezvous port' >&2; exit 64; }
done
TRAIN_ENV_SCRIPT=${TRAIN_ENV_SCRIPT:-/opt/dtk/env.sh}
[[ -r "$TRAIN_ENV_SCRIPT" ]] || { echo "ERROR: environment script not readable: $TRAIN_ENV_SCRIPT" >&2; exit 66; }
NUM_LAYERS=${NUM_LAYERS:-2}
SEQ_LENGTH=${SEQ_LENGTH:-128}
TRAIN_ITERS=${TRAIN_ITERS:-2000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
TP=${TP:-1}; PP=${PP:-1}; CP=${CP:-1}; EP=${EP:-1}; ETP=${ETP:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
LOCAL_SAVE_INTERVAL=${LOCAL_SAVE_INTERVAL:-5}
export MEGATRON_BRIDGE_PATH CKPT_DIR LOCAL_CKPT_DIR LOG_DIR GPUS NNODES MASTER_ADDR MASTER_PORT
export NUM_LAYERS SEQ_LENGTH TRAIN_ITERS MICRO_BATCH_SIZE TP PP CP EP ETP SAVE_INTERVAL LOCAL_SAVE_INTERVAL
export PYTHONDONTWRITEBYTECODE=1
mkdir -p -- "$CKPT_DIR" "$LOCAL_CKPT_DIR" "$LOG_DIR"
printf 'ST GLM53: %s nodes x %s GPUs, layers=%s hidden=512 seq=%s, mock/NullTokenizer\n' "$NNODES" "$GPUS" "$NUM_LAYERS" "$SEQ_LENGTH"
printf 'Checkpoints: local=%s every %s; torch_dist=%s every %s; synchronous; inner retries=0\n' "$LOCAL_CKPT_DIR" "$LOCAL_SAVE_INTERVAL" "$CKPT_DIR" "$SAVE_INTERVAL"

# Do not source Bridge env/env.sh: it depends on $0 and writes the recipe registry.
# The Python bootstrap supplies source paths and registers this recipe in memory.
RUNNER_ARGS=("$SCRIPT_DIR/st_run_recipe.py" --recipe glm53_st_pretrain_config --mode pretrain
             --dataset mock --seq_length "$SEQ_LENGTH" dataset.num_workers=0)
if [[ "$MODE" == dryrun ]]; then
    source "$TRAIN_ENV_SCRIPT"
    python3 "${RUNNER_ARGS[@]}" --dry-run
    exit $?
fi
TORCHRUN_ARGS=(--nnodes="$NNODES" --nproc_per_node="$GPUS" --max-restarts=0
               --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
NODE_COMMAND='set -euo pipefail; source "$1"; shift; exec torchrun --node_rank="${OMPI_COMM_WORLD_RANK:-0}" "$@"'
if (( NNODES == 1 )); then
    bash -c "$NODE_COMMAND" st-node "$TRAIN_ENV_SCRIPT" "${TORCHRUN_ARGS[@]}" "${RUNNER_ARGS[@]}" 2>&1 | tee "$LOG_DIR/${LOG_NAME:-train.log}"
else
    HOSTFILE_SLOTS="$LOG_DIR/hostfile_slots"
    awk 'NF && $1 !~ /^#/ {print $1 " slots=1"; n++; if(n==limit) exit}' limit="$NNODES" "$HOSTFILE" > "$HOSTFILE_SLOTS"
    env_args=(-x PATH)
    forward=(LD_LIBRARY_PATH PYTHONPATH MEGATRON_BRIDGE_PATH CKPT_DIR LOCAL_CKPT_DIR LOG_DIR GPUS NNODES
             MASTER_ADDR MASTER_PORT NUM_LAYERS SEQ_LENGTH TRAIN_ITERS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE
             TP PP CP EP ETP SAVE_INTERVAL LOCAL_SAVE_INTERVAL HCU_ST_RUN_ID HCU_ST_ATTEMPT_ID
             OMPI_MCA_mca_base_env_list PRTE_MCA_mca_base_env_list PYTHONDONTWRITEBYTECODE)
    for name in "${forward[@]}"; do
        [[ -v "$name" ]] && env_args+=(-x "$name")
    done
    printf -v remote_command '%q ' bash -c "$NODE_COMMAND" st-node "$TRAIN_ENV_SCRIPT" "${TORCHRUN_ARGS[@]}" "${RUNNER_ARGS[@]}"
    mpirun -np "$NNODES" --hostfile "$HOSTFILE_SLOTS" --allow-run-as-root --bind-to none \
        --mca plm_rsh_no_tree_spawn 1 --mca plm_rsh_args "-p $PORT" "${env_args[@]}" \
        bash -c "$remote_command" 2>&1 | tee "$LOG_DIR/${LOG_NAME:-train.log}"
fi
