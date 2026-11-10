# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Reuse native ST payloads across one Volcano restart in an isolated case."""
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

run = Path(os.environ["ST_RUN_ROOT"])
assert (run / "storage-marker").read_text(encoding="utf-8") == run.name
recovery_id = os.environ["ST_RECOVERY_ID"]
case = run / recovery_id
prepared = json.loads((run / "prepare-result.json").read_text(encoding="utf-8"))
st = run / "source/tests/st"
sys.path.insert(0, str(st))
from prepare_dependencies import prepared_metadata
from run_st import load_site
from payloads.st_glm53_recipe import settings

manifest = prepared_metadata(load_site(prepared["site"]))
payload = case / "payloads"
values = {
    "NNODES": "1", "GPUS": os.environ["ST_GPUS"], "TRAIN_ITERS": "1000", "NUM_LAYERS": "2", "SEQ_LENGTH": "128",
    "MICRO_BATCH_SIZE": "1", "TP": "1", "PP": "1", "CP": "1", "EP": "1", "ETP": "1",
    "SAVE_INTERVAL": "50", "LOCAL_SAVE_INTERVAL": "0", "MEGATRON_BRIDGE_PATH": prepared["bridge_root"],
    "TRAIN_ENV_SCRIPT": str(payload / "container.env.sh"),
}
values.update({k: str(v) for k, v in settings(values).items()})
if not case.exists():
    case.mkdir()
    payload.mkdir()
    (case / "RECOVERY_RUN_ID").write_text(recovery_id)
    for name in ("run_train.sh", "train_st.sh", "st_run_recipe.py", "st_glm53_recipe.py"):
        shutil.copyfile(st / "payloads" / name, payload / name)
    shutil.copyfile(st / "config/container.ci.env.sh", payload / "container.env.sh")
    (case / "training.env").write_text("\n".join("export " + k + "=" + shlex.quote(v) for k, v in values.items()) + "\n")
    (case / "hostfile").write_text("127.0.0.1\n")
assert (case / "RECOVERY_RUN_ID").read_text(encoding="utf-8") == recovery_id
receipt = {"recovery_id": recovery_id, "pod": os.environ["POD_NAME"], "pod_uid": os.environ["POD_UID"],
           "node": os.environ["NODE_NAME"], "bridge": manifest["bridge"], "case": str(case),
           "attempt_id": os.environ["POD_UID"], "log_source": "Kubernetes stdout", "workload": values}
(case / ("launch-" + os.environ["POD_UID"] + ".json")).write_text(json.dumps(receipt, indent=2))
attempt = case / "attempts" / os.environ["POD_UID"]
attempt.mkdir(parents=True)
(case / "checkpoints/global").mkdir(parents=True, exist_ok=True)
(case / "checkpoints/local").mkdir(parents=True, exist_ok=True)
(case / "current-attempt.json").write_text(json.dumps(receipt, indent=2))
os.environ.update(values)
os.environ.update(HCU_ST_CASE_ROOT=str(case), HCU_ST_RUN_ID=recovery_id,
                  HCU_ST_TRAIN_SCRIPT=str(payload / "train_st.sh"), HCU_ST_PYTHON=sys.executable,
                  PYTHONDONTWRITEBYTECODE="1", HCU_ST_ATTEMPT_ID=os.environ["POD_UID"],
                  CKPT_DIR=str(case / "checkpoints/global"), LOCAL_CKPT_DIR=str(case / "checkpoints/local"),
                  LOG_DIR=str(attempt))
os.environ["PYTHONPATH"] = str(run / "python-deps") + os.pathsep + os.environ.get("PYTHONPATH", "")
print("K8S_RECOVERY_LAUNCH=" + json.dumps(receipt), flush=True)
dataset = Path(prepared["bridge_root"]) / "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets"
# No tee pipeline: orphan checkpoint children must not hold PID 1 alive after torchrun exits.
command = 'set -e; source "$1"; shift; make -C "$1"; shift; exec torchrun "$@"'
os.execvp("bash", ["bash", "-c", command, "st-k8s", values["TRAIN_ENV_SCRIPT"], str(dataset),
                    "--nnodes=1", "--nproc_per_node=" + values["GPUS"], "--max-restarts=0", "--node_rank=0",
                    "--master_addr=127.0.0.1", "--master_port=29501", str(payload / "st_run_recipe.py"),
                    "--recipe", "glm53_st_pretrain_config", "--mode", "pretrain", "--dataset", "mock",
                    "--seq_length", values["SEQ_LENGTH"], "dataset.num_workers=0"])
