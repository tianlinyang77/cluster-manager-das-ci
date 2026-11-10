# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Read this Pod's current native ST attempt, checkpoint files and workers."""
import json
import os
from pathlib import Path
import sys

run = Path(os.environ["ST_RUN_ROOT"])
recovery_id = os.environ["ST_RECOVERY_ID"]
case = run / recovery_id
sys.path.insert(0, str(run / "source/tests/st"))
from lib import training

exec(training._PROCESS_HELPERS)
records = owned(recovery_id, "st_run_recipe.py")
current_file = case / "current-attempt.json"
current = json.loads(current_file.read_text(encoding="utf-8")) if current_file.exists() else None
files = []
saved_iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if saved_iteration:
    files = training.checkpoint_files(case / "checkpoints/global" / f"iter_{saved_iteration:07d}")
print(json.dumps({"pod_uid": os.environ["POD_UID"], "recovery_id": recovery_id, "case": str(case),
                  "current": current, "checkpoint_files": files,
                  "workers": [p for p in records if p["is_worker"]], "owned_processes": records}), flush=True)
