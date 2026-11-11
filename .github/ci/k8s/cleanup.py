# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compensating cleanup for one recorded CI run; never scan other jobs."""
import json
import os
from pathlib import Path
import shutil

from transport import cleanup_owned, kube_api

root = Path(os.environ["ST_K8S_ARTIFACTS"])
active = root / "active-crossnode.json"
if active.exists():
    evidence = Path(json.loads(active.read_text(encoding="utf-8"))["evidence"])
    receipt = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    receipt["checkpoint_cleanup_confirmed"] = False
    try:
        with kube_api() as (session, server):
            cleanup_owned(session, server, receipt, evidence)
        case = Path(receipt["case"]).resolve()
        checkpoints = case / "checkpoints"
        if checkpoints.exists():
            if (case.name != receipt["name"]
                    or (case / "RECOVERY_RUN_ID").read_text(encoding="utf-8") != receipt["name"]
                    or checkpoints.resolve().parent != case):
                raise ValueError("checkpoint directory does not belong to this recovery run")
            shutil.rmtree(checkpoints)
        if checkpoints.exists():
            raise RuntimeError("checkpoint deletion was not confirmed")
        receipt["checkpoint_cleanup_confirmed"] = True
        print(json.dumps({"checkpoints_removed": str(checkpoints)}), flush=True)
    except Exception as error:
        receipt.update(status="FAIL", cleanup_error=str(error))
        raise
    finally:
        (evidence / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
