# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Confirm completed recovery and durable checkpoints after training Pod deletion."""
import json
from datetime import datetime
import os
from pathlib import Path
import sys

scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts.parents[2] / "tests/st"))
from lib.training import checkpoint_files, parse_training_log

root = Path(os.environ["ST_K8S_ARTIFACTS"])
evidence = Path(json.loads((root / "active-crossnode.json").read_text(encoding="utf-8"))["evidence"])
receipt = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
assert receipt["status"] == "PASS" and receipt["cleanup_confirmed"] and receipt["node_taint_removed"]
case = Path(receipt["case"])
assert (case / "RECOVERY_RUN_ID").read_text(encoding="utf-8") == receipt["name"]
launches = [json.loads(p.read_text(encoding="utf-8")) for p in case.glob("launch-*.json")]
assert len(launches) == 2 and {x["node"] for x in launches} == {receipt["primary_node"], receipt["standby_node"]}
log = (evidence / ("pod-" + receipt["recovered_pod_uid"] + ".log")).read_text(encoding="utf-8")
parsed = parse_training_log(log)
assert max(parsed["iterations"]) == 1000 and 1000 in parsed["global_saves"]
assert all(c["state"]["terminated"]["exitCode"] == 0 for p in receipt["completion_pods"] for c in p["status"]["containerStatuses"])
checkpoints = []
for directory in sorted((case / "checkpoints/global").glob("iter_*")):
    files = checkpoint_files(directory)
    assert files
    checkpoints.append(dict(directory=str(directory), data_files=files, data_bytes=sum(f["size"] for f in files)))
assert checkpoints
first_progress = next(line.split(" ", 1)[0] for line in log.splitlines() if "elapsed time per iteration" in line)
recovery_seconds = datetime.fromisoformat(first_progress.replace("Z", "+00:00")).timestamp() - receipt["injection"]["sent_epoch"]
summary = dict(status="PASS", initial_node=receipt["primary_node"], recovered_node=receipt["standby_node"],
               loaded_iteration=parsed["loads"][-1]["iteration"], completed_iteration=max(parsed["iterations"]),
               pool=receipt["pool"], world_size=receipt["world_size"], checkpoints=checkpoints,
               storage_visible_after_training_pods_deleted=True, source_commit=receipt["source_commit"],
               controller_commit=receipt["controller_commit"], cleanup_confirmed=True,
               fault_to_first_resumed_iteration_seconds=round(recovery_seconds, 3))
(evidence / "validation-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
