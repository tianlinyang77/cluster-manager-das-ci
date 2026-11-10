# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compensating cleanup for one recorded CI run; never scan other jobs."""
import json
import os
from pathlib import Path

from transport import cleanup_owned, kube_api

root = Path(os.environ["ST_K8S_ARTIFACTS"])
active = root / "active-crossnode.json"
if active.exists():
    evidence = Path(json.loads(active.read_text(encoding="utf-8"))["evidence"])
    receipt = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    try:
        with kube_api() as (session, server):
            cleanup_owned(session, server, receipt, evidence)
    finally:
        (evidence / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
