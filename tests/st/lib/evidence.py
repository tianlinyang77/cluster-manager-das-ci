# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Small, append-only execution log and atomic ST result summary."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any


class STError(RuntimeError):
    """An observed failure, with a category kept separate from its reason code."""

    def __init__(self, category: str, code: str, detail: str):
        self.category = str(category)
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"[{self.category}/{self.code}] {self.detail}")


def _timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


_SECRET = re.compile(r"password|passwd|secret|token|private.?key|credential|authorization|api.?key", re.I)
_ENV_KEYS = {"env", "environ", "environment", "process_env", "process_environment"}
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S)


def _public(value: Any, key: str = "") -> Any:
    """Do not accidentally persist a process environment or credential mapping."""
    if key.lower() in _ENV_KEYS:
        return "<environment not recorded>"
    if _SECRET.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _public(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _PRIVATE_KEY.sub("<private key not recorded>", value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


class Evidence:
    """One case's raw ST log, explicit checks, and machine-readable result."""

    def __init__(self, root: str | Path, case_id: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.case_id = str(case_id)
        self.started_at = _timestamp()
        self.checks: list[dict[str, Any]] = []
        self.logs: dict[str, str] = {"st": str(self.root / "st.log")}
        self._lock = threading.RLock()
        (self.root / "st.log").touch(exist_ok=True)
        self.log(f"case={self.case_id} started")

    def log(self, message: object) -> None:
        """Write explicit diagnostic text; no process environment is inspected."""
        with self._lock:
            with (self.root / "st.log").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(f"{_timestamp()} {_PRIVATE_KEY.sub('<private key not recorded>', str(message))}\n")

    def check(self, name: str, passed: bool, detail: str = "", evidence: Any = None) -> None:
        """Record a check without raising; the caller controls case progression."""
        if not isinstance(passed, bool):
            raise TypeError("passed must be a bool")
        row = {"name": str(name), "passed": passed, "detail": str(detail)}
        if evidence is not None:
            row["evidence"] = _public(evidence)
        with self._lock:
            self.checks.append(row)
            self.log(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")

    def write_result(
        self,
        outcome: str,
        category: str = "",
        reason: str = "",
        extra: Any = None,
    ) -> Path:
        """Publish a complete summary atomically, including any failed checks."""
        with self._lock:
            if str(outcome).upper() == "PASS" and any(not row["passed"] for row in self.checks):
                outcome = "FAIL"
                category = category or "test-framework"
                reason = reason or "a required check failed"
                self.log("Refused PASS summary because a recorded check failed")
            result = {
                "schema_version": 1,
                "id": self.root.name,
                "case_id": self.case_id,
                "outcome": str(outcome),
                "category": str(category),
                "reason": str(reason),
                "started_at": self.started_at,
                "finished_at": _timestamp(),
                "run_dir": str(self.root),
                "checks": list(self.checks),
                "logs": dict(self.logs),
            }
            if extra is not None:
                result["extra"] = _public(extra)
            destination = self.root / "result.json"
            temporary = self.root / f".result.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return destination
