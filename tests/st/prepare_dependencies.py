#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned Bridge sources. Never install packages or execute fetched code."""
from __future__ import annotations

import argparse
import copy
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tomllib

ST_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ST_ROOT / "config/dependencies.toml"
REQUIRED_FILES = (
    "run_recipe.py", "env/env.sh",
    "3rdparty/Megatron-Bridge/src/megatron/bridge/__init__.py",
    "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/__init__.py",
)


class DependencyError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dependencies(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    bridge = config.get("bridge")
    if not isinstance(bridge, dict):
        raise ValueError("dependencies must contain a [bridge] table")
    url, commit = bridge.get("url"), bridge.get("commit")
    if (not isinstance(url, str) or not url or url.startswith("-")
            or any(c.isspace() or ord(c) < 32 for c in url)):
        raise ValueError("bridge.url must be a repository URL or an absolute local repository path without whitespace")
    if not (url.startswith(("https://", "ssh://", "file://", "git@")) or Path(url).is_absolute()):
        raise ValueError("bridge.url must use HTTPS, SSH, file://, or an absolute local repository path")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("bridge.commit must be an explicit 40-character commit SHA; branches, tags and empty values are not accepted")
    timeout = bridge.get("command_timeout", 600)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("bridge.command_timeout must be a positive finite number")
    mcore = bridge.get("mcore", "gitlink")
    if mcore not in ("gitlink", "dev"):
        raise ValueError("bridge.mcore must be gitlink or dev")
    return dict(url=url, commit=commit.lower(), command_timeout=timeout, mcore=mcore)


def _toml_value(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"unsupported site TOML value: {type(value).__name__}")


def dump_toml(config):
    """Write site values without an extra controller dependency; comments stay in the original."""
    lines = []

    def table(values, names):
        if names:
            lines.append("[" + ".".join(json.dumps(n) for n in names) + "]")
        for key, value in values.items():
            if not isinstance(value, dict):
                lines.append(json.dumps(key) + " = " + _toml_value(value))
        lines.append("")
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, [*names, key])

    table(config, [])
    text = "\n".join(lines)
    tomllib.loads(text)  # Validate before writing a generated site.
    return text


def _write(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    Path(path).chmod(0o600)


def prepared_metadata(config):
    """Read the exact preparation record associated with a generated site."""
    dependency = config.get("dependencies", {})
    if "bridge_manifest" not in dependency:
        return None
    path = Path(dependency["bridge_manifest"])
    if not path.is_absolute() or sha256(path) != dependency.get("bridge_manifest_sha256"):
        raise ValueError("prepared Bridge manifest is missing or its SHA256 changed")
    data = json.loads(path.read_text(encoding="utf-8"))
    bridge = data.get("bridge", {})
    if (data.get("status") != "READY" or bridge.get("commit") != bridge.get("requested_commit")
            or not re.fullmatch(r"[0-9a-f]{40}", bridge.get("commit", ""))
            or Path(bridge.get("root", "")).resolve() != Path(config["paths"]["bridge_root"]).resolve()
            or not Path(bridge.get("root", "")).is_dir()):
        raise ValueError("prepared Bridge identity does not match paths.bridge_root")
    return data


def prepare(site, dependencies=DEFAULT_CONFIG, deps_root=None, *, evidence_dir=None):
    from run_st import load_site
    site = Path(site).expanduser().resolve()
    dependencies = Path(dependencies).expanduser().resolve()
    spec = load_dependencies(dependencies)  # Fail before creating any directory or command.
    load_site(site)
    with site.open("rb") as stream:
        original = tomllib.load(stream)
    dump_toml(original)
    root = Path(deps_root).expanduser().resolve() if deps_root is not None else ST_ROOT / ".deps"
    root.mkdir(parents=True, exist_ok=True)
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(6))
    run.mkdir(mode=0o700, exist_ok=False)
    checkout = run / "Megatron-bridge-das"
    evidence = Path(evidence_dir).expanduser().resolve() if evidence_dir is not None else run
    if evidence != run:
        evidence.mkdir(parents=True, mode=0o700, exist_ok=False)
    logfile, manifest, output = evidence / "prepare.log", evidence / "manifest.json", evidence / "site.prepared.toml"
    record = dict(schema_version=1, status="PREPARING", source_only=True,
                  site_source=str(site), site_sha256=sha256(site), dependencies_source=str(dependencies),
                  dependencies_sha256=sha256(dependencies), log=str(logfile), warnings=[],
                  bridge=dict(url=spec["url"], requested_commit=spec["commit"], root=str(checkout)))
    _write(logfile, "Bridge source preparation; no package installation or training.\n")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")

    def git(*args, cwd=None):
        # Process-local setting only; nested submodule paths exceed Windows MAX_PATH.
        options = ["-c", "core.longpaths=true"] if sys.platform == "win32" else []
        command = ["git", *options, *map(str, args)]
        with logfile.open("a", encoding="utf-8") as log:
            log.write(json.dumps(command, ensure_ascii=False) + "\n")
            log.flush()
            try:
                result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=spec["command_timeout"])
            except (OSError, subprocess.TimeoutExpired) as error:
                log.write(str(error) + "\n")
                raise DependencyError(f"Git did not complete; see {logfile}") from error
            log.write(result.stdout + result.stderr + "\n")
        if result.returncode:
            raise DependencyError(f"Git exited {result.returncode}; see {logfile}")
        return result.stdout.rstrip("\r\n")

    try:
        record["git_version"] = git("--version")
        git("init", checkout)
        git("remote", "add", "origin", spec["url"], cwd=checkout)
        git("fetch", "--no-tags", "--depth=1", "origin", spec["commit"], cwd=checkout)
        git("checkout", "--detach", "FETCH_HEAD", cwd=checkout)
        actual = git("rev-parse", "HEAD", cwd=checkout)
        if actual != spec["commit"]:
            raise DependencyError("fetched Bridge HEAD does not equal the configured commit")
        git("submodule", "update", "--init", "--recursive", cwd=checkout)
        # Keep leading status characters: '+'/'-'/'U' are not a pinned, initialized checkout.
        statuses = git("submodule", "status", "--recursive", cwd=checkout)
        submodules = []
        for line in statuses.splitlines():
            row = line
            if not re.match(r"^ [0-9a-f]{40} ", row):
                raise DependencyError("Bridge submodule is uninitialized, conflicted, or at a different commit")
            submodules.append(dict(commit=row[1:41], path=row[42:].split(" (", 1)[0]))
        missing = [p for p in REQUIRED_FILES if not (checkout / p).is_file()]
        if missing:
            raise DependencyError("incomplete Bridge source tree: " + ", ".join(missing))
        record["bridge"].update(commit=actual, submodules=submodules)
        if spec["mcore"] == "dev":
            upstream = checkout / "3rdparty/Megatron-Bridge"
            dev = (upstream / ".dev.commit").read_text(encoding="utf-8").strip()
            if not re.fullmatch(r"[0-9a-f]{40}", dev):
                raise DependencyError("upstream .dev.commit must contain a full pinned commit")
            mcore = upstream / "3rdparty/Megatron-LM"
            before = git("rev-parse", "HEAD", cwd=mcore)
            git("fetch", "--no-tags", "--depth=1", "origin", dev, cwd=mcore)
            git("checkout", "--detach", dev, cwd=mcore)
            if git("rev-parse", "HEAD", cwd=mcore) != dev:
                raise DependencyError("MCore checkout does not equal upstream .dev.commit")
            git("submodule", "update", "--init", "--recursive", cwd=mcore)
            submodules = []
            for row in git("submodule", "status", "--recursive", cwd=checkout).splitlines():
                path = row[42:].split(" (", 1)[0]
                expected_override = path == "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM"
                if not re.match(r"^[ +][0-9a-f]{40} ", row) or row[0] == "+" and not expected_override:
                    raise DependencyError("unexpected submodule state after selecting pinned MCore dev")
                item = dict(path=path, commit=row[1:41])
                if expected_override:
                    item["gitlink_commit"] = before
                submodules.append(item)
            record["bridge"]["submodules"] = submodules
            record["bridge"]["mcore"] = dict(mode="dev", gitlink_commit=before, commit=dev)
        environment = (checkout / "env/env.sh").read_text(encoding="utf-8", errors="replace")
        if re.search(r"dirname[^\n]*\$0", environment):
            record["warnings"].append("Bridge env/env.sh derives paths from $0; built-in ST bypasses it, custom launchers must verify compatibility")
        if re.search(r"^\s*cp\s", environment, re.MULTILINE):
            record["warnings"].append("Bridge env/env.sh writes registration files; built-in ST registers in memory, custom launchers need an isolated writable checkout")
        record["status"] = "READY"
        _write(manifest, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        generated = copy.deepcopy(original)
        generated["paths"]["bridge_root"] = str(checkout)
        generated.setdefault("dependencies", {}).update(bridge_manifest=str(manifest), bridge_manifest_sha256=sha256(manifest))
        _write(output, dump_toml(generated))
        return dict(status="READY", source_only=True, bridge_root=str(checkout),
                    site=str(output), manifest=str(manifest), warnings=record["warnings"])
    except Exception as error:
        record.update(status="FAILED", error=str(error))
        _write(manifest, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        raise DependencyError(f"dependency preparation failed: {error}; evidence={evidence}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--dependencies", default=str(DEFAULT_CONFIG))
    parser.add_argument("--deps-root", help="isolated dependency parent; default tests/st/.deps")
    parser.add_argument("--evidence-dir", help="new directory for preparation logs, manifest and generated site")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.site, args.dependencies, args.deps_root, evidence_dir=args.evidence_dir)
    except (OSError, ValueError, DependencyError) as error:
        print(f"PREPARE ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
