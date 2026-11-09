#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent phase-one ST entry. list/plan never execute commands."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib

ST_ROOT = Path(__file__).resolve().parent
CASES = {
    "ENV-001": ("test_envcheck.py", "platform / idle-check / container-status; starts dedicated training"),
    "REL-010": ("test_training_recovery.py", "torch_dist save, optional local save, one worker failure, spare replacement resumes"),
}
TIMEOUTS = ("command", "prepare", "startup", "checkpoint", "recovery", "collect", "cleanup", "total", "poll")


def load_site(path: str | Path, *, execution: bool = False) -> dict:
    """Parse a site without importing products, pytest, SSH, or GPU libraries."""
    site_path = Path(path).expanduser().resolve()
    with site_path.open("rb") as stream:
        cfg = tomllib.load(stream)
    for section in ("site", "paths", "reliability", "training", "timeouts", "ssh"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"missing TOML section [{section}]")
    nodes = cfg["site"].get("nodes", [])
    if (not isinstance(nodes, list) or not nodes
            or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in nodes)
            or len(set(nodes)) != len(nodes)):
        raise ValueError("site.nodes must contain distinct plain node names and must not be empty")
    for key in ("release_nodes", "protected_containers"):
        value = cfg["site"].setdefault(key, [])
        if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
            raise ValueError(f"site.{key} must be a list of names")
    if not set(cfg["site"]["release_nodes"]).issubset(nodes):
        raise ValueError("site.release_nodes must be a subset of site.nodes")
    platform = cfg.setdefault("platform", {})
    if not isinstance(platform, dict):
        raise ValueError("platform must be a TOML section")
    ignored = platform.setdefault("ignored_reason_codes", [])
    if (not isinstance(ignored, list)
            or any(not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", code) for code in ignored)
            or len(set(ignored)) != len(ignored)):
        raise ValueError("platform.ignored_reason_codes must contain distinct reason codes")
    cfg["paths"].setdefault("repo_root", str(ST_ROOT.parents[1]))
    cfg["ssh"].setdefault("port", 22)
    cfg["ssh"].setdefault("connect_timeout", 10)
    cfg["ssh"].setdefault("options", [])
    if (isinstance(cfg["ssh"]["port"], bool) or not isinstance(cfg["ssh"]["port"], int)
            or not 1 <= cfg["ssh"]["port"] <= 65535):
        raise ValueError("ssh.port must be an integer in 1..65535")
    if not isinstance(cfg["ssh"]["options"], list) or any(not isinstance(v, str) for v in cfg["ssh"]["options"]):
        raise ValueError("ssh.options must be an argument list")
    connect_timeout = cfg["ssh"]["connect_timeout"]
    if (isinstance(connect_timeout, bool) or not isinstance(connect_timeout, (int, float))
            or not math.isfinite(connect_timeout) or connect_timeout <= 0):
        raise ValueError("ssh.connect_timeout must be a positive finite number")
    for section in ("training", "reliability"):
        env = cfg[section].setdefault("env", {})
        if not isinstance(env, dict) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) for k in env):
            raise ValueError(f"{section}.env must contain environment variable names")
        if any(not isinstance(v, (str, int, float, bool)) for v in env.values()):
            raise ValueError(f"{section}.env values must be scalar")
    for key in TIMEOUTS:
        value = cfg["timeouts"].get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"timeouts.{key} must be a nonnegative number (0 = not calibrated)")
        cfg["timeouts"][key] = value
    validate_topology(cfg)
    cfg["_site_file"] = str(site_path)
    if execution:
        validate_execution(cfg)
    return cfg


def validate_topology(cfg: dict, *, recovery: bool = False) -> None:
    """Validate sizes before any command, resource creation, or fault injection."""
    for key in ("nodes", "gpus"):
        value = cfg["training"].get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    count = cfg["training"]["nodes"]
    if count > len(cfg["site"]["nodes"]):
        raise ValueError("training.nodes exceeds the node pool")
    if recovery and count >= len(cfg["site"]["nodes"]):
        raise ValueError("REL-010 requires at least one spare node")
    if recovery and cfg["training"]["gpus"] < 2:
        raise ValueError("REL-010 requires at least two GPUs per node for LOCAL_RANK=1")


def validate_execution(cfg: dict, *, allow_missing_bridge: bool = False) -> None:
    """Fail closed before a CaseContext can issue remote or release commands."""
    if sys.platform != "linux":
        raise ValueError("real ST runs require a Linux controller; list/plan work locally")
    if cfg["training"].get("ready") is not True:
        raise ValueError("training.ready is false: prepare and verify the training script first")
    if not isinstance(cfg["site"].get("image"), str) or not cfg["site"]["image"].strip():
        raise ValueError("site.image is required")
    for key in TIMEOUTS:
        value = cfg["timeouts"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"timeouts.{key} must be calibrated and positive before run")
    if cfg["ssh"]["port"] != 22:
        raise ValueError("hcu-envcheck platform currently requires host SSH port 22")
    if cfg["ssh"].get("options") or cfg["ssh"].get("user") or cfg["ssh"].get("config_file"):
        raise ValueError("phase one uses the existing OpenSSH configuration for all product calls; do not override it only for ST")
    if cfg["timeouts"]["poll"] >= min(cfg["timeouts"][key] for key in TIMEOUTS if key != "poll"):
        raise ValueError("poll must be shorter than all stage/command timeouts")
    for key in ("repo_root", "shared_root", "control_python", "bridge_root", "env_script"):
        if key == "bridge_root" and allow_missing_bridge:
            continue
        value = cfg["paths"].get(key, "")
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
            raise ValueError(f"paths.{key} must be an explicit absolute Linux path")
    shared = PurePosixPath(cfg["paths"]["shared_root"])
    if len(shared.parts) < 4 or str(shared) in {"/tmp", "/home", "/public", "/public/home"}:
        raise ValueError("shared_root must be a dedicated ST subdirectory")
    repo = Path(cfg["paths"]["repo_root"]).resolve()
    shared_path = Path(shared).resolve()
    if shared_path == repo or repo in shared_path.parents:
        raise ValueError("shared_root must be outside the source checkout")
    required = (repo / "hcu-envcheck/bin/hcu-cluster-run", repo / "hcu-training-reliability/cluster_manager/main.py")
    for file in required:
        if not file.is_file():
            raise ValueError(f"required product entry missing: {file}")
    script = cfg["training"].get("script", "")
    if not isinstance(script, str) or not Path(script).is_absolute() or not Path(script).is_file():
        raise ValueError("training.script must name the existing, readable train_st.sh")
    control = Path(cfg["paths"]["control_python"])
    if not control.is_file() or not os.access(control, os.X_OK):
        raise ValueError("control_python is missing or not executable")
    rel = cfg["reliability"]
    if rel.get("env_script") and not Path(rel["env_script"]).is_file():
        raise ValueError("reliability.env_script must be an existing environment-only shell file")
    train_config = rel.get("train_config", "")
    if train_config != "auto:glm53-st" and (not train_config or not Path(train_config).is_file()):
        raise ValueError("reliability.train_config must be the existing compatible training configuration")
    training = cfg["training"]
    validate_topology(cfg)
    if (training.get("synchronous") is not True or isinstance(training.get("inner_restarts"), bool)
            or training.get("inner_restarts") != 0):
        raise ValueError("phase one requires synchronous checkpoints and zero inner restarts")
    local, global_ = training.get("local_interval"), training.get("global_interval")
    if (isinstance(local, bool) or not isinstance(local, int) or local < 0
            or isinstance(global_, bool) or not isinstance(global_, int) or global_ <= 0
            or local > 0 and (global_ <= local or global_ % local)):
        raise ValueError("local_interval must be 0 (disabled) or positive; global must be positive and, when local is enabled, an integer multiple greater than local")
    iters = training.get("train_iters")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters <= global_:
        raise ValueError("train_iters must leave room after the initial checkpoint saves")
    if not training.get("worker_pattern") or not training.get("python"):
        raise ValueError("training.worker_pattern and training.python are required")
    if not allow_missing_bridge:
        from prepare_dependencies import prepared_metadata
        prepared_metadata(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list cases; no commands or connections")
    prepare = commands.add_parser("prepare", help="fetch pinned Bridge sources only; never starts ST")
    prepare.add_argument("--site", required=True)
    prepare.add_argument("--dependencies", default=str(ST_ROOT / "config/dependencies.toml"))
    prepare.add_argument("--deps-root", default=None)
    prepare.add_argument("--evidence-dir", default=None, help="new directory for dependency preparation evidence")
    for command in ("plan", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--site", required=True, help="site TOML file")
        sub.add_argument("--case", choices=("all", *CASES), default="all")
        if command == "run":
            sub.add_argument("--dependencies", help="prepare this dependency TOML before running ST")
            sub.add_argument("--deps-root", help="parent for the fresh Bridge checkout")
    args = parser.parse_args(argv)
    if args.command == "list":
        for case, (_, description) in CASES.items():
            print(f"{case}: {description}")
        return 0
    if args.command == "prepare":
        from prepare_dependencies import main as prepare_main
        forwarded = ["--site", args.site, "--dependencies", args.dependencies]
        if args.deps_root:
            forwarded += ["--deps-root", args.deps_root]
        if args.evidence_dir:
            forwarded += ["--evidence-dir", args.evidence_dir]
        return prepare_main(forwarded)
    site = str(Path(args.site).resolve())
    try:
        cfg = load_site(site)
        validate_topology(cfg, recovery=args.case in ("all", "REL-010"))
        if args.command == "run":
            if args.deps_root and not args.dependencies:
                raise ValueError("--deps-root requires --dependencies")
            if args.dependencies:
                # Validate all other real-run requirements before fetching anything.
                validate_execution(cfg, allow_missing_bridge=True)
                from prepare_dependencies import prepare, DependencyError
                try:
                    prepared = prepare(site, args.dependencies, args.deps_root)
                except DependencyError as error:
                    raise ValueError(str(error)) from error
                site = prepared["site"]
                print(json.dumps({"dependency_preparation": prepared}, ensure_ascii=False))
                cfg = load_site(site)
            validate_execution(cfg)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return 2
    selected = list(CASES) if args.case == "all" else [args.case]
    if args.command == "plan":
        print(json.dumps({"mode": "local plan only; no SSH, Docker, release, or training",
                          "cases": selected, "nodes": cfg["site"]["nodes"], "image": cfg["site"].get("image", ""),
                          "training_ready": cfg["training"].get("ready", False),
                          "training_nodes": cfg["training"]["nodes"], "gpus_per_node": cfg["training"]["gpus"],
                          "world_size": cfg["training"]["nodes"] * cfg["training"]["gpus"],
                          "spare_count": len(cfg["site"]["nodes"]) - cfg["training"]["nodes"],
                          "release_nodes": cfg["site"]["release_nodes"], "paths": cfg["paths"],
                          "timeouts": cfg["timeouts"], "reliability_environment_keys": sorted(cfg["reliability"]["env"]),
                          "note": "Each case owns fresh resources. ENV-001 also starts training to test BUSY."},
                         ensure_ascii=False, indent=2))
        return 0
    try:
        import importlib.util
        if importlib.util.find_spec("pytest") is None:
            raise ValueError("pytest is required in the ST controller environment")
        files = [str(ST_ROOT / "testcases" / CASES[case][0]) for case in selected]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        return subprocess.call([sys.executable, "-B", "-m", "pytest", "-c", str(ST_ROOT / "pytest.ini"),
                                "--rootdir", str(ST_ROOT), "--site", site, *files], env=env)
    except (OSError, ValueError) as error:
        print(f"RUN ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
