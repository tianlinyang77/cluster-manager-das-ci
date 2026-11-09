#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CI hostfile adapter and owned-resource lifecycle; imports never contact nodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib

from prepare_dependencies import dump_toml
from run_st import load_site, validate_topology

ST = Path(__file__).resolve().parent
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def host_nodes(path):
    nodes = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not NODE.fullmatch(line) or line in nodes:
            raise ValueError(f"hostfile line {number}: expected one distinct hostname/IPv4, no slots or expressions")
        nodes.append(line)
    if not nodes:
        raise ValueError("ST_HOSTFILE contains no allocated nodes")
    return nodes


def environment_nodes(value):
    nodes=re.split(r"[,\s]+", value.strip())
    if not nodes or any(not NODE.fullmatch(node) for node in nodes) or len(set(nodes)) != len(nodes):
        raise ValueError("ST_NODES must contain distinct plain hostnames/IPv4, separated by commas or whitespace")
    return nodes


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def render(template, hostfile, control_python, shared_root, run_key, repo_root=None, *, nodes=None):
    """Generate one immutable run/site from explicit inputs; no SSH or training."""
    if not control_python or not shared_root or not (hostfile or nodes):
        raise ValueError("ST_CONTROL_PYTHON, ST_SHARED_ROOT and either ST_HOSTFILE or ST_NODES are required")
    if hostfile and nodes:
        raise ValueError("use either ST_HOSTFILE or ST_NODES, not both")
    if not all(Path(value).is_absolute() for value in (control_python, shared_root, *([hostfile] if hostfile else []))):
        raise ValueError("CI Python, shared directory and hostfile must be absolute paths")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", run_key):
        raise ValueError("run key must be a plain run-id/attempt, without path traversal")
    shared = Path(shared_root).expanduser().resolve()
    repo = Path(repo_root or ST.parents[1]).resolve()
    if len(shared.parts) < 3 or shared == repo or shared.is_relative_to(repo):
        raise ValueError("shared root must be a dedicated directory outside the checkout")
    source="ST_HOSTFILE" if hostfile else "ST_NODES"
    source_bytes=Path(hostfile).read_bytes() if hostfile else nodes.encode("utf-8")
    nodes = host_nodes(hostfile) if hostfile else environment_nodes(nodes)
    cfg = tomllib.loads(Path(template).read_text(encoding="utf-8"))
    cfg["site"]["nodes"] = nodes
    if cfg["site"].get("release_nodes"):
        raise ValueError("CI must not release existing workloads; release_nodes must remain empty")
    validate_topology(cfg, recovery=True)
    nhc = cfg.get("ci", {}).get("nhc_root", "")
    if not nhc.startswith("/") or ".." in Path(nhc).parts:
        raise ValueError("ci.nhc_root must be an absolute node-local dependency path")
    run = shared / "ci-runs" / run_key
    replacements = {"__REPO_ROOT__": str(repo), "__SHARED_ROOT__": str(shared),
                    "__RUN_ROOT__": str(run), "__CONTROL_PYTHON__": os.path.abspath(control_python),
                    "__NHC_ROOT__": nhc}

    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):
            for key, replacement in replacements.items():
                value = value.replace(key, replacement)
            if re.search(r"__[A-Z_]+__", value):
                raise ValueError("unresolved CI template placeholder")
        return value

    # Quote complete Docker arguments, rather than substituting shell fragments.
    docker = shlex.split(cfg["reliability"]["env"]["DOCKER_RUN_ARGS"])
    cfg = expand(cfg)
    cfg["reliability"]["env"]["DOCKER_RUN_ARGS"] = shlex.join([expand(arg) for arg in docker])
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(mode=0o700, exist_ok=False)
    (run / "env").mkdir()
    shutil.copyfile(ST / "config/container.ci.env.sh", run / "env/container.ci.env.sh")
    (run / "pool.hostfile").write_text("\n".join(nodes) + "\n", encoding="utf-8")
    site = run / "site.toml"
    site.write_text(dump_toml(cfg), encoding="utf-8")
    site.chmod(0o600)
    load_site(site)
    write_json(run / "ci-run.json", {"schema_version": 1, "run_key": run_key, "run_root": str(run),
                                   "repo_root": str(repo), "nodes": nodes, "site": str(site),
                                   "nodes_source": source, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                                   "hostfile_sha256": hashlib.sha256(Path(hostfile).read_bytes() if hostfile else
                                                                     (run / "pool.hostfile").read_bytes()).hexdigest()})
    return run, cfg


def ssh(node, script, args=(), timeout=40):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node,
           shlex.join(["bash", "-s", "--", *map(str, args)])]
    return subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout, check=True)


def preflight(run, cfg):
    if sys.platform != "linux":
        raise ValueError("real CI execution requires Linux")
    python = cfg["paths"]["control_python"]
    subprocess.run([python, "-B", "-c", "import sys; assert (3, 11) <= sys.version_info < (3, 13); import pytest, ClusterShell, yaml, requests, pytz, urllib3, torch"],
                   check=True, timeout=60)
    for binary in ("git", "ssh", "bash", "clush"):
        if not shutil.which(binary):
            raise ValueError(f"controller command missing: {binary}")
    repo = Path(cfg["paths"]["repo_root"])
    for entry in ("hcu-envcheck/bin/hcu-cluster-run", "hcu-training-reliability/cluster_manager/main.py"):
        if not (repo / entry).is_file():
            raise ValueError(f"checkout missing {entry}")
    # Share the product's supported tool discovery, including DTK absolute paths.
    idle_probe = (repo / "hcu-envcheck/hcu_envcheck/payloads/idle_probe.sh").read_text(encoding="utf-8")
    marker = run / "shared-marker"
    marker.write_text(run.name, encoding="utf-8")
    ci = cfg["ci"]
    records = []
    users = set()
    for node in cfg["site"]["nodes"]:
        resolved = subprocess.run(["ssh", "-G", node], capture_output=True, text=True, timeout=20, check=True)
        fields = dict(line.split(None, 1) for line in resolved.stdout.splitlines() if len(line.split(None, 1)) == 2)
        if fields.get("port") != "22" or not fields.get("user"):
            raise ValueError(f"{node}: hcu-envcheck currently requires SSH port 22 and a resolved user")
        users.add(fields["user"])
        result = ssh(node, idle_probe + '''
set -eu
test "$(cat "$1")" = "$2"
hcu_idle_init host
test -x "$3/bin/run_nhc"
test -s "$3/node_check/etc/nhc.json"
test "$(cat "$3/NHC_COMMIT")" = "$4"
test "$(sha256sum "$3/bin/run_nhc" | cut -d ' ' -f1)" = "$5"
shift 5
image=$1; shift
for peer in "$@"; do getent hosts "$peer" >/dev/null; done
docker image inspect --format '{{.Id}}' "$image"
''', (marker, run.name, ci["nhc_root"], ci["nhc_commit"], ci["nhc_binary_sha256"],
      cfg["site"]["image"], *cfg["site"]["nodes"]), timeout=60)
        image_id = result.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError(f"{node}: invalid image identity")
        records.append({"node": node, "image_id": image_id, "ssh_user": fields["user"]})
    if len(users) != 1 or len({r["image_id"] for r in records}) != 1:
        raise ValueError("pool must have one resolved SSH user and matching image IDs")
    write_json(run / "preflight.json", records)


def _manager_alive(manager, run_id):
    """Verify the recorded identity before considering a manager safe to signal."""
    pid = manager.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise ValueError("unverified manager identity")
    proc = Path(f"/proc/{pid}")
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if (len(fields) < 20 or manager.get("pgid") != pid
                or fields[19] != manager.get("starttime")):
            raise ValueError("manager PID identity changed; refusing signal")
        if fields[0] in {"Z", "X"}:
            return False
        cmd = (proc / "cmdline").read_bytes().decode().split("\0")
        if (os.getpgid(pid) != pid or "--job_name" not in cmd
                or cmd[cmd.index("--job_name") + 1] != run_id):
            raise ValueError("manager PID identity changed; refusing signal")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except IndexError as error:
        raise ValueError("unverified manager process evidence") from error
    return True


def _stop_manager(manager, run_id):
    if not manager:
        return
    # closed/exited are historical receipts, not proof that a process is gone.
    for sig, budget in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        if not _manager_alive(manager, run_id):
            return
        try:
            os.killpg(manager["pid"], sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + budget
        while _manager_alive(manager, run_id):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    if _manager_alive(manager, run_id):
        raise ValueError("manager stop unconfirmed; refusing container removal")


def cleanup(run):
    """Read only this CI run's registry; never enumerate/delete other containers."""
    run = Path(run).resolve()
    marker = json.loads((run / "ci-run.json").read_text(encoding="utf-8"))
    if Path(marker["run_root"]).resolve() != run:
        raise ValueError("CI run marker mismatch")
    errors = []
    for path in sorted((run / "cases").glob("*/ownership.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            case = path.parent.resolve()
            rid = data.get("run_id", "")
            name = data.get("container_name", "")
            if (not case.is_relative_to(run / "cases") or Path(data["case_root"]).resolve() != case
                    or not re.fullmatch(r"hcust[0-9]{14}[0-9a-f]{10}", rid)
                    or name != case.name or name not in {rid + "-env-001", rid + "-rel-010"}
                    or (case / "ST_RUN_ID").read_text().strip() != rid
                    or data.get("nodes") != marker["nodes"]):
                raise ValueError("case registry identity mismatch")
            if data.get("manager_launching") and not data.get("manager"):
                raise ValueError("manager launch interrupted before identity registration; manual reconciliation required")
            _stop_manager(data.get("manager"), rid)
            registered = data.get("containers", {})
            if not set(registered).issubset(data.get("absent_before_start", [])):
                raise ValueError("registered container predates the case or has an unknown node")
            for node in marker["nodes"]:
                if node not in data.get("absent_before_start", []):
                    continue
                item = registered.get(node)
                if item is None:
                    # Startup can be interrupted before the first registration poll.
                    # Adopt only the exact new name with this case's mounted marker.
                    if not data.get("manager"):
                        continue
                    response = ssh(node, '''set -eu
if ! docker container inspect "$1" >/dev/null 2>&1; then docker info >/dev/null; printf 'null\\n'; exit 0; fi
docker inspect -f '{"id":{{json .Id}},"image_id":{{json .Image}},"name":{{json .Name}},"pid_mode":{{json .HostConfig.PidMode}}}' "$1"
''', (name,))
                    item = json.loads(response.stdout)
                    if item is None:
                        continue
                    if item.get("name") != "/" + name or item.get("pid_mode") not in ("", "private"):
                        raise ValueError("unregistered container identity changed")
                    item["owner_verified"] = True  # Removal still requires the shared marker below.
                cid = item.get("id", "")
                if not item.get("owner_verified") or not re.fullmatch(r"[0-9a-f]{64}", cid):
                    raise ValueError("unverified container registry")
                ssh(node, '''set -eu
if ! docker container inspect "$1" >/dev/null 2>&1; then docker info >/dev/null; exit 0; fi
test "$(docker inspect -f '{{.Name}}' "$1")" = "/$2"
test "$(docker inspect -f '{{.Image}}' "$1")" = "$3"
case "$(docker inspect -f '{{.HostConfig.PidMode}}' "$1")" in ''|private) ;; *) exit 1 ;; esac
test "$(docker exec "$1" cat "$4/ST_RUN_ID")" = "$5"
docker rm -f -- "$1" >/dev/null
if docker container inspect "$1" >/dev/null 2>&1; then exit 1; fi
docker info >/dev/null
''', (cid, name, item["image_id"], case, rid), timeout=60)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            errors.append(f"{path.parent.name}: {error}")
    write_json(run / "ci-cleanup.json", {"confirmed": not errors, "errors": errors})
    return not errors


def collect(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination == run or destination.is_relative_to(run):
        raise ValueError("artifact destination must be outside the shared run")
    destination.mkdir(parents=True, exist_ok=True)
    for p in run.rglob("*"):
        relative = p.relative_to(run)
        if any(part in {"checkpoints", ".git", "__pycache__"} for part in relative.parts):
            continue
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(run):
            continue
        if p.suffix not in {".log", ".json", ".tsv", ".toml", ".sh", ".py", ".hostfile"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    files = [{"path": str(p.relative_to(run)), "size": p.stat().st_size}
             for p in run.glob("cases/*/checkpoints/**/*") if p.is_file() and not p.is_symlink()]
    write_json(destination / "checkpoint-files.json", files)


def execute(run, cfg):
    """Forward cancellation to the owned pytest group, then use the registry."""
    active = None
    interrupted = None
    handlers = {}

    def on_signal(number, _frame):
        nonlocal interrupted
        interrupted = interrupted or (number, time.monotonic())
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    for number in (signal.SIGINT, signal.SIGTERM):
        handlers[number] = signal.signal(number, on_signal)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment["PYTHONPATH"] = str(Path(cfg["paths"]["repo_root"]) / "hcu-training-reliability")

    def phase(name, args):
        nonlocal active
        if interrupted:
            raise InterruptedError("CI cancelled")
        print(f"CI_PHASE={name}", flush=True)
        with (run / f"{name}.log").open("wb") as log:
            active = subprocess.Popen([cfg["paths"]["control_python"], "-B", str(ST / "run_st.py"), *args],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      env=environment, start_new_session=True)
            try:
                while active.poll() is None:
                    if interrupted and time.monotonic() - interrupted[1] > 90:
                        os.killpg(active.pid, signal.SIGKILL)
                    time.sleep(0.2)
                if active.returncode:
                    raise subprocess.CalledProcessError(active.returncode, name)
            finally:
                if active.poll() is None:
                    os.killpg(active.pid, signal.SIGTERM)
                    active.wait(timeout=20)
                active = None
        if interrupted:
            raise InterruptedError("CI cancelled")

    try:
        phase("prepare", ["prepare", "--site", str(run / "site.toml"), "--dependencies", str(ST / "config/dependencies.toml"),
                          "--deps-root", str(run.parents[1] / "dependencies"),
                          "--evidence-dir", str(run / "bridge-dependency")])
        prepared = json.loads((run / "prepare.log").read_text(encoding="utf-8"))
        site = prepared["site"]
        shutil.copyfile(site, run / "site.prepared.toml")
        shutil.copyfile(prepared["manifest"], run / "bridge-manifest.json")
        phase("plan", ["plan", "--site", site])
        phase("pytest", ["run", "--site", site, "--case", "all"])
        return 0
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "run"):
        p = sub.add_parser(name)
        p.add_argument("--template", default=str(ST / "config/site.ci.toml"))
        p.add_argument("--hostfile", default=os.getenv("ST_HOSTFILE"))
        p.add_argument("--nodes", default=os.getenv("ST_NODES"), help="allocated nodes, comma/whitespace separated; alternative to --hostfile")
        p.add_argument("--control-python", default=os.getenv("ST_CONTROL_PYTHON"))
        p.add_argument("--shared-root", default=os.getenv("ST_SHARED_ROOT"))
        p.add_argument("--run-key", required=True)
        p.add_argument("--artifacts", required=name == "run")
    for name in ("cleanup", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--run-root", required=True)
        if name == "collect": p.add_argument("--artifacts", required=True)
    args = parser.parse_args(argv)
    run = None
    code = 0
    try:
        if args.command == "cleanup": return 0 if cleanup(args.run_root) else 1
        if args.command == "collect": collect(args.run_root, args.artifacts); return 0
        run, cfg = render(args.template, args.hostfile, args.control_python, args.shared_root, args.run_key, nodes=args.nodes)
        print(json.dumps({"run_root": str(run), "site": str(run / "site.toml"), "nodes": cfg["site"]["nodes"]}), flush=True)
        if args.command == "run":
            preflight(run, cfg)
            code = execute(run, cfg)
    except (OSError, ValueError, KeyError, RuntimeError, InterruptedError, subprocess.SubprocessError) as error:
        print(f"CI ERROR: {error}", file=sys.stderr)
        code = 130 if isinstance(error, InterruptedError) else getattr(error, "returncode", 1)
        code = code if isinstance(code, int) and 0 < code < 256 else 1
    finally:
        if run is not None and args.command == "run":
            try:
                if not cleanup(run): code = code or 1
            except Exception as error:
                print(f"CI CLEANUP ERROR: {error}", file=sys.stderr); code = code or 1
            try:
                write_json(run / "ci-result.json", {"exit_code": code})
                collect(run, args.artifacts)
            except Exception as error:
                print(f"CI COLLECT ERROR: {error}", file=sys.stderr); code = code or 1
                write_json(run / "ci-result.json", {"exit_code": code, "collection_error": str(error)})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
