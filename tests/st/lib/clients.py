# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real CLI orchestration for configurable training allocations; no work occurs on import."""
from __future__ import annotations

import csv
import json
import hashlib
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import time
from datetime import datetime, timezone

from run_st import validate_topology
from lib.evidence import Evidence, STError
from lib.envcheck import command as envcheck_command, freeze_source, receipt as envcheck_receipt
from lib.remote import Remote, Runner
from lib.training import checkpoint_files, inspect_workers, kill_worker, parse_training_log


def read_json(path):
    """A product may be writing its state file; retry incomplete JSON in the caller."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class CaseContext:
    """One independently owned training allocation and its evidence directory."""

    def __init__(self, config: dict, case_id: str):
        validate_topology(config, recovery=case_id == "REL-010")
        self.config = config
        self.case_id = case_id
        self.nodes = list(config["site"]["nodes"])
        self.training_nodes = config["training"]["nodes"]
        self.gpus = config["training"]["gpus"]
        self.world_size = self.training_nodes * self.gpus
        self.timeouts = config["timeouts"]
        self.run_id = "hcust" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + secrets.token_hex(5)
        self.container_name = self.run_id + "-" + case_id.lower()
        self.case_root = Path(config["paths"]["shared_root"]) / (self.run_id + "-" + case_id.lower())
        self.case_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.evidence = Evidence(self.case_root, case_id)
        self.runner = Runner(self.evidence, self.timeouts["command"])
        ssh = config.get("ssh", {})
        self.remote = Remote(self.runner, ssh.get("port", 22), ssh.get("connect_timeout", 10),
                             ssh.get("options", ()))
        self.manager = None
        self.manager_launching = False
        self.manager_env = None
        self.containers = {}
        self.absent_before_start = set()
        self.reports = []
        self.checkpoint_evidence = {}
        self.configuration = {}
        self.injection = None
        self.cleanup_errors = []
        self.collection_errors = []
        self.final_evidence = {}
        self.active_nodes = []
        self.standby_nodes = []
        self.recovery_nodes = []
        self.fault_node = None
        self.active_attempt = None
        self.total_deadline = time.monotonic() + self.timeouts["total"]
        self.phase_deadline = None
        self.closing = self.closed = False
        self.result = None
        self._persist_ownership()
        self.evidence.log(f"case={case_id} run={self.run_id} nodes={self.nodes} container={self.container_name}")

    def _persist_ownership(self):
        """Publish only resource identities, never environment or credentials."""
        manager = None
        if self.manager is not None:
            manager = {"pid": self.manager.pid}
            if os.name == "posix":
                try:
                    manager["starttime"] = Path(f"/proc/{self.manager.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                    manager["pgid"] = os.getpgid(self.manager.pid)
                except (OSError, IndexError):
                    manager["exited"] = True
        data = {"schema_version": 1, "run_id": self.run_id, "case_root": str(self.case_root.resolve()),
                "container_name": self.container_name, "manager": manager, "nodes": self.nodes,
                "absent_before_start": sorted(self.absent_before_start), "containers": self.containers,
                "closed": self.closed, "manager_launching": self.manager_launching}
        path = self.case_root / "ownership.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)

    def _remaining(self):
        deadlines = [d for d in (None if self.closing else self.total_deadline, self.phase_deadline) if d]
        remaining = min(deadlines) - time.monotonic() if deadlines else self.timeouts["command"]
        if remaining <= 0:
            raise STError("unclassified", "ST_TIMEOUT", "The current stage or total deadline expired")
        return min(float(self.timeouts["command"]), remaining)

    def _require(self, name, condition, detail, category="engineering", code="ASSERTION_FAILED", evidence=None):
        self.evidence.check(name, bool(condition), detail, evidence=evidence)
        if not condition:
            raise STError(category, code, detail)

    def _wait(self, stage, budget, observe):
        previous = self.phase_deadline
        deadline = time.monotonic() + budget
        self.phase_deadline = min(deadline, self.total_deadline)
        self.evidence.log(f"stage={stage} begin budget={budget}s")
        try:
            while time.monotonic() < self.phase_deadline:
                self._check_manager(stage)
                value = observe()
                self._check_manager(stage)
                self._remaining()
                if value:
                    self.evidence.log(f"stage={stage} complete")
                    return value
                state = read_json(self.case_root / "manager/workspace/running_state.json")
                if state and state.get("run_state") == "FAULT":
                    raise STError("engineering", "MANAGER_FAULT", f"manager entered FAULT while waiting for {stage}")
                time.sleep(min(float(self.timeouts["poll"]), max(0, self.phase_deadline - time.monotonic())))
            raise STError("unclassified", "ST_TIMEOUT", f"{stage}: expected condition not observed within {budget}s")
        finally:
            self.phase_deadline = previous

    def _check_manager(self, stage):
        if self.manager is not None and self.manager.poll() is not None:
            raise STError("unclassified", "MANAGER_EXITED",
                          f"manager exited ({self.manager.returncode}) while waiting for {stage}; see manager.log")

    def _inspect(self, node, name=None):
        script = """set -eu
command -v docker >/dev/null
if docker container inspect "$1" >/dev/null 2>&1; then
  docker inspect -f '{"id":{{json .Id}},"image_id":{{json .Image}},"name":{{json .Name}},"running":{{json .State.Running}},"pid_mode":{{json .HostConfig.PidMode}},"mounts":{{json .Mounts}}}' "$1"
else
  docker info >/dev/null
  printf 'null\n'
fi
"""
        result = self.remote.script(node, script, (name or self.container_name,), timeout=self._remaining())
        try:
            item = json.loads(result.stdout)
        except ValueError as error:
            raise STError("environment", "DOCKER_EVIDENCE_INVALID", f"{node}: invalid docker inspect: {error}") from error
        if item is not None and (not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{64}", item.get("id", ""))):
            raise STError("environment", "DOCKER_EVIDENCE_INVALID", f"{node}: invalid full container identity")
        return item

    def _register(self, node, item):
        if node in self.containers:
            if item["id"] != self.containers[node]["id"]:
                self._require("container_identity", False,
                              f"{node}: container ID changed during a worker-fault case")
            return
        self._require("container_ownership", self.manager is not None and node in self.absent_before_start
                      and item.get("name") == "/" + self.container_name,
                      f"{node}: only this manager's new, exact-name container can be registered", category="environment")
        self._require("container_pid_namespace", item.get("pid_mode", "") in ("", "private"),
                      f"{node}: container must have a private PID namespace", category="environment")
        self._require("container_proc_namespace", not any(m.get("Destination") == "/proc" for m in item.get("mounts", [])),
                      f"{node}: container /proc must not be replaced by a host mount", category="environment")
        # Capture the owned ID before readiness checks, so a bad mount cannot
        # strand a container that this manager just created.
        self.containers[node] = dict(item, owner_verified=False)
        self._persist_ownership()
        image = self.remote.script(node, "set -eu\ndocker image inspect -f '{{.Id}}' \"$1\"\n",
                                   (self.config["site"]["image"],), timeout=self._remaining()).stdout.strip()
        self._require("container_image", item["image_id"] == image,
                      f"{node}: actual image={item['image_id']} expected={image}", category="environment")
        marker = self.remote.script(node, "set -eu\ndocker exec \"$1\" cat \"$2\"\n",
                                    (item["id"], str(self.case_root / "ST_RUN_ID")), timeout=self._remaining()).stdout.strip()
        self._require("container_shared_path", marker == self.run_id,
                      f"{node}: case path must be mounted at the same absolute path", category="environment")
        self.containers[node]["owner_verified"] = True
        self._persist_ownership()
        self._require("container_running", item["running"] is True, f"{node}: container must be running")
        self.evidence.log(f"registered node={node} container_id={item['id']} image_id={item['image_id']}")

    def _envcheck(self, operation, *, nodes=None, release=False):
        nodes = list(nodes or self.nodes)
        call = self.case_root / "envcheck" / f"{len(self.reports) + 1:02d}-{operation}"
        call.mkdir(parents=True)
        hosts = call / "hostfile"
        hosts.write_text("\n".join(nodes) + "\n", encoding="utf-8")
        repo = Path(self.config["paths"]["repo_root"])
        env = dict(self.manager_env or os.environ)
        env.update(HCU_CONTROL_PYTHON=self.config["paths"]["control_python"], PYTHONDONTWRITEBYTECODE="1")
        # Shell idle/lifecycle tools use one explicit host user; resolve it from
        # the same OpenSSH configuration used by the manager and ST inspector.
        users = set()
        for node in nodes:
            resolved = self.runner.run(["ssh", "-G", node], timeout=self._remaining(), check=True)
            fields = dict(line.split(None, 1) for line in resolved.stdout.splitlines() if len(line.split(None, 1)) == 2)
            self._require("envcheck_ssh_port", fields.get("port") == "22",
                          f"{node}: hcu-envcheck backend currently requires SSH port 22", category="environment")
            users.add(fields.get("user"))
        self._require("envcheck_ssh_user", len(users) == 1 and None not in users,
                      "hcu-envcheck shell operations require one resolved SSH user for the node pool", category="environment")
        env["HCU_SSH_USER"] = users.pop()
        argv = envcheck_command(repo, operation, hosts, call, self.config, self.container_name)
        if operation == "platform":
            snapshot = self.case_root / "envcheck-source"
            if not snapshot.exists():
                freeze_source(repo, snapshot)
            argv += ["--container-workdir", str(snapshot)]
        if release:
            argv.append("--release")
        result = self.runner.run(argv, timeout=self._remaining(), env=env, check=False)
        (call / "command.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        try:
            report, path = envcheck_receipt(call, operation, nodes, result.returncode, repo=repo)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise STError("test-framework", "ENVCHECK_RECEIPT_INVALID", f"{operation}: {error}; see {call}") from error
        self.reports.append(str(path))
        return report

    @staticmethod
    def _idle_nodes(report):
        records = report.get("idle_check", {}).get("nodes", [])
        return {row.get("node"): row for row in records}

    def _ensure_idle(self):
        report = self._envcheck("idle-check")
        records = self._idle_nodes(report)
        self._require("idle_coverage", set(records) == set(self.nodes), "idle-check must return the entire node pool")
        busy = []
        for node, record in records.items():
            decision = record.get("decision", {})
            state = decision.get("status")
            self._require("idle_valid", state in {"IDLE", "BUSY"},
                          f"{node}: unknown or missing idle evidence", category="environment", code="IDLE_UNKNOWN")
            self._require("idle_card_count", len(decision.get("cards", [])) >= self.gpus,
                          f"{node}: fewer than {self.gpus} cards in host idle evidence", category="environment")
            if state == "BUSY":
                busy.append(node)
                allowed = node in self.config["site"].get("release_nodes", [])
                self._require("release_scope", allowed, f"{node}: BUSY and not declared releasable in site.release_nodes",
                              category="environment", code="RELEASE_NOT_AUTHORIZED")
                owners = record.get("owners", [])
                self._require("release_owners", bool(owners) and not record.get("error"),
                              f"{node}: occupation ownership is incomplete", category="environment")
                protected = set(self.config["site"].get("protected_containers", []))
                hits = [o for o in owners if o.get("container") in protected or o.get("container_id") in protected]
                self._require("release_protected", not hits, f"{node}: protected workload owns devices; not releasing",
                              category="environment", code="PROTECTED_WORKLOAD")
        if busy:
            released = self._envcheck("idle-check", nodes=busy, release=True)
            final = self._idle_nodes(released)
            actions_ok = all(
                final.get(n, {}).get("initial_decision", {}).get("status") == "IDLE" or (
                    final.get(n, {}).get("action_status") == "CONFIRMED"
                    and bool(final.get(n, {}).get("actions"))
                    and all(a.get("status") == "CONFIRMED" for a in final[n]["actions"])
                ) for n in busy
            )
            self._require("release_confirmed", released.get("status") == "PASS" and all(
                final.get(n, {}).get("decision", {}).get("status") == "IDLE" and
                final[n].get("action_status") != "UNCONFIRMED" for n in busy) and actions_ok,
                "release must finish with confirmed IDLE, not merely a successful stop command", category="environment")
            records = self._idle_nodes(self._envcheck("idle-check"))
        self._require("training_pre_idle", set(records) == set(self.nodes) and all(
            records[n].get("decision", {}).get("status") == "IDLE"
            and len(records[n].get("decision", {}).get("cards", [])) >= self.gpus for n in self.nodes),
            "all pool nodes must be IDLE immediately before the training case", category="environment")

    def _source_environment(self):
        env = dict(os.environ)
        profile = self.config.get("reliability", {}).get("env_script", "")
        if profile:
            capture = self.runner.run(["bash", "--noprofile", "--norc", "-c",
                                       'set -e; set -a; source "$1"; set +a; printf "\\0HCU_ST_ENV\\0"; env -0',
                                       "st-profile", profile], timeout=self._remaining(), check=True)
            marker = "\x00HCU_ST_ENV\x00"
            self._require("profile_environment", marker in capture.stdout,
                          "environment profile did not complete", category="environment")
            for entry in capture.stdout.split(marker, 1)[1].split("\x00"):
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                        env[key] = value
        env.update({str(k): str(v) for k, v in self.config.get("reliability", {}).get("env", {}).items()})
        return env

    @staticmethod
    def _validate_docker_args(raw):
        """The product's process stop must remain inside this case's PID space."""
        if any(symbol in raw for symbol in ("$", "`", ";", "\n", "\r", "|", "&")):
            raise STError("environment", "DYNAMIC_DOCKER_ARGS", "DOCKER_RUN_ARGS must be resolved literal arguments")
        try:
            args = shlex.split(raw)
        except ValueError as error:
            raise STError("environment", "INVALID_DOCKER_ARGS", str(error)) from error
        for index, arg in enumerate(args):
            flag, _, inline = arg.partition("=")
            value = inline if "=" in arg else (args[index + 1] if index + 1 < len(args) else "")
            if flag == "--name":
                raise STError("environment", "CONTAINER_NAME_OVERRIDE", "DOCKER_RUN_ARGS cannot override the case container name")
            if flag == "--pid" and value != "private":
                raise STError("environment", "SHARED_PID_NAMESPACE", "host/container PID sharing is incompatible with isolated ST")
            if flag in ("-v", "--volume") and len(value.split(":")) >= 2 and value.split(":")[1] == "/proc":
                raise STError("environment", "HOST_PROC_MOUNT", "container /proc cannot be a host bind mount")
            if flag == "--mount" and any(part in ("target=/proc", "dst=/proc", "destination=/proc") for part in value.split(",")):
                raise STError("environment", "HOST_PROC_MOUNT", "container /proc cannot be a host bind mount")

    @staticmethod
    def _write_exports(path, values):
        lines = ["# ST-generated runtime context; this file is not repository source."]
        for key, value in values.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or "\x00" in str(value):
                raise STError("test-framework", "INVALID_ENVIRONMENT", f"invalid environment key/value: {key}")
            lines.append(f"export {key}={shlex.quote(str(value))}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o600)

    def _prepare_runtime(self):
        cfg = self.config
        for folder in ("manager", "monitor", "attempts", "payloads", "checkpoints/global", "checkpoints/local"):
            (self.case_root / folder).mkdir(parents=True)
        self.hostfile = self.case_root / "hostfile"
        self.hostfile.write_text("\n".join(self.nodes) + "\n", encoding="utf-8")
        (self.case_root / "ST_RUN_ID").write_text(self.run_id + "\n", encoding="utf-8")
        source = Path(cfg["training"]["script"])
        shutil.copyfile(source, self.case_root / "payloads/train_st.sh")
        wrapper = Path(__file__).resolve().parents[1] / "payloads/run_train.sh"
        shutil.copyfile(wrapper, self.case_root / "payloads/run_train.sh")
        for name in ("st_run_recipe.py", "st_glm53_recipe.py"):
            shutil.copyfile(wrapper.parent / name, self.case_root / "payloads" / name)
        training_env = dict(cfg["training"].get("env", {}))
        training_env.update(GPUS=self.gpus, NNODES=self.training_nodes, MODE="train", TRAIN_ITERS=cfg["training"]["train_iters"],
                            SAVE_INTERVAL=cfg["training"]["global_interval"],
                            LOCAL_SAVE_INTERVAL=cfg["training"]["local_interval"],
                            MEGATRON_BRIDGE_PATH=cfg["paths"]["bridge_root"])
        if cfg["paths"].get("hf_path"):
            training_env["LOCAL_HF_PATH"] = cfg["paths"]["hf_path"]
        if cfg["reliability"].get("train_config") == "auto:glm53-st":
            from payloads.st_glm53_recipe import settings
            training_env.update(settings(training_env))
            training_env["HCU_ST_PREPARE_DATASET_HELPERS"] = "1"
        self._write_exports(self.case_root / "training.env", training_env)
        context = {"HCU_ST_CASE_ROOT": self.case_root, "HCU_ST_RUN_ID": self.run_id,
                   "HCU_ST_TRAIN_SCRIPT": self.case_root / "payloads/train_st.sh",
                   "HCU_ST_PYTHON": cfg["training"].get("python", "python3")}
        self._write_exports(self.case_root / "payloads/context.env", context)
        env = self._source_environment()
        self._validate_docker_args(env.get("DOCKER_RUN_ARGS", ""))
        port = str(env.get("MPI_HOST_SSH_PORT") or env.get("MPIRUN_PLM_RSH_ARGS", "22")).strip()
        if port.startswith("-p"):
            port = port[2:].strip()
        self._require("ssh_port_consistent", port.isdigit() and int(port) == int(cfg["ssh"]["port"]),
                      "site SSH port must agree with the existing reliability environment", category="environment")
        train_config = cfg["reliability"].get("train_config") or env.get("MEGATRON_SCRIPT_PATH")
        self._require("manager_train_config", train_config == "auto:glm53-st" or bool(train_config) and Path(train_config).is_file(),
                      "an existing reliability-compatible training config is required; ST does not infer Hydra model settings",
                      category="dependency-image", code="TRAIN_CONFIG_MISSING")
        if train_config == "auto:glm53-st":
            from payloads.st_glm53_recipe import manager_config
            frozen_config = self.case_root / "manager-train-config.json"
            frozen_config.write_text(json.dumps(manager_config(training_env), indent=2) + "\n", encoding="utf-8")
        else:
            frozen_config = self.case_root / ("manager-train-config" + Path(train_config).suffix)
            shutil.copyfile(train_config, frozen_config)
        env.update({"CLUSTER_LAUNCH_MODE": "docker", "CLUSTER_SCHEDULE": "NONE",
                    "WORK_DIR": str(self.case_root / "manager"), "LOG_DIR": str(self.case_root / "monitor"),
                    "BLACKLIST_PERSISTENCE_PATH": str(self.case_root / "manager/blacklist.json"),
                    "BLACKLIST_PERSISTENCE_BACKUP_PATH": str(self.case_root / "manager/blacklist.json.bak"),
                    "DOCKER_CONTAINER_NAME": self.container_name, "DOCKER_IMAGE": cfg["site"]["image"],
                    "DOCKER_REUSE_CONTAINER": "0", "DOCKER_REMOVE_CONTAINER_ON_STOP": "0",
                    "DOCKER_EXEC_PATH": str(self.case_root / "payloads/run_train.sh"),
                    "MEGATRON_SCRIPT_PATH": str(frozen_config), "PYTHONDONTWRITEBYTECODE": "1"})
        # docker exec does not inherit this process's environment. The wrapper's
        # sibling context.env provides its minimal startup context inside Docker.
        self.manager_env = env
        self.runner.environment = env
        self.configuration = {
            "site_file": cfg.get("_site_file"), "repo_root": str(cfg["paths"]["repo_root"]),
            "image": cfg["site"]["image"], "script_source": str(source),
            "script_sha256": hashlib.sha256((self.case_root / "payloads/train_st.sh").read_bytes()).hexdigest(),
            "reliability_profile": cfg["reliability"].get("env_script", ""),
            "train_config_source": str(train_config), "train_config_copy": str(frozen_config),
            "train_config_sha256": hashlib.sha256(frozen_config.read_bytes()).hexdigest(),
            "training": {key: cfg["training"].get(key) for key in
                         ("nodes", "gpus", "local_interval", "global_interval", "synchronous", "inner_restarts", "train_iters")},
            "timeouts": dict(self.timeouts)}
        self.configuration["payload_sha256"] = {
            name: hashlib.sha256((self.case_root / "payloads" / name).read_bytes()).hexdigest()
            for name in ("train_st.sh", "run_train.sh", "st_run_recipe.py", "st_glm53_recipe.py")}
        if "bridge_manifest" in cfg.get("dependencies", {}):
            from prepare_dependencies import prepared_metadata
            metadata = prepared_metadata(cfg)
            frozen = self.case_root / "bridge-dependency.json"
            frozen.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.configuration["bridge_dependency"] = {"record": str(frozen), "bridge": metadata["bridge"],
                                                       "source_only": metadata["source_only"], "warnings": metadata["warnings"]}
        self.evidence.log(f"runtime script={source} checkpoint roots={self.case_root / 'checkpoints'}")

    def _workers(self, node):
        return inspect_workers(self.remote, node, self.containers[node]["id"], self.run_id,
                               self.config["training"].get("worker_pattern", "run_recipe.py"),
                               self.config["training"].get("python", "python3"), self._remaining())

    def _attempts(self, node=None):
        found = []
        for path in (self.case_root / "attempts").glob("*/attempt.json"):
            item = read_json(path)
            if not isinstance(item, dict):
                continue
            slots = item.get("slots", [])
            hosts = [str(row).split()[0] for row in slots if str(row).strip()]
            if hosts and (node is None or node in hosts):
                log = Path(item.get("log", path.parent / "train.log"))
                try:
                    if not log.resolve().is_relative_to(self.case_root.resolve()):
                        raise ValueError("attempt log escapes the case directory")
                    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
                except (OSError, ValueError) as error:
                    raise STError("test-framework", "ATTEMPT_EVIDENCE_INVALID", str(error)) from error
                found.append((item, parse_training_log(text)))
                self.evidence.logs["training:" + str(item.get("attempt_id", path.parent.name))] = str(log)
        return sorted(found, key=lambda pair: pair[0].get("started_epoch", 0))

    def _valid_workers(self, evidence):
        workers = evidence.get("workers", [])
        return (len(workers) == self.gpus
                and {p.get("local_rank") for p in workers} == set(range(self.gpus))
                and all(p.get("world_size") == self.world_size for p in workers))

    @staticmethod
    def _attempt_nodes(item):
        return [str(row).split()[0] for row in item.get("slots", []) if str(row).strip()]

    def _allocation(self, item, observations):
        """A complete allocation has one shared attempt and every global rank once."""
        nodes = self._attempt_nodes(item)
        attempt_id = item.get("attempt_id")
        if (not attempt_id or len(nodes) != self.training_nodes or len(set(nodes)) != len(nodes)
                or not set(nodes).issubset(self.nodes)):
            return False
        workers = []
        for node in self.nodes:
            evidence = observations[node]
            if node in nodes:
                if not self._valid_workers(evidence):
                    return False
                workers.extend(evidence["workers"])
            elif evidence.get("owned_processes"):
                return False
        return (all(w.get("attempt_id") == attempt_id for w in workers)
                and {w.get("rank") for w in workers} == set(range(self.world_size)))

    def start(self):
        self.phase_deadline = min(self.total_deadline, time.monotonic() + self.timeouts["prepare"])
        try:
            self._prepare_runtime()
            for node in self.nodes:
                self._require("unique_container_name", self._inspect(node) is None,
                              f"{node}: generated container name must not already exist", category="environment")
                self.absent_before_start.add(node)
            self._persist_ownership()
            self._ensure_idle()
            repo = Path(self.config["paths"]["repo_root"])
            argv = [self.config["paths"]["control_python"], "-B",
                    str(repo / "hcu-training-reliability/cluster_manager/main.py"),
                    "--nodes_num", str(self.training_nodes), "--slots", str(self.gpus), "--mpi-ranks-per-node", "1",
                    "--exec", str(self.case_root / "payloads/run_train.sh"),
                    "--hostfile", str(self.hostfile), "--job_name", self.run_id]
            self.manager_launching = True
            self._persist_ownership()
            try:
                self.manager = self.runner.start(argv, self.case_root / "manager.log", env=self.manager_env,
                                                 cwd=self.case_root / "manager")
            except STError as error:
                if error.code == "COMMAND_START_FAILED":
                    self.manager_launching = False
                    self._persist_ownership()
                raise
            self.manager_launching = False
            self._persist_ownership()
        finally:
            self.phase_deadline = None

        def ready():
            for node in self.nodes:
                item = self._inspect(node)
                if item:
                    self._register(node, item)
            if len(self.containers) != len(self.nodes):
                return False
            self._require("cluster_image_consistent", len({item["image_id"] for item in self.containers.values()}) == 1,
                          "all pool nodes must run the same actual image ID", category="environment")
            state = read_json(self.case_root / "manager/workspace/running_state.json") or {}
            observations = {node: self._workers(node) for node in self.nodes}
            if state.get("run_state") == "RUNNING":
                for item, parsed in reversed(self._attempts()):
                    if parsed["iterations"] and self._allocation(item, observations):
                        self.active_attempt = item
                        self.active_nodes = self._attempt_nodes(item)
                        self.standby_nodes = [n for n in self.nodes if n not in self.active_nodes]
                        self.evidence.check("training_started", True,
                                            f"training={self.active_nodes} standby={self.standby_nodes} workers={self.world_size}")
                        return True
            return False

        self._wait("training allocation ready", self.timeouts["startup"], ready)

    def _check_platform(self, report):
        """Retain raw health results; only explicit WARN reason codes may be waived."""
        allowed = set(self.config.get("platform", {}).get("ignored_reason_codes", []))
        ignored = []
        accepted = report.get("status") == "PASS"
        if report.get("status") == "WARN" and allowed:
            path = Path(self.reports[-1]).with_name("platform-findings.tsv")
            try:
                with path.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if not {"node", "severity", "reason_code", "message"}.issubset(reader.fieldnames or []):
                        raise ValueError("invalid platform findings columns")
                    findings = list(reader)
                rows = report.get("results", [])
                warn_nodes = {row["node"] for row in rows if row.get("status") == "WARN"}
                accepted = (bool(findings) and bool(warn_nodes)
                            and all(row.get("status") in {"PASS", "WARN"}
                                    and str(row.get("exit_code")) == "0"
                                    and str(row.get("transport_exit_code")) == "0" for row in rows)
                            and warn_nodes == {row.get("node") for row in findings}
                            and all(row.get("severity") == "WARN" and row.get("reason_code") in allowed
                                    and row.get("node") in self.nodes for row in findings))
                if accepted:
                    ignored = findings
            except (OSError, ValueError, csv.Error) as error:
                self.evidence.log(f"platform waiver evidence invalid: {error}")
        self.configuration["platform_policy"] = {
            "ignored_reason_codes": sorted(allowed), "reported_status": report.get("status"),
            "ignored_findings": ignored,
        }
        if ignored:
            self.evidence.log("platform WARN waived by site configuration: "
                              + json.dumps(ignored, ensure_ascii=False))
        self._require("platform", accepted,
                      f"platform health={report.get('status')}; explicitly ignored WARN findings={len(ignored)}; "
                      "original report retained",
                      category="environment", code="PLATFORM_NOT_READY",
                      evidence=self.configuration["platform_policy"])

    def check_environment(self):
        previous = self.phase_deadline
        self.phase_deadline = min(self.total_deadline, time.monotonic() + self.timeouts["prepare"])
        try:
            status = self._envcheck("container-status")
            self._require("container_status", status.get("status") == "PASS",
                          "all pool containers must be running with matching image identities", category="environment")
            for row in status["results"]:
                self._require("container_status_identity", row.get("container_id") == self.containers[row["node"]]["id"],
                              f"{row['node']}: envcheck receipt must match the registered full container ID")
            platform = self._envcheck("platform")
            self._check_platform(platform)
            busy = self._idle_nodes(self._envcheck("idle-check"))
            self._require("training_idle_evidence", set(busy) == set(self.nodes) and all(
                busy[node].get("decision", {}).get("status") in {"IDLE", "BUSY"} for node in self.nodes),
                "training-time idle samples must be valid on all pool nodes", category="environment")
            for node in self.active_nodes:
                self._require("training_busy", busy[node].get("decision", {}).get("status") == "BUSY",
                              f"{node}: active training must be detected BUSY")
            for node in self.standby_nodes:
                self._require("standby_idle", busy[node].get("decision", {}).get("status") == "IDLE",
                              f"{node}: backup must remain IDLE", category="environment")
        finally:
            self.phase_deadline = previous

    def recover(self):
        local_enabled = self.config["training"]["local_interval"] > 0
        def saved():
            attempts = [pair for pair in self._attempts()
                        if pair[0].get("attempt_id") == self.active_attempt.get("attempt_id")]
            if not attempts:
                return False
            item, parsed = attempts[-1]
            local_saves = [step for step in parsed["local_saves"] if step > 0]
            global_saves = [step for step in parsed["global_saves"] if step > 0
                            and parsed["save_formats"].get(step) == "torch_dist"]
            if not global_saves or local_enabled and not local_saves:
                return False
            kinds = ("local", "global") if local_enabled else ("global",)
            files = {kind: checkpoint_files(self.case_root / "checkpoints" / kind) for kind in kinds}
            if not all(files.values()):
                return False
            self.active_attempt = item
            self.checkpoint_evidence = {"local_enabled": local_enabled,
                                        "local_status": "PASS" if local_enabled else "NOT_EXECUTED",
                                        "local_iteration": local_saves[-1] if local_enabled else None,
                                        "torch_dist_iteration": global_saves[-1], "files": files}
            if local_enabled:
                self.evidence.check("checkpoint_local_saved", True, f"iteration={local_saves[-1]}", evidence=files["local"])
            else:
                self.evidence.log("local checkpoint NOT_EXECUTED: local_interval=0; checking torch_dist only")
            self.evidence.check("checkpoint_torch_dist_saved", True, f"iteration={global_saves[-1]} format=torch_dist", evidence=files["global"])
            return True

        self._wait("both checkpoint saves" if local_enabled else "torch_dist checkpoint save", self.timeouts["checkpoint"], saved)
        self.phase_deadline = min(self.total_deadline, time.monotonic() + self.timeouts["command"])
        try:
            observations = {node: self._workers(node) for node in self.nodes}
            self._require("fault_target_allocation", self._allocation(self.active_attempt, observations),
                          f"a complete current {self.training_nodes}x{self.gpus} allocation is required before injection")
            self.fault_node = self.active_nodes[0]
            workers = observations[self.fault_node]
            self._require("inner_restarts_disabled", all(str(w.get("max_restarts")) == "0"
                          for node in self.active_nodes for w in observations[node]["workers"]),
                          "TORCHELASTIC_MAX_RESTARTS must be 0 in the actual workers", category="dependency-image")
            targets = [w for w in workers["workers"] if w.get("local_rank") == 1]
            self._require("fault_target_unique", len(targets) == 1, "LOCAL_RANK=1 must identify exactly one worker")
            self.injection = kill_worker(self.remote, self.fault_node, self.containers[self.fault_node]["id"],
                                         targets[0], self.run_id,
                                         self.config["training"].get("worker_pattern", "run_recipe.py"),
                                         self.config["training"].get("python", "python3"), self._remaining(),
                                         world_size=self.world_size)
            self.injection["node"] = self.fault_node
            self.evidence.log("injection=" + json.dumps(self.injection, ensure_ascii=False))
        finally:
            self.phase_deadline = None

        def recovered():
            for node in self.nodes:
                item = self._inspect(node)
                self._require("recovery_container_identity", item is not None and item["id"] == self.containers[node]["id"],
                              f"{node}: a worker fault must preserve this case's registered container")
            observations = {node: self._workers(node) for node in self.nodes}
            old_id = self.active_attempt.get("attempt_id")
            if any(p.get("attempt_id") == old_id for sample in observations.values()
                   for p in sample.get("owned_processes", [])):
                return False
            for item, parsed in reversed(self._attempts()):
                nodes = self._attempt_nodes(item)
                replacements = sorted(set(nodes) & set(self.standby_nodes))
                if (item.get("attempt_id") == old_id or self.fault_node in nodes
                        or not replacements or not self._allocation(item, observations)):
                    continue
                loads = [row for row in parsed["loads"] if row.get("iteration", 0) > 0]
                if not loads:
                    continue
                loaded = loads[-1]
                load_root = Path(loaded["path"]).resolve()
                checkpoint_root = self.case_root / "checkpoints"
                if not local_enabled:
                    checkpoint_root = checkpoint_root / "global"
                checkpoint_root = checkpoint_root.resolve()
                self._require("checkpoint_load_scope", load_root == checkpoint_root or load_root.is_relative_to(checkpoint_root),
                              f"restarted training must load a checkpoint under {checkpoint_root}: {loaded['path']}")
                if not any(step > loaded["iteration"] for step in parsed["load_progress"]):
                    continue
                if (read_json(self.case_root / "manager/workspace/running_state.json") or {}).get("run_state") != "RUNNING":
                    return False
                self.evidence.check("manager_selected_backup", True,
                                    "manager recognizes the restarted training as RUNNING")
                self.recovery_nodes = nodes
                self.evidence.check("backup_replacement", True, f"fault={self.fault_node} replacements={replacements} training={nodes}")
                self.evidence.check("backup_checkpoint_loaded", True, f"nodes={nodes} iteration={loaded['iteration']} path={loaded['path']}")
                self.evidence.check("backup_progress", True, f"iteration={max(parsed['load_progress'])} > loaded={loaded['iteration']}")
                self.evidence.check("old_workers_exited", True, f"attempt={old_id}: no old training processes on any pool node")
                self.checkpoint_evidence["recovery"] = {"nodes": nodes, "replacements": replacements, "attempt_id": item.get("attempt_id"),
                                                         "loaded": loaded, "last_iteration": max(parsed["load_progress"])}
                return True
            return False

        self._wait("replacement checkpoint load and progress", self.timeouts["recovery"], recovered)

    def _collect(self):
        """Retain small final observations; original logs already live in the case."""
        self.final_evidence["manager_state"] = read_json(self.case_root / "manager/workspace/running_state.json")
        attempts = []
        for item, parsed in self._attempts():
            self._remaining()
            attempts.append({"nodes": self._attempt_nodes(item), "attempt_id": item.get("attempt_id"),
                             "status": item.get("status"), "exit_code": item.get("exit_code"),
                             "log": item.get("log"), "observed": parsed})
        for node in self.nodes:
            self._remaining()
            if node in self.containers:
                self.final_evidence.setdefault("containers", {})[node] = self._inspect(node)
        self.final_evidence["attempts"] = attempts
        self.evidence.check("evidence_collection", True, "original manager/attempt logs and final observations retained")

    def close(self, outcome="PASS", category="", reason=""):
        if self.closed:
            return self.result
        self.closing = True
        functional_outcome = outcome
        stop_started = time.monotonic()
        self.phase_deadline = time.monotonic() + self.timeouts["cleanup"]
        try:
            if self.manager is not None:
                try:
                    self.runner.stop(self.manager, self._remaining())
                    if self.manager.poll() is None:
                        raise RuntimeError("manager remained alive; do not delete its containers while it can relaunch")
                except Exception as error:
                    self.cleanup_errors.append(str(error))
            cleanup_remaining = max(0.0, self.timeouts["cleanup"] - (time.monotonic() - stop_started))
            self.phase_deadline = time.monotonic() + self.timeouts["collect"]
            try:
                self._collect()
            except Exception as error:
                self.collection_errors.append(str(error))
                self.evidence.check("evidence_collection", False, str(error))
            # Collection has its own budget; preserve the unused stop/cleanup
            # budget instead of consuming it while reading evidence.
            self.phase_deadline = time.monotonic() + cleanup_remaining
            if self.manager is None or self.manager.poll() is not None:
                for node in self.nodes:
                    try:
                        item = self._inspect(node) if node in self.absent_before_start and (self.manager is not None or node in self.containers) else None
                        if not item:
                            continue
                        if node not in self.containers:
                            # A failed startup can create a container before the ready
                            # poll registers it. Adopt only through the normal image
                            # and case-marker checks, never from its name alone.
                            self._register(node, item)
                        expected = self.containers[node]["id"]
                        if item["id"] != expected:
                            raise RuntimeError(f"{node}: container ID changed; unknown replacement is left untouched")
                        result = self.remote.script(node, "docker rm -f -- \"$1\"\n", (expected,),
                                                    timeout=self._remaining(), check=False)
                        if result.returncode != 0 or self._inspect(node) is not None:
                            raise RuntimeError(f"{node}: container removal was not confirmed")
                    except Exception as error:
                        self.cleanup_errors.append(str(error))
        finally:
            if self.collection_errors and outcome == "PASS":
                outcome, category, reason = "FAIL", "test-framework", "EVIDENCE_COLLECTION_FAILED: " + "; ".join(self.collection_errors)
            if self.cleanup_errors:
                self.evidence.check("cleanup", False, "; ".join(self.cleanup_errors))
                if outcome == "PASS":
                    outcome, category, reason = "FAIL", "environment", "CLEANUP_FAILED: " + "; ".join(self.cleanup_errors)
            else:
                self.evidence.check("cleanup", True, "manager stopped and owned containers removed; run evidence retained")
            extra = {"run_id": self.run_id, "nodes": self.nodes, "container_name": self.container_name,
                     "configuration": self.configuration,
                     "containers": self.containers, "active_nodes": self.active_nodes, "standby_nodes": self.standby_nodes,
                     "recovery_nodes": self.recovery_nodes, "fault_node": self.fault_node,
                     "topology": {"training_nodes": self.training_nodes, "gpus_per_node": self.gpus,
                                  "world_size": self.world_size, "spare_count": len(self.nodes) - self.training_nodes},
                     "envcheck_reports": self.reports, "checkpoint": self.checkpoint_evidence,
                     "injection": self.injection, "cleanup_errors": self.cleanup_errors,
                     "collection_errors": self.collection_errors, "final_evidence": self.final_evidence,
                     "functional_outcome": functional_outcome}
            destination = self.evidence.write_result(outcome, category, reason, extra=extra)
            self.result = read_json(destination)
            self.closed = True
            self._persist_ownership()
            return self.result
