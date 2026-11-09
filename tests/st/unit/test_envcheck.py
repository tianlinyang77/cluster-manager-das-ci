# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline hcu-envcheck contract tests; never execute a remote operation."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lib.envcheck import command, freeze_source, receipt
from lib.evidence import STError


def write_report(tmp_path, name, data):
    folder = tmp_path / "native"
    folder.mkdir(exist_ok=True)
    p = folder / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def platform():
    names = ["a", "b"]
    out = {"schema_version": "2.0", "run": {"execution": {"scope": "container", "categories": ["platform"], "status": "PASS"}}}
    for category in ("node_status", "hardware_devices", "system", "driver_dtk", "software_components",
                     "network_rdma", "resource_state", "network_health", "execution_evidence"):
        payload = {"members": names, "node_count": 2}
        if category == "node_status":
            payload.update(status="READY", reachable=True, checks=[{"check_id": "PAYLOAD_PARSE", "status": "PASS"}])
        if category == "network_health":
            payload.update(findings=[dict(severity="WARN", reason_code="RDMA_USERSPACE_DEVICE_OPEN_FAILED", message="open failed")])
        if category == "execution_evidence":
            payload.update(probe_transport={"returncode": 0, "success": True, "timed_out": False, "error_kind": None})
        out[category] = {"nodes": {"a,b": payload}}
    return out


def test_platform_preserves_raw_report_and_expands_folded_nodes(tmp_path):
    data = platform()
    path = write_report(tmp_path, "cluster-result.json", data)
    original = path.read_bytes()
    result, derived = receipt(tmp_path, "platform", ["a", "b"], 0)
    assert result["status"] == "WARN" and [r["node"] for r in result["results"]] == ["a", "b"]
    assert path.read_bytes() == original
    assert len((derived.parent / "platform-findings.tsv").read_text().splitlines()) == 3


@pytest.mark.parametrize("problem", ["missing_node", "duplicate", "execution", "scope", "transport", "unknown_check", "no_checks", "schema", "exit"])
def test_platform_does_not_waive_failed_or_incomplete_collection(tmp_path, problem):
    data = platform()
    if problem == "missing_node": data["network_health"]["nodes"]["a,b"]["members"] = ["a"]
    if problem == "duplicate": data["node_status"]["nodes"]["a,b"]["members"] = ["a", "a"]
    if problem == "execution": data["run"]["execution"]["status"] = "FAIL"
    if problem == "scope": data["run"]["execution"]["categories"] = ["resource"]
    if problem == "schema": data["schema_version"] = "unexpected"
    if problem == "transport": data["execution_evidence"]["nodes"]["a,b"]["probe_transport"]["success"] = False
    if problem == "unknown_check": data["node_status"]["nodes"]["a,b"]["checks"][0]["status"] = "UNKNOWN"
    if problem == "no_checks": data["node_status"]["nodes"]["a,b"]["checks"] = []
    write_report(tmp_path, "cluster-result.json", data)
    if problem in {"transport", "unknown_check"}:
        result, _ = receipt(tmp_path, "platform", ["a", "b"], 0)
        assert result["status"] == "FAIL"
    else:
        with pytest.raises(STError): receipt(tmp_path, "platform", ["a", "b"], 3 if problem == "exit" else 0)


def test_native_platform_serializer_contract(tmp_path):
    # Exercise the actual product serializer using pure facts, not a hand-written schema.
    repo = Path(os.environ.get("ST_CONTRACT_REPO", Path(__file__).resolve().parents[3]))
    code = '''import json,sys
sys.path.insert(0,sys.argv[1])
from hcu_envcheck.baremetal_cluster import build_node_first_result
report={"policy":{"check_categories":["platform"]},"execution":{"categories":["platform"],"scope":"container","status":"PASS"},
"nodes":[{"node":n,"status":"READY","reachable":True,"checks":[{"check_id":"DRIVER_VERSION","status":"PASS"}],
"devices":[],"environment":{},"findings":[],"probe_transport":{"returncode":0,"success":True,"timed_out":False,"error_kind":None}}
for n in ["a","b"]]}
print(json.dumps(build_node_first_result(report)))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(repo / "hcu-envcheck")],
                            capture_output=True, text=True, encoding="utf-8", timeout=20, check=True)
    write_report(tmp_path, "cluster-result.json", json.loads(result.stdout))
    assert receipt(tmp_path, "platform", ["a", "b"], 0)[0]["status"] == "PASS"


@pytest.mark.parametrize("state,rc", [("IDLE", 0), ("BUSY", 1)])
def test_idle_native_report_and_exit(tmp_path, state, rc):
    decision = {"status": state, "cards": [{"card": 0}]}
    data = {"operation": "idle-check", "status": "PASS" if rc == 0 else "FAIL", "cancelled": False,
            "nodes": [{"node": "a", "decision": decision, "history": [{"decision": decision}], "exit_code": 0 if rc == 0 else 2}]}
    raw = write_report(tmp_path, "report.json", data)
    result, _ = receipt(tmp_path, "idle-check", ["a"], rc)
    assert result["idle_check"]["nodes"][0]["initial_decision"] == decision
    assert "initial_decision" not in json.loads(raw.read_text())["nodes"][0]
    with pytest.raises(STError): receipt(tmp_path, "idle-check", ["a", "b"], rc)


def test_cancelled_idle_cannot_start_training(tmp_path):
    write_report(tmp_path, "report.json", {"operation": "idle-check", "status": "FAIL", "cancelled": True,
                                          "nodes": [{"node": "a", "exit_code": 0, "decision": {"status": "IDLE"}}]})
    with pytest.raises(STError, match="cancelled"): receipt(tmp_path, "idle-check", ["a"], 130)


def test_container_lifecycle_tsv_identity(tmp_path):
    folder = tmp_path / "native"; folder.mkdir()
    with (folder / "results.tsv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=["stage_id", "node", "status", "exit_code", "container_id"], delimiter="\t")
        w.writeheader(); w.writerow(dict(stage_id="01-inspect", node="a", status="PASS", exit_code="0", container_id="a"*64))
    result, _ = receipt(tmp_path, "container-status", ["a"], 0)
    assert result["results"][0]["container_id"] == "a"*64
    with pytest.raises(STError): receipt(tmp_path, "container-status", ["a"], 2)


def test_native_lifecycle_renderer_contract(tmp_path):
    repo = Path(os.environ.get("ST_CONTRACT_REPO", Path(__file__).resolve().parents[3]))
    folder = tmp_path / "native"; state = folder / ".state"; state.mkdir(parents=True)
    (state / "groups.tsv").write_text("group\tnodes\ng1\ta\n")
    fields = ["stage_id", "operation", "task_id", "token", "node", "action", "status", "reason_code",
              "exit_code", "container_id", "image", "detail", "started_epoch", "ended_epoch", "evidence"]
    with (state / "events.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerow(["01-inspect", "container-status", "a/01-inspect", "token", "a", "inspect", "PASS",
                         "OK", "0", "a"*64, "image", "running", "1", "2", "evidence"])
    bash = r"C:\Program Files\Git\bin\bash.exe" if sys.platform == "win32" else shutil.which("bash")
    script = '''set -eu
source "$1/cluster_run/plan.sh"
plan_stages container-status 1 > "$2/.state/stages.tsv"
source "$1/hcu_envcheck/reporting/lifecycle.sh"
NODES=(a)
LIFECYCLE_OUTPUT_DIR=$2
LIFECYCLE_TERMINAL_FD=1
OPERATION=container-status
lifecycle_report
exit "$LIFECYCLE_FINAL_RC"
'''
    subprocess.run([bash, "-c", script, "contract", (repo / "hcu-envcheck").as_posix(), folder.as_posix()],
                   capture_output=True, text=True, timeout=20, check=True)
    assert receipt(tmp_path, "container-status", ["a"], 0)[0]["status"] == "PASS"


def test_command_uses_public_repository_backend_flags(tmp_path):
    hostfile = tmp_path / "hosts";hostfile.write_text("a\nb\n")
    cfg = {"paths": {"control_python": "/python", "env_script": "/env"}, "timeouts": {"command": 120},
           "site": {"image": "image"}, "training": {"python": "python3"}}
    args = command("/repo", "platform", hostfile, tmp_path, cfg, "owned")
    assert str(Path("/repo/hcu-envcheck/bin/hcu-cluster-run")) in args and "--remote-python" not in args
    assert not {"--target-python", "--ssh-port", "--remote-workdir"}.intersection(args)


def test_platform_public_cli_dry_run_accepts_adapter_command(tmp_path):
    repo = Path(os.environ.get("ST_CONTRACT_REPO", Path(__file__).resolve().parents[3]))
    hosts = tmp_path / "hosts"; hosts.write_text("a\nb\n")
    cfg = {"paths": {"control_python": Path(sys.executable).as_posix(), "env_script": "/env"},
           "timeouts": {"command": 120}, "site": {"image": "image"}}
    args = command(repo, "platform", hosts, tmp_path / "output", cfg, "owned")
    args = [Path(a).as_posix() if "\\" in a else a for a in args]
    args[0] = r"C:\Program Files\Git\bin\bash.exe" if sys.platform == "win32" else shutil.which("bash")
    completed = subprocess.run(args + ["--container-workdir", "/case", "--dry-run"],
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DRY_RUN scenario=per-node-container operation=platform nodes=2" in completed.stdout


@pytest.fixture
def staged_platform(tmp_path):
    repo = Path(os.environ.get("ST_CONTRACT_REPO", Path(__file__).resolve().parents[3]))
    folder = tmp_path / "native" / ".state"; folder.mkdir(parents=True)
    code = r'''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from hcu_envcheck.reporting.aggregate import Run
root=Path(sys.argv[2])
(root/'control.tsv').write_text('operation\tplatform\nscenario\tper-node-container\nstatus\tFINISHED\ngroup_count\t1\n')
(root/'groups.tsv').write_text('group\tnodes\ngroup-001\ta,b\n')
(root/'stages.tsv').write_text('stage_id\titem\titeration\tlabel\n01-platform\tplatform\t1\tplatform\n')
task=root/'stages/01-platform/groups/group-001';task.mkdir(parents=True)
(root/'prepared/group-001').mkdir(parents=True)
header='status\texit_code\tcleanup\tstarted_epoch\tended_epoch\treason_code\tlog\n'
row='PASS\t0\tCONFIRMED\t1\t2\tOK\tgroup.log\n'
(task/'task.tsv').write_text(header+row)
(root/'prepared/group-001/task.tsv').write_text(header+row)
(task/'node-results.tsv').write_text('node\texit_code\tcleanup\tlog\na\t0\tCONFIRMED\ta.log\nb\t0\tCONFIRMED\tb.log\n')
payload={'schema_version':'1.0','system':{},'driver':{'device_nodes':['/dev/kfd']},
         'torch':{'importable':True,'hcu_available':True,'hip_version':'6.0'}}
for node in ['a','b']:(task/(node+'.log')).write_text(json.dumps(payload)+'\n')
(root/'report.json').write_text(json.dumps(Run(root).snapshot(final=True)))
'''
    subprocess.run([sys.executable, "-B", "-c", code, str(repo / "hcu-envcheck"), str(folder)],
                   capture_output=True, text=True, encoding="utf-8", check=True, timeout=30)
    path = folder / "report.json"
    return repo, path, json.loads(path.read_text())


def test_staged_platform_native_aggregate_and_original_warning_reasons(tmp_path, staged_platform):
    repo, path, data = staged_platform
    original = path.read_bytes()
    assert data["status"] == "WARN"
    report, derived = receipt(tmp_path, "platform", ["a", "b"], 2, repo=repo)
    assert report["status"] == "WARN"
    assert path.read_bytes() == original
    assert "HIPCC_NOT_AVAILABLE" in derived.with_name("platform-findings.tsv").read_text()
    assert [r["node"] for r in report["results"]] == ["a", "b"]


@pytest.mark.parametrize("problem", ["cleanup", "preflight", "nonfinal", "scope", "coverage", "node", "exit", "payload", "raw_disagreement"])
def test_staged_platform_rejects_incomplete_execution_and_wrong_evidence(tmp_path, staged_platform, problem):
    repo, path, data = staged_platform
    if problem == "cleanup": data["results"][0]["cleanup"] = "UNCONFIRMED"
    if problem == "preflight": data["preflight"][0]["status"] = "FAIL"
    if problem == "nonfinal": data["final"] = False
    if problem == "scope": data["control"]["scenario"] = "baremetal"
    if problem == "coverage": data["executed"] = 0
    if problem == "node": data["results"][0]["nodes"] = "a"
    if problem == "payload": data["results"][0]["checks"][0]["status"] = "FAIL"
    if problem == "raw_disagreement": data["results"][0]["checks"][1]["detail"] = "wrong fact"
    path.write_text(json.dumps(data))
    with pytest.raises(STError):
        receipt(tmp_path, "platform", ["a", "b"], 0 if problem == "exit" else 2, repo=repo)


def test_native_backend_snapshot_is_importable_from_shared_case(tmp_path):
    repo = Path(os.environ.get("ST_CONTRACT_REPO", Path(__file__).resolve().parents[3]))
    frozen = freeze_source(repo, tmp_path / "case" / "envcheck-source")
    manifest = json.loads((frozen.parent / "envcheck-source.json").read_text())
    assert not (frozen / "third-party").exists()
    assert "hcu_envcheck/payloads/task_guard.sh" in manifest["files"]
    assert "cluster_run/worker.sh" in manifest["files"]
    assert not any("__pycache__" in p for p in manifest["files"])
    import hashlib
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((repo / "hcu-envcheck" / relative).read_bytes()).hexdigest() == digest
    result = subprocess.run([sys.executable, "-B", "-c",
                             "from hcu_envcheck.reporting.aggregate import Run; print('reporter import OK')"],
                            cwd=frozen, env=dict(os.environ, PYTHONPATH=str(frozen)),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    with pytest.raises(FileExistsError): freeze_source(repo, frozen)
