# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Adapt repository hcu-envcheck receipts without changing the product reports."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lib.evidence import STError


def _need(condition, detail):
    if not condition:
        raise STError("test-framework", "ENVCHECK_RECEIPT_INVALID", detail)


def command(repo, operation, hosts, output, config, container):
    """Use only the public arguments supported by hcu-envcheck."""
    argv = ["bash", str(Path(repo) / "hcu-envcheck/bin/hcu-cluster-run"),
            "--controller-python", config["paths"]["control_python"], "per-node-container", operation,
            "--hostfile", str(hosts), "--output-dir", str(output),
            "--concurrency", str(len(Path(hosts).read_text(encoding="utf-8").splitlines())),
            "--timeout", str(max(40, int(config["timeouts"]["command"])))]
    if operation != "idle-check":
        argv += ["--container", container, "--image", config["site"]["image"]]
    if operation == "platform":
        argv += ["--env-script", config["paths"]["env_script"]]
    return argv


def freeze_source(repo, destination):
    """Expose the tested public backend inside the case's shared mount."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    original = Path(repo) / "hcu-envcheck"
    for name in ("hcu_envcheck", "cluster_run"):
        shutil.copytree(original / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    manifest = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination.parent / "envcheck-source.json").write_text(
        json.dumps(dict(source=str(original), snapshot=str(destination), files=manifest), indent=2) + "\n", encoding="utf-8")
    return destination


def _one(output, pattern):
    paths = list(Path(output).glob(pattern))
    _need(len(paths) == 1, f"expected one {pattern}, found {len(paths)} in {output}")
    return paths[0]


def _coverage(rows, nodes):
    _need(len(rows) == len(nodes) and {r.get("node") for r in rows} == set(nodes),
          "receipt must cover exactly the requested nodes, without duplicates")


def _groups(data, category, nodes):
    result = {}
    for group in data.get(category, {}).get("nodes", {}).values():
        members = group.get("members")
        _need(isinstance(members, list) and bool(members) and group.get("node_count") == len(members),
              f"invalid {category} members")
        for node in members:
            _need(node in nodes and node not in result, f"unexpected/duplicate {category} node {node}")
            result[node] = group
    _need(set(result) == set(nodes), f"incomplete {category} evidence")
    return result



def _staged_platform(data, nodes, returncode, repo):
    """Adapt the public staged CLI, keeping execution separate from health."""
    _need(data.get("schema_version") == 1 and data.get("protocol_version") == 3,
          "unsupported staged platform protocol")
    control = data.get("control", {})
    _need(control.get("operation") == "platform" and control.get("scenario") == "per-node-container",
          "platform report belongs to a different operation/scope")
    _need(data.get("final") is True and control.get("status") == "FINISHED",
          "platform report is not final")
    groups = data.get("groups", [])
    covered = [node for group in groups for node in group.get("nodes", "").split(",")]
    _need(len(covered) == len(nodes) and set(covered) == set(nodes), "incomplete/duplicate platform groups")
    results = data.get("results", [])
    _need(len(data.get("stages", [])) == 1 and len(results) == len(groups) and bool(results),
          "incomplete platform stages/results")
    _need({r.get("group") for r in results} == {g.get("group") for g in groups}, "platform group mismatch")
    _need(all(data.get(k) == len(results) for k in ("planned", "executed", "completed")),
          "platform execution coverage incomplete")
    expected_rc = {"PASS": 0, "WARN": 2, "FAIL": 1}.get(data.get("status"))
    _need(returncode == expected_rc, "platform report/CLI exit disagreement")
    preflight = data.get("preflight", [])
    execution_ok = (len(preflight) == len(groups)
                    and {r.get("group") for r in preflight} == {g.get("group") for g in groups}
                    and all(r.get("status") == "PASS" and str(r.get("exit_code")) == "0"
                            and r.get("cleanup") == "CONFIRMED" for r in preflight)
                    and all(r.get("operation") == "platform" and r.get("dispatched") is True
                            and str(r.get("exit_code")) == "0" and r.get("cleanup") == "CONFIRMED"
                            and r.get("reason_code") in {"PLATFORM_HEALTH_PASS", "PLATFORM_HEALTH_WARN", "PLATFORM_HEALTH_FAILED"}
                            for r in results))
    if not execution_ok:
        raise STError("unclassified", "ENVCHECK_EXECUTION_FAILED", "platform collection/preflight/cleanup failed; inspect original report")
    per_node = {}
    findings = []
    for result in results:
        group_nodes = result.get("nodes", "").split(",")
        expected_group = next(g["nodes"] for g in groups if g["group"] == result["group"])
        _need(result["nodes"] == expected_group, "platform result node mismatch")
        checks = result.get("checks", [])
        _need({c.get("node") for c in checks} == set(group_nodes), "incomplete platform node checks")
        for node in group_nodes:
            current = [c for c in checks if c["node"] == node]
            _need(sum(c.get("check") == "PAYLOAD_PARSE" and c.get("status") == "PASS" for c in current) == 1,
                  f"{node}: platform payload collection missing/invalid")
            health = ("FAIL" if any(c.get("status") not in {"PASS", "WARN", "NOT_APPLICABLE"} for c in current)
                      else "WARN" if any(c.get("status") == "WARN" for c in current) else "PASS")
            per_node[node] = dict(node=node, status=health, exit_code="0", transport_exit_code="0")
            if health == "WARN":
                # The staged summary omits detailed reason codes. Recover them
                # from the same native rule/profile and checksummed logical log.
                code = """import dataclasses,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from hcu_envcheck.reporting.parsers import _platform_payload, PLATFORM_PROFILE
from hcu_envcheck.environment import evaluate_environment
checks=json.loads(sys.argv[2])
path=next(c.get('raw_evidence',c['evidence']) for c in checks if c['check']=='PAYLOAD_PARSE')
payload=_platform_payload(Path(path))
if payload is None: raise ValueError('platform raw payload absent')
findings,_,rules=evaluate_environment(payload,**PLATFORM_PROFILE)
actual=[(c['check'],c['status'],c['detail']) for c in checks if c['check']!='PAYLOAD_PARSE']
assert actual==[(c['check_id'],c['status'],c['message']) for c in rules], 'platform raw/report checks disagree'
print(json.dumps([dataclasses.asdict(f) for f in findings]))
"""
                native = subprocess.run([sys.executable, "-B", "-c", code, str(Path(repo) / "hcu-envcheck"),
                                         json.dumps(current)], capture_output=True, text=True, encoding="utf-8", timeout=30)
                _need(native.returncode == 0, f"{node}: platform reason evidence invalid: {native.stderr[-1000:]}")
                details = json.loads(native.stdout)
                _need(bool(details), f"{node}: WARN without reason findings")
                findings.extend(dict(node=node, severity=f["severity"], reason_code=f["reason_code"], message=f["message"])
                                for f in details if f["severity"] != "INFO")
    _need(set(per_node) == set(nodes), "platform checks do not cover requested nodes")
    status = "FAIL" if any(r["status"] == "FAIL" for r in per_node.values()) else (
        "WARN" if any(r["status"] == "WARN" for r in per_node.values()) else "PASS")
    _need(data.get("status") == status, "platform aggregate/health disagreement")
    return dict(status=status, results=[per_node[n] for n in nodes]), findings


def receipt(output, operation, nodes, returncode, *, repo=None):
    """Return a derived ST view; raw JSON/TSV remain untouched beside it."""
    output = Path(output)
    findings = []
    if operation == "idle-check":
        source = _one(output, "*/report.json")
        data = json.loads(source.read_text(encoding="utf-8"))
        _need(data.get("operation") == operation and isinstance(data.get("nodes"), list), "invalid idle report")
        records = data["nodes"]
        _coverage(records, nodes)
        _need(not data.get("cancelled"), "idle sampling was cancelled")
        _need(data.get("status") == ("PASS" if all(r.get("exit_code") == 0 for r in records) else "FAIL"),
              "idle report status disagrees with node exit codes")
        for record in records:
            if record.get("decision", {}).get("status") == "IDLE":
                _need(record.get("exit_code") == 0 and not record.get("error"), "IDLE row has failed execution")
            history = record.get("history") or []
            if history:
                record["initial_decision"] = history[0].get("decision", {})
        expected = 130 if data.get("cancelled") else 0 if data.get("status") == "PASS" else 1
        _need(returncode == expected, "idle report/CLI exit disagreement")
        rows = [dict(node=r["node"], status="PASS" if r.get("exit_code") == 0 else "FAIL") for r in records]
        view = dict(status=data["status"], idle_check={"nodes": records}, results=rows)
    elif operation == "container-status":
        source = _one(output, "*/results.tsv")
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            _need({"stage_id", "node", "status", "exit_code", "container_id"}.issubset(reader.fieldnames or []),
                  "invalid container status columns")
            rows = list(reader)
        _coverage(rows, nodes)
        _need(all(r["stage_id"] == "01-inspect" for r in rows), "unexpected container lifecycle stage")
        status = "FAIL" if any(r["status"] not in {"PASS", "WARN"} for r in rows) else (
            "WARN" if any(r["status"] == "WARN" for r in rows) else "PASS")
        _need(returncode == {"PASS": 0, "WARN": 2, "FAIL": 1}[status], "container report/CLI exit disagreement")
        if status == "PASS":
            _need(all(r["exit_code"] == "0" for r in rows), "PASS container row has nonzero exit")
        view = dict(status=status, results=rows)
    elif operation == "platform":
        if list(output.glob("*/.state/report.json")) or list(output.glob("*/report.json")):
            pattern = "*/.state/report.json" if list(output.glob("*/.state/report.json")) else "*/report.json"
            source = _one(output, pattern)
            data = json.loads(source.read_text(encoding="utf-8"))
            repo = repo or os.environ.get("ST_CONTRACT_REPO") or Path(__file__).resolve().parents[3]
            view, findings = _staged_platform(data, nodes, returncode, repo)
        else:
            source = _one(output, "*/cluster-result.json")
            data = json.loads(source.read_text(encoding="utf-8"))
            _need(data.get("schema_version") == "2.0", "unsupported platform schema")
            execution = data.get("run", {}).get("execution", {})
            _need(execution.get("categories") == ["platform"] and execution.get("scope") == "container",
                  "platform report belongs to a different operation/scope")
            states = _groups(data, "node_status", nodes)
            transport = _groups(data, "execution_evidence", nodes)
            per_node = {node: {"checks": [], "findings": []} for node in nodes}
            for category in ("node_status", "hardware_devices", "system", "driver_dtk", "software_components",
                             "network_rdma", "resource_state", "network_health", "execution_evidence"):
                for node, group in _groups(data, category, nodes).items():
                    per_node[node]["checks"].extend(group.get("checks", []))
                    per_node[node]["findings"].extend(group.get("findings", []))
            rows = []
            for node in nodes:
                state, info = states[node], per_node[node]
                probe = transport[node].get("probe_transport", {})
                sampled = (probe.get("returncode") == 0 and probe.get("success") is True
                           and not probe.get("timed_out") and not probe.get("error_kind")
                           and state.get("reachable") is True)
                checks = info["checks"]
                _need(bool(checks), f"{node}: platform checks absent")
                health = "PASS"
                if (not sampled or state.get("status") != "READY"
                        or any(c.get("status") not in {"PASS", "WARN", "NOT_APPLICABLE"} for c in checks)
                        or any(f.get("severity") not in {"WARN", "INFO"} for f in info["findings"])):
                    health = "FAIL"
                elif any(f.get("severity") == "WARN" for f in info["findings"]):
                    health = "WARN"
                elif any(c.get("status") == "WARN" for c in checks):
                    health = "FAIL"  # A WARN without its reason cannot be waived.
                for f in info["findings"]:
                    if f.get("severity") != "INFO":
                        findings.append(dict(node=node, severity=f.get("severity", "UNKNOWN"),
                                             reason_code=f.get("reason_code", ""), message=f.get("message", "")))
                rows.append(dict(node=node, status=health, exit_code="0" if sampled else "1",
                                 transport_exit_code=str(probe.get("returncode", "missing"))))
            # Health findings do not determine this product's CLI exit; execution does.
            if returncode != 0 or execution.get("status") != "PASS":
                raise STError("unclassified", "ENVCHECK_EXECUTION_FAILED",
                              "platform execution/cleanup failed; inspect original receipt and command.log")
            view = dict(status="FAIL" if any(r["status"] == "FAIL" for r in rows) else
                        "WARN" if any(r["status"] == "WARN" for r in rows) else "PASS", results=rows)
    else:
        raise ValueError(f"unsupported ST operation: {operation}")
    view.update(operation=operation, exit_code=returncode, source_report=str(source))
    normalized = output / "st-view"
    normalized.mkdir(exist_ok=False)
    path = normalized / "results.json"
    path.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if operation == "platform":
        with (normalized / "platform-findings.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["node", "severity", "reason_code", "message"], delimiter="\t")
            writer.writeheader()
            writer.writerows(findings)
    return view, path
