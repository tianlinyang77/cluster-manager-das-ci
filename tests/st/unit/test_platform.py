# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit health waivers must not hide failed collection or unrelated findings."""
import copy
import csv
from pathlib import Path

import pytest

import run_st
from lib.clients import CaseContext
from lib.evidence import Evidence, STError
from unit.site_fixture import site_config


CODES = ["RDMA_USERSPACE_DEVICE_OPEN_FAILED", "IB_ENDPOINT_EVIDENCE_MISSING"]


def context(tmp_path, *, allowed=True):
    ctx = object.__new__(CaseContext)
    ctx.config = {"platform": {"ignored_reason_codes": CODES if allowed else []}}
    ctx.nodes = ["a", "b"]
    ctx.reports = [str(tmp_path / "results.json")]
    ctx.evidence = Evidence(tmp_path, "REL-010")
    ctx.configuration = {}
    return ctx


def report():
    return {"status": "WARN", "results": [dict(node=node, status="WARN", exit_code="0",
                                               transport_exit_code="0") for node in ["a", "b"]]}


def findings(tmp_path, *, code=None, severity="WARN", nodes=("a", "b")):
    with (tmp_path / "platform-findings.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["node", "severity", "reason_code", "message"], delimiter="\t")
        writer.writeheader()
        for node in nodes:
            for item in CODES:
                writer.writerow(dict(node=node, severity=severity, reason_code=code or item, message="raw warning"))


def test_explicit_waiver_retains_original_report_and_records_findings(tmp_path):
    ctx = context(tmp_path)
    original = report()
    snapshot = copy.deepcopy(original)
    findings(tmp_path)
    ctx._check_platform(original)
    assert original == snapshot
    policy = ctx.configuration["platform_policy"]
    assert policy["reported_status"] == "WARN" and len(policy["ignored_findings"]) == 4
    result = ctx.evidence.write_result("PASS", extra={"configuration": ctx.configuration})
    assert '"reported_status": "WARN"' in result.read_text(encoding="utf-8")


@pytest.mark.parametrize("problem", ["not_opted_in", "unrelated", "fail", "unknown", "missing",
                                     "empty", "partial", "unexpected_node", "bad_columns",
                                     "transport_failed", "payload_failed", "overall_failed"])
def test_waiver_does_not_hide_other_problems(tmp_path, problem):
    ctx = context(tmp_path, allowed=problem != "not_opted_in")
    value = report()
    findings(tmp_path, code="UNRELATED_WARN" if problem == "unrelated" else None,
             severity={"fail": "FAIL", "unknown": "UNKNOWN"}.get(problem, "WARN"),
             nodes={"partial": ("a",), "unexpected_node": ("a", "other")}.get(problem, ("a", "b")))
    path = tmp_path / "platform-findings.tsv"
    if problem == "missing":
        path.unlink()
    if problem == "empty":
        findings(tmp_path, nodes=())
    if problem == "bad_columns":
        path.write_text("broken\n", encoding="utf-8")
    if problem == "transport_failed":
        value["results"][0]["transport_exit_code"] = "255"
    if problem == "payload_failed":
        value["results"][0]["exit_code"] = "1"
    if problem == "overall_failed":
        value["status"] = "FAIL"
    with pytest.raises(STError, match="PLATFORM_NOT_READY"):
        ctx._check_platform(value)
    assert ctx.configuration["platform_policy"]["ignored_findings"] == []


def test_pass_needs_no_waiver_file(tmp_path):
    ctx = context(tmp_path, allowed=False)
    ctx._check_platform({"status": "PASS"})


@pytest.mark.parametrize("value", [True, "ANY", ["*"], ["lowercase"], ["A", "A"], [1]])
def test_invalid_waiver_configuration_is_rejected(tmp_path, value):
    from prepare_dependencies import dump_toml
    config = site_config()
    config["platform"]["ignored_reason_codes"] = value
    site = tmp_path / "site.toml"
    site.write_text(dump_toml(config), encoding="utf-8")
    with pytest.raises(ValueError, match="platform.ignored_reason_codes"):
        run_st.load_site(site)
