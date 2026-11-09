# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real ST is opt-in. Collection without --site never touches a cluster."""
from __future__ import annotations

from pathlib import Path
import sys
import pytest

ST_ROOT = Path(__file__).resolve().parent
if str(ST_ROOT) not in sys.path:
    sys.path.insert(0, str(ST_ROOT))

from run_st import load_site


def pytest_addoption(parser):
    parser.getgroup("hcu-st").addoption("--site", default=None, help="explicit phase-one ST site TOML")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, "st_report_" + report.when, report)
    if call.excinfo is not None:
        setattr(item, "st_exception_" + report.when, call.excinfo.value)


@pytest.fixture
def st_context(request, pytestconfig):
    site = pytestconfig.getoption("--site")
    if not site:
        pytest.skip("real cluster not executed: provide a ready --site through run_st.py")
    try:
        cfg = load_site(site, execution=True)
    except (OSError, ValueError) as error:
        pytest.fail(f"ST configuration not ready: {error}", pytrace=False)
    marker = request.node.get_closest_marker("st_case")
    if marker is None or len(marker.args) != 1:
        pytest.fail("ST test needs one stable st_case ID", pytrace=False)
    from lib.clients import CaseContext
    context = CaseContext(cfg, str(marker.args[0]))
    try:
        yield context
    finally:
        report = getattr(request.node, "st_report_call", None)
        error = getattr(request.node, "st_exception_call", None)
        if report is None or report.skipped:
            outcome, category, reason = "NOT_EXECUTED", "unexecuted", str(error or "test call not executed")
        elif report.failed:
            outcome = "FAIL"
            category = getattr(error, "category", "test-framework")
            reason = f"{getattr(error, 'code', type(error).__name__)}: {getattr(error, 'detail', str(error))}"
        else:
            outcome, category, reason = "PASS", "", ""
        result = context.close(outcome=outcome, category=category, reason=reason)
        if outcome == "PASS" and result.get("outcome") != "PASS":
            pytest.fail(f"ST teardown did not complete: {result.get('reason', 'see result.json')}", pytrace=False)
