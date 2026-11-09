# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real environment checks; this case starts its own training to verify BUSY."""
import pytest


@pytest.mark.st_case("ENV-001")
def test_environment_checks(st_context):
    st_context.start()
    st_context.check_environment()
