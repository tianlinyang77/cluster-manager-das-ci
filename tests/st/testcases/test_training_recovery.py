# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""One worker exit followed by actual backup-node checkpoint recovery."""
import pytest


@pytest.mark.st_case("REL-010")
def test_worker_checkpoint_recovery(st_context):
    st_context.start()
    st_context.check_environment()
    st_context.recover()
