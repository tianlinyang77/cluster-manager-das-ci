#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Sourced independently on every training node and by platform.
hcu_st_source_dtk() {
    local restore_u=0 status=0
    case "$-" in *u*) restore_u=1 ;; esac
    set +u
    source /opt/dtk/env.sh || status=$?
    if (( restore_u )); then set -u; fi
    return "$status"
}
hcu_st_source_dtk
unset -f hcu_st_source_dtk
# Put reviewed site RCCL/UCX overrides here if needed; no transport is forced.
