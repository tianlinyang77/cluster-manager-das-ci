# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline fixtures derived from the maintained ST template; no site inventory."""
from pathlib import Path
import tomllib

from prepare_dependencies import dump_toml


def site_config():
    st=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((st / "config/site.ci.toml").read_text(encoding="utf-8"))
    cfg["site"]["nodes"]=["node1","node2","node3","node4"]
    cfg["training"].update(ready=False,script="/workspace/cluster-manager-das/tests/st/payloads/train_st.sh")
    cfg["training"]["env"]={}
    cfg["paths"].update(repo_root="/workspace/cluster-manager-das",shared_root="/shared/hcu-st/cases",
                         control_python="/opt/hcu-st/python",bridge_root="",env_script="")
    cfg["reliability"]["env"]={}
    return cfg


def site_file(path):
    path=Path(path)
    path.write_text(dump_toml(site_config()),encoding="utf-8")
    return path
