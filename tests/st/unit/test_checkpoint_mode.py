# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint mode regressions; no remote commands, framework or training."""
from pathlib import Path, PurePosixPath

import pytest

import run_st
from lib import clients
from lib.clients import CaseContext
from lib.evidence import STError
from lib.training import parse_training_log
from unit.site_fixture import site_config


ST=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("local,global_,valid", [(0,10,True),(5,10,True),(-1,10,False),
                                                (True,10,False),(0,0,False),(5,7,False),(5,5,False)])
def test_real_run_checkpoint_interval_validation(monkeypatch, local, global_, valid):
    cfg=site_config()
    cfg["training"].update(ready=True,script="/repo/train_st.sh",local_interval=local,global_interval=global_)
    cfg["paths"].update(repo_root="/repo",shared_root="/shared/hcu-st/cases",control_python="/python",
                         bridge_root="/bridge",env_script="/env.sh")
    cfg["timeouts"]={name:30 if name!="poll" else 1 for name in run_st.TIMEOUTS}
    class ExistingPath(PurePosixPath):
        def resolve(self): return self
        def is_file(self): return True
    monkeypatch.setattr(run_st,"Path",ExistingPath)
    monkeypatch.setattr(run_st.sys,"platform","linux")
    monkeypatch.setattr(run_st.os,"access",lambda *a:True)
    if valid:
        run_st.validate_execution(cfg,allow_missing_bridge=True)
    else:
        with pytest.raises(ValueError,match="local_interval"):
            run_st.validate_execution(cfg,allow_missing_bridge=True)


def context(tmp_path, local_interval):
    cfg=site_config()
    cfg["site"]["nodes"]=["a","b"]
    cfg["training"].update(nodes=1,gpus=8,local_interval=local_interval)
    cfg["paths"]["shared_root"]=str(tmp_path / "cases")
    cfg["timeouts"].update(command=30,checkpoint=30,recovery=30,total=120)
    ctx=CaseContext(cfg,"REL-010")
    ctx.active_attempt={"attempt_id":"old","slots":["a"]}
    return ctx


@pytest.mark.parametrize("interval,local_log,local_file,global_log,global_file,expected", [
    (0,False,False,True,True,True),
    (0,True,True,True,True,True),
    (0,False,False,False,True,False),
    (0,False,False,True,False,False),
    (5,False,False,True,True,False),
    (5,True,False,True,True,False),
    (5,True,True,True,True,True),
])
def test_waits_only_for_enabled_checkpoint_data(tmp_path, interval, local_log, local_file, global_log, global_file, expected):
    ctx=context(tmp_path,interval)
    lines=[]
    if local_log:lines.append("successfully saved local checkpoint from iteration 5")
    if global_log:lines.extend(["saving checkpoint at iteration 10 to /global in torch_dist format",
                               "successfully saved checkpoint from iteration 10"])
    parsed=parse_training_log("\n".join(lines))
    ctx._attempts=lambda:[(ctx.active_attempt,parsed)]
    for kind,exists in (("local",local_file),("global",global_file)):
        if exists:
            folder=ctx.case_root / "checkpoints" / kind;folder.mkdir(parents=True)
            (folder / "weights.distcp").write_bytes(b"checkpoint")
    class Observed(Exception):pass
    def wait(stage,budget,observe):
        assert observe() is expected
        assert stage == ("both checkpoint saves" if interval else "torch_dist checkpoint save")
        raise Observed
    ctx._wait=wait
    with pytest.raises(Observed):ctx.recover()
    if expected:
        assert ctx.checkpoint_evidence["local_status"] == ("PASS" if interval else "NOT_EXECUTED")
        names={row["name"] for row in ctx.evidence.checks}
        assert "checkpoint_torch_dist_saved" in names
        assert ("checkpoint_local_saved" in names) is bool(interval)
        assert ("local" in ctx.checkpoint_evidence["files"]) is bool(interval)


@pytest.mark.parametrize("load_kind", ["global","local"])
def test_global_only_recovery_requires_global_load_and_progress(tmp_path,monkeypatch,load_kind):
    ctx=context(tmp_path,0)
    ctx.active_nodes=["a"];ctx.standby_nodes=["b"]
    ctx.containers={node:{"id":"a"*64 if node=="a" else "b"*64} for node in ctx.nodes}
    folder=ctx.case_root / "checkpoints/global";folder.mkdir(parents=True)
    (folder / "weights.distcp").write_bytes(b"checkpoint")
    before=parse_training_log("saving checkpoint at iteration 10 to /global in torch_dist format\n"
                              "successfully saved checkpoint from iteration 10")
    load=ctx.case_root / "checkpoints" / load_kind
    after=parse_training_log(f"successfully loaded checkpoint from {load} at iteration 10\n"
                             "iteration 11 / 2000 elapsed time per iteration 1 ms")
    injected=[]
    def kill(*a,**kw):
        injected.append(True)
        return {"signal":"SIGKILL"}
    monkeypatch.setattr(clients,"kill_worker",kill)
    monkeypatch.setattr(clients,"read_json",lambda _:{"run_state":"RUNNING"})
    ctx._workers=lambda node:{"workers":[{"local_rank":rank,"max_restarts":"0"} for rank in range(8)],"owned_processes":[]}
    ctx._allocation=lambda *a:True
    ctx._inspect=lambda node:ctx.containers[node]
    ctx._attempts=lambda:[({"attempt_id":"new","slots":["b"]},after)] if injected else [(ctx.active_attempt,before)]
    ctx._wait=lambda stage,budget,observe:observe() or pytest.fail(f"stage did not complete: {stage}")
    if load_kind=="local":
        with pytest.raises(STError,match="must load a checkpoint under"):ctx.recover()
        assert any(row["name"]=="checkpoint_load_scope" and not row["passed"] for row in ctx.evidence.checks)
    else:
        ctx.recover()
        assert ctx.recovery_nodes==["b"]
        assert ctx.checkpoint_evidence["recovery"]["last_iteration"]==11
    assert len(injected)==1
