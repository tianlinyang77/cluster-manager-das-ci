# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Local configuration, lifecycle identity, failure and artifact boundaries."""
import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
import sys

import pytest

import ci_entry as ci
from run_st import load_site


def render(tmp_path, text="node1\nnode2\nnode3\n", key="123-1"):
    hosts = tmp_path / "allocated";hosts.write_text(text, encoding="utf-8")
    return ci.render(ci.ST / "config/site.ci.toml", hosts, sys.executable, tmp_path / "shared space", key)


@pytest.mark.parametrize("text", ["", "# empty\n", "a\na\n", "a slots=4\nb\nc", "a[1-3]", "user@host", "a;touch file", "-bad"])
def test_bad_hostfile_is_rejected_before_creating_run(tmp_path, text):
    with pytest.raises(ValueError): render(tmp_path, text)
    assert not (tmp_path / "shared space/ci-runs").exists()


def test_insufficient_pool_rejected_before_side_effects(tmp_path):
    with pytest.raises(ValueError, match="spare"): render(tmp_path, "a\n")
    assert not (tmp_path / "shared space/ci-runs").exists()


@pytest.mark.parametrize("key", ["../escape", "x/y", "", "a.b"])
def test_run_key_rejects_traversal(tmp_path, key):
    with pytest.raises(ValueError): render(tmp_path, key=key)


def test_generated_config_freezes_hosts_and_quotes_mount_paths(tmp_path):
    run, cfg = render(tmp_path, "# allocated only\nnode1\nnode2\n")
    loaded = load_site(run / "site.toml")
    assert loaded["site"]["nodes"] == ["node1", "node2"]
    assert loaded["training"]["nodes"] == 1 and loaded["training"]["gpus"] == 8
    assert not loaded["site"]["release_nodes"]
    assert "__" not in (run / "site.toml").read_text()
    assert loaded["paths"]["env_script"] == loaded["training"]["env"]["TRAIN_ENV_SCRIPT"]
    args = shlex.split(cfg["reliability"]["env"]["DOCKER_RUN_ARGS"])
    assert args[args.index("-v") + 1] == "/dev/shm:/dev/shm"
    (tmp_path / "allocated").write_text("other\n")
    assert (run / "pool.hostfile").read_text() == "node1\nnode2\n"
    with pytest.raises(FileExistsError): render(tmp_path)


def registry(tmp_path, *, registered=True):
    run = tmp_path / "ci-run";case = run / "cases/hcust20261106000000abcdef1234-rel-010"
    case.mkdir(parents=True)
    rid = "hcust20261106000000abcdef1234"
    (case / "ST_RUN_ID").write_text(rid)
    ci.write_json(run / "ci-run.json", {"run_root": str(run.resolve()), "nodes": ["node1"]})
    item = dict(id="a"*64, image_id="sha256:"+"b"*64, owner_verified=True)
    data = dict(run_id=rid, case_root=str(case.resolve()), container_name=case.name, nodes=["node1"],
                absent_before_start=["node1"], containers={"node1": item} if registered else {}, closed=True,
                manager={"pid": 111, "starttime": "1", "pgid": 111} if not registered else None)
    ci.write_json(case / "ownership.json", data)
    return run,case,data


@pytest.mark.parametrize("problem", ["cid", "marker", "case_root", "owner", "nodes", "preexisting", "manager_launch"])
def test_cleanup_refuses_unverified_resources(tmp_path, monkeypatch, problem):
    run,case,data = registry(tmp_path)
    if problem == "cid": data["containers"]["node1"]["id"] = "unsafe"
    if problem == "marker": (case / "ST_RUN_ID").write_text("another task")
    if problem == "case_root": data["case_root"] = str(tmp_path / "elsewhere")
    if problem == "owner": data["containers"]["node1"]["owner_verified"] = False
    if problem == "nodes": data["nodes"] = ["other"]
    if problem == "preexisting": data["absent_before_start"] = []
    if problem == "manager_launch": data["manager_launching"] = True
    ci.write_json(case / "ownership.json", data)
    calls=[]
    monkeypatch.setattr(ci,"ssh",lambda *a,**kw:calls.append(a))
    assert not ci.cleanup(run)
    assert not calls


def test_cleanup_targets_full_registered_id_with_marker_checks(tmp_path, monkeypatch):
    run,case,data = registry(tmp_path)
    calls=[]
    monkeypatch.setattr(ci,"ssh",lambda *a,**kw:calls.append(a))
    assert ci.cleanup(run)
    assert len(calls)==1 and calls[0][0]=="node1" and calls[0][2][0]=="a"*64
    assert "docker rm -f --" in calls[0][1] and "ST_RUN_ID" in calls[0][1]


def test_cleanup_adopts_only_marker_checked_startup_container(tmp_path, monkeypatch):
    run,case,data = registry(tmp_path, registered=False)
    calls=[]
    def ssh(*args,**kw):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps({"id":"a"*64,"image_id":"sha256:"+"b"*64,
                                                  "name":"/"+case.name,"pid_mode":""}))
    monkeypatch.setattr(ci,"ssh",ssh)
    stopped=[]
    monkeypatch.setattr(ci,"_stop_manager",lambda *a:stopped.append(a))
    assert ci.cleanup(run) and len(calls)==2
    assert stopped == [(data["manager"], data["run_id"])]
    assert calls[1][2][0]=="a"*64 and "ST_RUN_ID" in calls[1][1]


def test_artifacts_exclude_weights_and_external_symlinks(tmp_path):
    run = tmp_path / "run";run.mkdir()
    (run / "entry.log").write_text("evidence")
    checkpoints=run / "cases/case/checkpoints/local";checkpoints.mkdir(parents=True)
    (checkpoints / "weight.pt").write_bytes(b"weights")
    (checkpoints / "metadata.json").write_text("must not copy checkpoint content")
    out=tmp_path / "artifacts"
    ci.collect(run,out)
    assert (out / "entry.log").read_text()=="evidence"
    assert not (out / "cases").exists()
    assert len(json.loads((out / "checkpoint-files.json").read_text()))==2
    with pytest.raises(ValueError): ci.collect(run,run / "nested")


def test_run_preserves_failure_status_and_still_collects(tmp_path, monkeypatch):
    hosts=tmp_path / "hosts";hosts.write_text("a\nb\nc\n")
    monkeypatch.setattr(ci,"preflight",lambda *a:None)
    def fail(*a):raise ci.subprocess.CalledProcessError(2,"real ST failed")
    monkeypatch.setattr(ci,"execute",fail)
    out=tmp_path / "artifacts"
    code=ci.main(["run","--hostfile",str(hosts),"--control-python",sys.executable,
                  "--shared-root",str(tmp_path / "shared"),"--run-key","1-1","--artifacts",str(out)])
    assert code==2 and json.loads((out / "ci-result.json").read_text())["exit_code"]==2


@pytest.mark.parametrize("training,gpus,pool",[(2,4,3),(1,8,2),(2,4,4),(2,3,3)])
def test_configurable_pool_and_training_topology(tmp_path,training,gpus,pool):
    cfg=ci.tomllib.loads((ci.ST / "config/site.ci.toml").read_text())
    cfg["training"].update(nodes=training,gpus=gpus)
    template=tmp_path / "template.toml";template.write_text(ci.dump_toml(cfg))
    hosts=tmp_path / "hosts";hosts.write_text("\n".join(f"node{i}" for i in range(pool)))
    _,cfg=ci.render(template,hosts,sys.executable,tmp_path / "shared","topology")
    assert len(cfg["site"]["nodes"])==pool
    assert cfg["training"]["nodes"]==training and cfg["training"]["gpus"]==gpus
    assert len(cfg["site"]["nodes"])-cfg["training"]["nodes"]==pool-training


@pytest.mark.parametrize("nodes", ["node1,node2", "node1 node2", "node1\nnode2"])
def test_node_list_environment_uses_same_st_rendering(tmp_path,monkeypatch,nodes):
    monkeypatch.delenv("ST_HOSTFILE",raising=False)
    monkeypatch.setenv("ST_NODES",nodes)
    monkeypatch.setenv("ST_SHARED_ROOT",str(tmp_path / "shared"))
    monkeypatch.setenv("ST_CONTROL_PYTHON",sys.executable)
    assert ci.main(["render","--run-key","env-1"])==0
    run=tmp_path / "shared/ci-runs/env-1"
    cfg=load_site(run / "site.toml")
    assert cfg["site"]["nodes"]==["node1","node2"]
    assert cfg["training"]["nodes"]==1 and cfg["training"]["gpus"]==8
    marker=json.loads((run / "ci-run.json").read_text())
    assert marker["nodes_source"]=="ST_NODES"
    assert (run / "pool.hostfile").read_text()=="node1\nnode2\n"


@pytest.mark.parametrize("nodes", [" ","a,a","a slots=8,b","a;cmd,b","a[1-2],b"])
def test_bad_environment_node_list_rejected_before_run(tmp_path,nodes):
    with pytest.raises(ValueError):
        ci.render(ci.ST / "config/site.ci.toml",None,sys.executable,tmp_path / "shared","bad",nodes=nodes)
    assert not (tmp_path / "shared").exists()


def test_ambiguous_node_sources_rejected(tmp_path):
    with pytest.raises(ValueError,match="not both"):
        ci.render(ci.ST / "config/site.ci.toml",tmp_path / "hosts",sys.executable,tmp_path / "shared","bad",nodes="a,b")
    assert not (tmp_path / "shared").exists()


def test_render_preserves_virtualenv_interpreter_path(tmp_path,monkeypatch):
    executable=tmp_path / "venv/bin/python"
    base=tmp_path / "base/bin/python3.12"
    original=Path.resolve
    def resolve(path,*args,**kwargs):
        return base if path==executable else original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"resolve",resolve)
    hosts=tmp_path / "hosts";hosts.write_text("a\nb\nc\n")
    run,cfg=ci.render(ci.ST / "config/site.ci.toml",hosts,str(executable),tmp_path / "shared","venv-1")
    assert cfg["paths"]["control_python"]==str(executable)
    assert load_site(run / "site.toml")["paths"]["control_python"]==str(executable)


@pytest.mark.parametrize("missing_clush",[False,True])
def test_preflight_checks_docker_backend_dependencies(tmp_path,monkeypatch,missing_clush):
    repo=Path(os.environ.get("ST_CONTRACT_REPO",Path(__file__).resolve().parents[3]))
    hosts=tmp_path / "hosts";hosts.write_text("a\nb\nc\n")
    run,cfg=ci.render(ci.ST / "config/site.ci.toml",hosts,sys.executable,tmp_path / "shared","probe-1",repo_root=repo)
    monkeypatch.setattr(ci.sys,"platform","linux")
    monkeypatch.setattr(ci.shutil,"which",lambda name:None if name=="mpirun" or name=="clush" and missing_clush else name)
    monkeypatch.setattr(ci.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout="user stuser\nport 22\n"))
    scripts=[]
    def ssh(node,script,args,**kw):
        scripts.append(script)
        return SimpleNamespace(stdout="sha256:"+"a"*64)
    monkeypatch.setattr(ci,"ssh",ssh)
    if missing_clush:
        with pytest.raises(ValueError,match="command missing: clush"):ci.preflight(run,cfg)
        assert not scripts
        return
    ci.preflight(run,cfg)
    native=(repo / "hcu-envcheck/hcu_envcheck/payloads/idle_probe.sh").read_text(encoding="utf-8")
    assert len(scripts)==3 and all(script.startswith(native) and "\nhcu_idle_init host\n" in script for script in scripts)
    assert len(json.loads((run / "preflight.json").read_text()))==3


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_cleanup_requires_manager_stop_even_for_closed_case(tmp_path, monkeypatch, closed, stop_fails):
    run,case,data = registry(tmp_path)
    data.update(closed=closed, manager={"pid": 123, "pgid": 123, "starttime": "1"})
    ci.write_json(case / "ownership.json", data)
    calls=[]
    def stop(manager, run_id):
        assert manager == data["manager"] and run_id == data["run_id"]
        calls.append("stop")
        if stop_fails:
            raise ValueError("manager stop unconfirmed")
    monkeypatch.setattr(ci, "_stop_manager", stop)
    monkeypatch.setattr(ci, "ssh", lambda *a,**kw:calls.append("container"))
    assert ci.cleanup(run) is not stop_fails
    assert calls == (["stop"] if stop_fails else ["stop", "container"])
    receipt=json.loads((run / "ci-cleanup.json").read_text())
    assert receipt["confirmed"] is not stop_fails
    if stop_fails:
        assert "manager stop unconfirmed" in receipt["errors"][0]


@pytest.mark.parametrize("kill_succeeds", [False, True])
def test_stop_manager_confirms_escalation_before_return(monkeypatch, kill_succeeds):
    signals=SimpleNamespace(SIGTERM=15, SIGKILL=9)
    monkeypatch.setattr(ci, "signal", signals)
    alive=True
    sent=[]
    def kill(pid, sig):
        nonlocal alive
        assert pid == 123
        sent.append(sig)
        if sig == signals.SIGKILL and kill_succeeds:
            alive=False
    monkeypatch.setattr(ci, "_manager_alive", lambda *a:alive)
    monkeypatch.setattr(ci.os, "killpg", kill, raising=False)
    clock=iter(range(0, 1000, 30))
    monkeypatch.setattr(ci.time, "monotonic", lambda:next(clock))
    monkeypatch.setattr(ci.time, "sleep", lambda _:None)
    if kill_succeeds:
        ci._stop_manager({"pid":123}, "run")
    else:
        with pytest.raises(ValueError, match="stop unconfirmed"):
            ci._stop_manager({"pid":123}, "run")
    assert sent == [signals.SIGTERM, signals.SIGKILL]


def test_stop_manager_returns_after_confirmed_term(monkeypatch):
    monkeypatch.setattr(ci, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    alive=True
    sent=[]
    def kill(pid, sig):
        nonlocal alive
        sent.append(sig)
        alive=False
    monkeypatch.setattr(ci, "_manager_alive", lambda *a:alive)
    monkeypatch.setattr(ci.os, "killpg", kill, raising=False)
    ci._stop_manager({"pid":123}, "run")
    assert sent == [15]


@pytest.mark.parametrize("problem", ["reused_pid", "job", "pgid", "unreadable", "missing", "zombie", "running"])
def test_manager_live_identity_is_verified(tmp_path, monkeypatch, problem):
    proc=tmp_path / "proc";proc.mkdir()
    fields=["Z" if problem == "zombie" else "S"]+["0"]*19
    fields[19]="other" if problem == "reused_pid" else "100"
    (proc / "stat").write_text("123 (manager) " + " ".join(fields))
    job="other" if problem == "job" else "run"
    (proc / "cmdline").write_bytes(("python\0manager\0--job_name\0"+job+"\0").encode())
    monkeypatch.setattr(ci, "Path", lambda _:proc)
    monkeypatch.setattr(ci.os, "getpgid", lambda _:456 if problem == "pgid" else 123, raising=False)
    manager={"pid":123,"pgid":123,"starttime":"100","exited":True}
    if problem == "missing":
        (proc / "stat").unlink()
    if problem == "unreadable":
        def denied(*a,**kw): raise PermissionError("cannot verify process")
        monkeypatch.setattr(Path, "read_text", denied)
        with pytest.raises(PermissionError): ci._manager_alive(manager,"run")
    elif problem in {"reused_pid", "job", "pgid"}:
        with pytest.raises(ValueError, match="identity changed"): ci._manager_alive(manager,"run")
    else:
        assert ci._manager_alive(manager,"run") is (problem == "running")


def test_ci_passes_per_run_dependency_evidence_directory(tmp_path, monkeypatch):
    run,cfg=render(tmp_path)
    calls=[]
    def failed_prepare(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(poll=lambda:2, returncode=2)
    monkeypatch.setattr(ci.subprocess, "Popen", failed_prepare)
    with pytest.raises(ci.subprocess.CalledProcessError): ci.execute(run,cfg)
    assert len(calls)==1
    argv=calls[0]
    assert argv[argv.index("--evidence-dir")+1] == str(run / "bridge-dependency")
