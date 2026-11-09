# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests: temporary local Git repositories only, never SSH/Docker/training."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ST = Path(__file__).resolve().parents[1]
if str(ST) not in sys.path:
    sys.path.insert(0, str(ST))
import prepare_dependencies as prep
import run_st
from unit.site_fixture import site_file


def git(path, *args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="file")
    result = subprocess.run(["git", "-c", "core.longpaths=true", "-c", "user.name=ST fixture", "-c", "user.email=st@example.invalid",
                             *map(str, args)], cwd=path, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def repository(path, files):
    path.mkdir(parents=True)
    git(path, "init")
    for name, text in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    git(path, "add", ".")
    # Commits exist only in disposable fixture repositories, never the project.
    git(path, "commit", "-m", "fixture")
    return path


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Nested submodule .git/modules paths hit old Git for Windows limits easily.
    return tmp_path_factory.mktemp("b")


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    root = tmp_path_factory.mktemp("u")
    core = repository(root / "core", {"megatron/core/__init__.py": ""})
    bridge = repository(root / "upstream-bridge", {"src/megatron/bridge/__init__.py": ""})
    git(bridge, "submodule", "add", core.as_uri(), "3rdparty/Megatron-LM")
    git(bridge, "commit", "-am", "nested submodule")
    core_sha = git(core, "rev-parse", "HEAD")
    (core / "dev-feature.txt").write_text("pinned dev fixture")
    git(core, "add", ".")
    git(core, "commit", "-m", "dev fixture")
    (bridge / ".dev.commit").write_text(git(core, "rev-parse", "HEAD") + "\n")
    git(bridge, "add", ".dev.commit")
    git(bridge, "commit", "-m", "pin dev fixture")
    das = repository(root / "das", {"run_recipe.py": "raise RuntimeError('must never execute')\n",
                                    "env/env.sh": 'CURRENT_DIR=$(dirname "$0")\ncp source destination\n'})
    git(das, "submodule", "add", bridge.as_uri(), "3rdparty/Megatron-Bridge")
    git(das, "commit", "-am", "pinned bridge")
    pinned = git(das, "rev-parse", "HEAD")
    (das / "later.txt").write_text("newer head must not be selected")
    git(das, "add", ".")
    git(das, "commit", "-m", "newer commit")
    return das, pinned, core_sha, git(bridge, "rev-parse", "HEAD")


def dependency_file(path, url="https://example.invalid/bridge.git", commit="a"*40, timeout=30):
    path.write_text(prep.dump_toml({"bridge": dict(url=url, commit=commit, command_timeout=timeout)}), encoding="utf-8")
    return path


def test_fetch_pinned_commit_and_recursive_submodules(tmp_path, upstream, monkeypatch):
    das, commit, core_sha, bridge_sha = upstream
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    site = site_file(tmp_path / "site.toml")
    original = site.read_bytes()
    deps = dependency_file(tmp_path / "dependencies.toml", das.as_uri(), commit)
    result = prep.prepare(site, deps, tmp_path / "deps")
    checkout = Path(result["bridge_root"])
    assert git(checkout, "rev-parse", "HEAD") == commit
    assert not (checkout / "later.txt").exists()
    assert all((checkout / name).is_file() for name in prep.REQUIRED_FILES)
    record = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert record["status"] == "READY" and record["source_only"]
    assert {row["commit"] for row in record["bridge"]["submodules"]} == {core_sha, bridge_sha}
    assert len(record["warnings"]) == 2
    generated = run_st.load_site(result["site"])
    assert prep.prepared_metadata(generated) == record
    assert generated["paths"]["bridge_root"] == str(checkout)
    assert generated["training"]["ready"] is False
    expected = run_st.load_site(site)
    expected["paths"]["bridge_root"] = str(checkout)
    expected.pop("_site_file")
    generated.pop("_site_file")
    generated.pop("dependencies")
    assert generated == expected
    assert site.read_bytes() == original
    assert result["warnings"]


def test_each_preparation_is_isolated(tmp_path, upstream, monkeypatch):
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    site = site_file(tmp_path / "site.toml")
    deps = dependency_file(tmp_path / "deps.toml", upstream[0].as_uri(), upstream[1])
    first = prep.prepare(site, deps, tmp_path / "cache")
    marker = Path(first["bridge_root"]) / "keep-user-file"
    marker.write_text("retain")
    second = prep.prepare(site, deps, tmp_path / "cache")
    assert first["bridge_root"] != second["bridge_root"]
    assert marker.read_text() == "retain"


@pytest.mark.parametrize("commit", ["", "main", "latest", "v0.6.0", "a"*39, "g"*40, True])
def test_unpinned_config_fails_before_commands_or_directories(tmp_path, commit):
    deps = dependency_file(tmp_path / "deps.toml", commit=commit)
    with patch("subprocess.run", side_effect=AssertionError("Git must not run")), pytest.raises(ValueError, match="commit"):
        prep.prepare(tmp_path / "missing-site", deps, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("url", ["", "--upload-pack=bad", "https://host/repo\nextra", "relative/repo"])
def test_invalid_repository_rejected(tmp_path, url):
    with pytest.raises(ValueError, match="url"):
        prep.load_dependencies(dependency_file(tmp_path / "deps.toml", url=url))


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_invalid_git_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        prep.load_dependencies(dependency_file(tmp_path / "deps.toml", timeout=timeout))


def test_missing_commit_has_clear_cli_error_without_git(tmp_path, capsys):
    deps = dependency_file(tmp_path / "deps.toml", commit="")
    with patch("subprocess.run", side_effect=AssertionError("no subprocess")):
        assert run_st.main(["prepare", "--site", "not-needed", "--dependencies", str(deps)]) == 2
    assert "40-character" in capsys.readouterr().err


def test_failure_keeps_evidence_without_overwriting_site(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    site = site_file(tmp_path / "site.toml")
    original = site.read_bytes()
    deps = dependency_file(tmp_path / "deps.toml", (tmp_path / "nonexistent").as_uri())
    with pytest.raises(prep.DependencyError):
        prep.prepare(site, deps, tmp_path / "cache")
    record = json.loads(next((tmp_path / "cache").glob("*/manifest.json")).read_text())
    assert record["status"] == "FAILED"
    assert Path(record["log"]).is_file()
    assert not list((tmp_path / "cache").glob("*/site.prepared.toml"))
    assert site.read_bytes() == original


@pytest.mark.parametrize("failure", ["fetch", "timeout"])
def test_failed_ci_preparation_evidence_is_archived(tmp_path, monkeypatch, failure):
    import ci_entry as ci
    site=site_file(tmp_path / "site.toml")
    original=site.read_bytes()
    deps=dependency_file(tmp_path / "deps.toml", (tmp_path / "nonexistent").as_uri())
    run=tmp_path / "ci-run"
    evidence=run / "bridge-dependency"
    if failure == "fetch":
        monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    else:
        def timeout(*a, **kw): raise subprocess.TimeoutExpired("git",1)
        monkeypatch.setattr(prep.subprocess, "run", timeout)
    # Exercise both public CLI forwarding layers and the real failure handler.
    assert run_st.main(["prepare", "--site", str(site), "--dependencies", str(deps),
                        "--deps-root", str(tmp_path / "dependencies"),
                        "--evidence-dir", str(evidence)]) == 2
    record=json.loads((evidence / "manifest.json").read_text())
    log=(evidence / "prepare.log").read_text()
    assert record["status"] == "FAILED" and record["error"]
    assert str(tmp_path / "dependencies") in record["bridge"]["root"]
    assert "timed out" in log if failure == "timeout" else "nonexistent" in log
    out=tmp_path / "artifacts"
    ci.collect(run,out)
    assert json.loads((out / "bridge-dependency/manifest.json").read_text()) == record
    assert (out / "bridge-dependency/prepare.log").read_text() == log
    assert not (out / "bridge-dependency/site.prepared.toml").exists()
    assert not list(out.rglob(".git"))
    assert site.read_bytes() == original


def test_successful_ci_preparation_retains_generated_site_and_manifest(tmp_path, upstream, monkeypatch):
    import ci_entry as ci
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    site=site_file(tmp_path / "site.toml")
    deps=dependency_file(tmp_path / "deps.toml", upstream[0].as_uri(), upstream[1])
    run=tmp_path / "ci-run"
    evidence=run / "bridge-dependency"
    result=prep.prepare(site,deps,tmp_path / "dependencies",evidence_dir=evidence)
    assert Path(result["manifest"]) == evidence / "manifest.json"
    cfg=run_st.load_site(result["site"])
    assert prep.prepared_metadata(cfg)["status"] == "READY"
    out=tmp_path / "artifacts"
    ci.collect(run,out)
    assert (out / "bridge-dependency/manifest.json").is_file()
    assert (out / "bridge-dependency/site.prepared.toml").is_file()
    assert not (out / "Megatron-bridge-das").exists()


def test_incomplete_checkout_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    origin = repository(tmp_path / "incomplete", {"run_recipe.py": ""})
    deps = dependency_file(tmp_path / "deps.toml", origin.as_uri(), git(origin, "rev-parse", "HEAD"))
    with pytest.raises(prep.DependencyError, match="incomplete"):
        prep.prepare(site_file(tmp_path / "site.toml"), deps, tmp_path / "cache")


@pytest.mark.parametrize("change", ["manifest", "bridge-path"])
def test_prepared_identity_mismatch_rejected(tmp_path, change):
    root = tmp_path / "bridge"
    root.mkdir()
    data = dict(status="READY", bridge=dict(root=str(root), commit="a"*40, requested_commit="a"*40))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    cfg = dict(paths=dict(bridge_root=str(root)), dependencies=dict(bridge_manifest=str(manifest), bridge_manifest_sha256=prep.sha256(manifest)))
    if change == "manifest":
        manifest.write_text(json.dumps(dict(status="FAILED")))
    else:
        cfg["paths"]["bridge_root"] = str(tmp_path / "other")
    with pytest.raises(ValueError):
        prep.prepared_metadata(cfg)


def test_git_timeout_is_recorded(tmp_path):
    site = site_file(tmp_path / "site.toml")
    deps = dependency_file(tmp_path / "deps.toml")
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 1)), pytest.raises(prep.DependencyError):
        prep.prepare(site, deps, tmp_path / "cache")
    manifest = next((tmp_path / "cache").glob("*/manifest.json"))
    assert json.loads(manifest.read_text())["status"] == "FAILED"


def test_plan_does_not_fetch_and_windows_run_does_not_fetch(capsys, tmp_path):
    site = str(site_file(tmp_path / "site.toml"))
    with patch("subprocess.run", side_effect=AssertionError("no Git")), patch("subprocess.Popen", side_effect=AssertionError("no commands")):
        assert run_st.main(["plan", "--site", site]) == 0
        with patch.object(run_st.sys, "platform", "win32"):
            assert run_st.main(["run", "--site", site, "--dependencies", str(prep.DEFAULT_CONFIG)]) == 2


def test_automatic_run_preparation_passes_generated_site(tmp_path):
    site = site_file(tmp_path / "site.toml")
    prepared = str(tmp_path / "prepared.toml")
    Path(prepared).write_bytes(site.read_bytes())
    with patch.object(run_st, "validate_execution") as validate, \
         patch.object(prep, "prepare", return_value=dict(site=prepared)) as prepare, \
         patch("subprocess.call", return_value=0) as call:
        assert run_st.main(["run", "--site", str(site), "--dependencies", "deps.toml"]) == 0
    assert validate.call_args_list[0].kwargs == {"allow_missing_bridge": True}
    prepare.assert_called_once()
    argv = call.call_args.args[0]
    assert argv[argv.index("--site")+1] == prepared


def test_toml_roundtrip_preserves_site_values():
    values = {"table.with.dots": {"text": '中文 "quoted"\nnext', "flag": False,
                                  "list": ["one", "two"], "nested": {"x": 2}}}
    assert prep.tomllib.loads(prep.dump_toml(values)) == values


def test_mcore_dev_uses_upstream_pin_without_executing_switch_script(tmp_path, upstream, monkeypatch):
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    deps = dependency_file(tmp_path / "deps.toml", upstream[0].as_uri(), upstream[1])
    config = prep.tomllib.loads(deps.read_text())
    config["bridge"]["mcore"] = "dev"
    deps.write_text(prep.dump_toml(config))
    result = prep.prepare(site_file(tmp_path / "site.toml"), deps, tmp_path / "d")
    record = json.loads(Path(result["manifest"]).read_text())
    pin = (Path(result["bridge_root"]) / "3rdparty/Megatron-Bridge/.dev.commit").read_text().strip()
    assert record["bridge"]["mcore"] == dict(mode="dev", gitlink_commit=upstream[2], commit=pin)
    assert pin != upstream[2]
    mcore = Path(result["bridge_root"]) / "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM"
    assert git(mcore, "rev-parse", "HEAD") == pin
    assert next(row for row in record["bridge"]["submodules"] if row["path"].endswith("Megatron-LM"))["commit"] == pin


def test_case_freezes_prepared_bridge_evidence(tmp_path):
    from lib.clients import CaseContext
    cfg = run_st.load_site(site_file(tmp_path / "site.toml"))
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    source = tmp_path / "train_st.sh"
    source.write_text("# never executed\n")
    train_config = tmp_path / "training.json"
    train_config.write_text("{}")
    manifest = tmp_path / "manifest.json"
    metadata = dict(status="READY", source_only=True, warnings=[],
                    bridge=dict(root=str(bridge), commit="a"*40, requested_commit="a"*40, submodules=[]))
    manifest.write_text(json.dumps(metadata))
    cfg["paths"].update(shared_root=str(tmp_path / "runs"), bridge_root=str(bridge))
    cfg["training"]["script"] = str(source)
    cfg["reliability"]["train_config"] = str(train_config)
    cfg["reliability"]["env"] = {"MPI_HOST_SSH_PORT": "22"}
    cfg["timeouts"].update(command=1, total=5)
    cfg["dependencies"] = dict(bridge_manifest=str(manifest), bridge_manifest_sha256=prep.sha256(manifest))
    ctx = CaseContext(cfg, "ENV-001")
    with patch("subprocess.run", side_effect=AssertionError("no cluster commands")):
        ctx._prepare_runtime()
    copied = Path(ctx.configuration["bridge_dependency"]["record"])
    assert json.loads(copied.read_text()) == metadata
    manifest.write_text("changed after capture")
    assert json.loads(copied.read_text()) == metadata


def test_wrong_fetched_head_is_rejected(tmp_path):
    site = site_file(tmp_path / "site.toml")
    deps = dependency_file(tmp_path / "deps.toml")
    def fake_git(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "b"*40 + "\n" if "rev-parse" in argv else "", "")
    with patch("subprocess.run", side_effect=fake_git), pytest.raises(prep.DependencyError, match="configured commit"):
        prep.prepare(site, deps, tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*/site.prepared.toml"))


@pytest.mark.parametrize("prefix", ["-", "+", "U"])
def test_uninitialized_or_unpinned_submodule_is_rejected(tmp_path, prefix):
    site = site_file(tmp_path / "site.toml")
    deps = dependency_file(tmp_path / "deps.toml")
    def fake_git(argv, **kwargs):
        output = "a"*40 + "\n" if "rev-parse" in argv else ""
        if "status" in argv:
            output = prefix + "c"*40 + " 3rdparty/Megatron-Bridge\n"
        return subprocess.CompletedProcess(argv, 0, output, "")
    with patch("subprocess.run", side_effect=fake_git), pytest.raises(prep.DependencyError, match="submodule"):
        prep.prepare(site, deps, tmp_path / "cache")
