# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Local configuration/launcher tests; no framework, network, GPU or training."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from unit.site_fixture import site_config

import pytest

ST = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("st_glm53_recipe", ST / "payloads/st_glm53_recipe.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


@pytest.mark.parametrize("nodes,gpus", [(1,8), (2,4), (3,4), (2,3)])
def test_topology_and_manager_record_agree(nodes, gpus):
    env = dict(NNODES=str(nodes), GPUS=str(gpus))
    values = recipe.settings(env)
    record = recipe.manager_config(env)
    assert values["GLOBAL_BATCH_SIZE"] == nodes*gpus
    assert record["world-size"] == nodes*gpus
    assert record["global-batch-size"] == nodes*gpus
    assert record["num-layers"] == 2 and record["hidden-size"] == 512


@pytest.mark.parametrize("extra", [dict(GLOBAL_BATCH_SIZE="7"), dict(TP="3"), dict(EP="3"),
                                  dict(NUM_LAYERS="1"), dict(PP="3"), dict(SEQ_LENGTH="16")])
def test_invalid_model_parallelism_or_batch_rejected(extra):
    with pytest.raises(ValueError):
        recipe.settings(dict(NNODES="2", GPUS="4", **extra))


@pytest.mark.parametrize("local_interval", [None, 0, 5])
def test_no_hf_recipe_preserves_glm_structure_and_checkpoint_mode(monkeypatch, tmp_path, local_interval):
    def module(name):
        result = ModuleType(name)
        result.__path__ = []
        monkeypatch.setitem(sys.modules, name, result)
        return result

    for name in ("torch", "torch.nn", "megatron", "megatron.core", "megatron.core.models",
                 "megatron.core.models.gpt", "megatron.bridge", "megatron.bridge.models",
                 "megatron.bridge.recipes", "megatron.bridge.training", "megatron.bridge.training.tokenizers"):
        module(name)
    functional = module("torch.nn.functional")
    functional.silu = object()
    attention = module("megatron.core.models.gpt.experimental_attention_variant_module_specs")
    attention.get_transformer_block_with_experimental_attention_variant_spec = object()
    provider = module("megatron.bridge.models.mla_provider")
    provider.MLAModelProvider = lambda **kw: SimpleNamespace(**kw)
    tokenizer = module("megatron.bridge.training.tokenizers.config")
    tokenizer.TokenizerConfig = lambda **kw: SimpleNamespace(**kw)
    common = module("megatron.bridge.recipes.common")
    common._pretrain_common = lambda: SimpleNamespace(**{name: SimpleNamespace() for name in
        ("dataset", "train", "validation", "ddp", "optimizer", "scheduler", "checkpoint", "logger")})
    env = dict(NNODES="2", GPUS="4", CKPT_DIR=str(tmp_path / "global"),
               LOCAL_CKPT_DIR=str(tmp_path / "local"), LOG_DIR=str(tmp_path / "logs"))
    if local_interval is not None:
        env["LOCAL_SAVE_INTERVAL"] = str(local_interval)
    with patch.dict(os.environ, env, clear=True):
        cfg = recipe.glm53_st_pretrain_config()
    assert cfg.model.multi_latent_attention and cfg.model.experimental_attention_variant == "dsa"
    assert cfg.model.moe_layer_freq == [0,1] and cfg.model.num_moe_experts == 8
    assert cfg.model.moe_token_dispatcher_type == "alltoall" and cfg.model.dsa_kernel_backend == "none"
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer" and cfg.tokenizer.vocab_size == 4096
    assert cfg.dataset.blend is None and cfg.dataset.num_workers == 0
    assert cfg.checkpoint.load == cfg.checkpoint.save == env["CKPT_DIR"]
    assert cfg.checkpoint.save_interval == 10
    if local_interval == 0:
        assert cfg.checkpoint.non_persistent_save_interval is None
        assert cfg.checkpoint.non_persistent_ckpt_type is None
        assert cfg.checkpoint.non_persistent_local_ckpt_dir is None
    else:
        assert cfg.checkpoint.non_persistent_save_interval == 5
        assert cfg.checkpoint.non_persistent_ckpt_type == "local"
        assert cfg.checkpoint.non_persistent_local_ckpt_dir == env["LOCAL_CKPT_DIR"]
    assert cfg.checkpoint.ckpt_format == "torch_dist" and cfg.checkpoint.async_save is False
    assert cfg.checkpoint.pretrained_checkpoint is None and cfg.checkpoint.also_save_hf_checkpoint is False
    assert cfg.inprocess_restart is None


def test_bootstrap_registers_in_memory_then_calls_bridge(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("st_bootstrap", ST / "payloads/st_run_recipe.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    registry = ModuleType("megatron.bridge.recipes")
    for name in ("megatron", "megatron.bridge"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "megatron.bridge.recipes", registry)
    monkeypatch.setitem(sys.modules, "st_glm53_recipe", recipe)
    monkeypatch.setenv("MEGATRON_BRIDGE_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["st_run_recipe.py", "--mode", "pretrain"])
    with patch.object(bootstrap.runpy, "run_path") as call:
        bootstrap.main()
    call.assert_called_once_with(str(tmp_path / "run_recipe.py"), run_name="__main__")
    assert getattr(registry, recipe.RECIPE_NAME) is recipe.glm53_st_pretrain_config
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert not list(tmp_path.iterdir())


def test_auto_manager_config_uses_same_values_without_hf(tmp_path):
    import run_st
    from lib.clients import CaseContext
    cfg = site_config()
    cfg["site"]["nodes"] = ["a", "b", "c", "d"]
    cfg["training"].update(nodes=3, gpus=4, script=str(ST / "payloads/train_st.sh"))
    cfg["paths"].update(shared_root=str(tmp_path / "runs"), bridge_root=str(tmp_path / "bridge"))
    cfg["reliability"]["env"] = {"MPI_HOST_SSH_PORT": "22"}
    cfg["timeouts"].update(command=1, total=5)
    ctx = CaseContext(cfg, "REL-010")
    with patch("subprocess.run", side_effect=AssertionError("no remote commands")):
        ctx._prepare_runtime()
    import json
    record = json.loads(Path(ctx.configuration["train_config_copy"]).read_text())
    assert record["world-size"] == 12 and record["global-batch-size"] == 12
    text = (ctx.case_root / "training.env").read_text()
    assert "LOCAL_HF_PATH" not in text and "export GLOBAL_BATCH_SIZE=12" in text
    assert "export HCU_ST_PREPARE_DATASET_HELPERS=1" in text
    assert (ctx.case_root / "payloads/st_run_recipe.py").is_file()
    assert (ctx.case_root / "payloads/st_glm53_recipe.py").is_file()
    assert "st_glm53_recipe.py" in ctx.configuration["payload_sha256"]


@pytest.mark.parametrize("nodes,gpus,code", [(1,8,0),(2,4,0),(3,4,0),(2,3,7)])
def test_shell_launch_contract_without_training(tmp_path, nodes, gpus, code):
    bash = r"C:\Program Files\Git\bin\bash.exe" if sys.platform == "win32" else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    root = tmp_path / "case with spaces"
    payload = root / "payloads"
    payload.mkdir(parents=True)
    for name in ("train_st.sh", "st_run_recipe.py", "st_glm53_recipe.py"):
        shutil.copyfile(ST / "payloads" / name, payload / name)
    bridge = root / "bridge"
    bridge.mkdir()
    (bridge / "run_recipe.py").write_text("raise RuntimeError('must never execute')")
    hosts = root / "hosts"
    hosts.write_text("\n".join(f"node-{i} slots={gpus}" for i in range(nodes)) + "\n")
    environment = root / "env.sh"
    environment.write_text("export ST_ENV_READY=1\n", newline="\n")
    bins = root / "bin"
    bins.mkdir()
    (bins / "torchrun").write_text('#!/usr/bin/env bash\nset -eu\ntest "$ST_ENV_READY" = 1\nprintf "%s\\n" "$@" > "$ST_TEST_ARGS"\nexit "$ST_TEST_RETURN"\n', newline="\n")
    (bins / "mpirun").write_text('#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$@" > "$ST_MPI_ARGS"\nwhile [[ "$1" != bash ]]; do shift; done\nexport OMPI_COMM_WORLD_RANK=1\nexec "$@"\n', newline="\n")
    for name in ("torchrun", "mpirun"):
        (bins / name).chmod(0o755)
    prefix = bins.as_posix()
    if sys.platform == "win32":
        prefix = "/" + prefix[0].lower() + prefix[2:]
    env = dict(os.environ, PATH=prefix + (":/usr/bin:/bin" if sys.platform == "win32" else ":" + os.environ["PATH"]),
               MSYS2_ARG_CONV_EXCL="*", MEGATRON_BRIDGE_PATH=bridge.as_posix(),
               CKPT_DIR=(root / "global").as_posix(), LOCAL_CKPT_DIR=(root / "local").as_posix(),
               LOG_DIR=(root / "logs").as_posix(), TRAIN_ENV_SCRIPT=environment.as_posix(),
               GPUS=str(gpus), NNODES=str(nodes), ST_TEST_ARGS=(root / "args").as_posix(),
               ST_MPI_ARGS=(root / "mpi-args").as_posix(), ST_TEST_RETURN=str(code),
               HCU_ST_RUN_ID="test-run", HCU_ST_ATTEMPT_ID="test-attempt")
    env.pop("LOCAL_HF_PATH", None)
    result = subprocess.run([bash, (payload / "train_st.sh").as_posix(), hosts.as_posix()], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
    assert result.returncode == code, result.stdout + result.stderr
    args = (root / "args").read_text().splitlines()
    assert f"--nnodes={nodes}" in args and f"--nproc_per_node={gpus}" in args and "--max-restarts=0" in args
    assert f"--node_rank={0 if nodes == 1 else 1}" in args
    expected = (payload / "st_run_recipe.py").as_posix()
    if sys.platform == "win32":
        expected = "/" + expected[0].lower() + expected[2:]
    assert expected in args
    assert args[args.index("--dataset")+1] == "mock"
    assert "HF" not in " ".join(args)
    if nodes > 1:
        assert "LOCAL_CKPT_DIR" in (root / "mpi-args").read_text()
        assert "HCU_ST_ATTEMPT_ID" in (root / "mpi-args").read_text()


@pytest.mark.parametrize("enabled,build_code", [(True,0),(True,9),(False,0)])
def test_wrapper_builds_shared_dataset_helper_before_training_and_preserves_failure(tmp_path, enabled, build_code):
    from lib.clients import CaseContext
    root = tmp_path / "case with spaces"; payload = root / "payloads"; payload.mkdir(parents=True)
    shutil.copyfile(ST / "payloads/run_train.sh", payload / "run_train.sh")
    script = payload / "fake-train.sh"
    script.write_text('set -eu\nprintf "train\\n" >> "$ST_ORDER"\n', newline="\n")
    bridge = root / "bridge"; datasets = bridge / "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets"
    datasets.mkdir(parents=True)
    env_script = root / "env.sh"; env_script.write_text('export ST_ENV_READY=1\n', newline="\n")
    order = root / "order"
    CaseContext._write_exports(payload / "context.env", {
        "HCU_ST_CASE_ROOT": root.as_posix(), "HCU_ST_RUN_ID": "unit-wrapper",
        "HCU_ST_TRAIN_SCRIPT": script.as_posix(), "HCU_ST_PYTHON": Path(sys.executable).as_posix()})
    CaseContext._write_exports(root / "training.env", {
        "MEGATRON_BRIDGE_PATH": bridge.as_posix(), "TRAIN_ENV_SCRIPT": env_script.as_posix(),
        "HCU_ST_PREPARE_DATASET_HELPERS": "1" if enabled else "0", "ST_ORDER": order.as_posix()})
    bins = root / "bin"; bins.mkdir()
    for name, body in {
        "flock": "exit 0",
        "make": 'set -eu; test "$ST_ENV_READY" = 1; printf "build\\n" >> "$ST_ORDER"; exit '+str(build_code),
    }.items():
        path = bins / name; path.write_text("#!/usr/bin/env bash\n"+body+"\n", newline="\n"); path.chmod(0o755)
    hosts = root / "slots"; hosts.write_text("node-a slots=8\n")
    prefix = bins.as_posix()
    if sys.platform == "win32": prefix = "/" + prefix[0].lower() + prefix[2:]
    environment = dict(os.environ, PATH=prefix+(":/usr/bin:/bin" if sys.platform=="win32" else ":"+os.environ["PATH"]))
    environment.pop("MSYS2_ARG_CONV_EXCL", None)
    bash = r"C:\Program Files\Git\bin\bash.exe" if sys.platform=="win32" else shutil.which("bash")
    result = subprocess.run([bash, (payload / "run_train.sh").as_posix(), hosts.as_posix()], env=environment,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == (build_code if enabled else 0), result.stdout+result.stderr
    assert order.read_text().splitlines() == (["build"] if enabled and build_code else ["build","train"] if enabled else ["train"])
    import json
    record = json.loads(next((root / "attempts").glob("*/attempt.json")).read_text())
    assert record["status"] == "EXITED" and record["exit_code"] == result.returncode
