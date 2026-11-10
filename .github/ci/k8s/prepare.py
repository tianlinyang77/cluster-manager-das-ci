# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Stage a PR's training payload and pinned dependencies on the shared PVC."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

from transport import kube_api

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
sys.path.insert(0, str(REPO / "tests/st"))
from prepare_dependencies import dump_toml, load_dependencies, prepare


def main():
    nodes, image, shared, key = validate_inputs(os.environ)
    source = Path(os.environ["ST_SOURCE_ROOT"]).resolve()
    artifacts = Path(os.environ["ST_K8S_ARTIFACTS"]).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    namespace = os.environ.get("ST_K8S_NAMESPACE", "arc-runners")
    pool = os.environ.get("ST_HCU_POOL", "hygon-bw1100-8card")
    with kube_api() as (session, server):
        response = session.get(server + f"/api/v1/namespaces/{namespace}/configmaps/job-template-{pool}", timeout=(5, 20))
        response.raise_for_status()
        profile = json.loads(response.json()["data"]["content"])
        cards = profile["spec"]["containers"][0]["resources"]["requests"]["hygon.com/hcu"]
        if cards != 8:
            raise ValueError("the selected CI pool must allocate eight HCU devices")
        for name in nodes:
            response = session.get(server + "/api/v1/nodes/" + name, timeout=(5, 20))
            response.raise_for_status()
            validate_node(response.json(), cards, os.environ.get("ST_ALLOW_DEGRADED") == "true")
    if os.environ.get("ST_ALLOW_DEGRADED") == "true":
        profile["spec"].get("nodeSelector", {}).pop("ci.hygon.cn/hcu-health", None)
    run = shared / "ci-runs" / key
    run.mkdir(parents=True, exist_ok=False)
    (run / "storage-marker").write_text(run.name)
    stage_source(source, run / "source/tests/st")
    st = run / "source/tests/st"
    cfg = tomllib.loads((REPO / "tests/st/config/site.ci.toml").read_text(encoding="utf-8"))
    cfg["site"].update(nodes=nodes, image=image)
    cfg["training"].update(ready=False, script=str(st / "payloads/train_st.sh"))
    cfg["paths"].update(repo_root=str(run / "source"), shared_root=str(run / "cases"), control_python=sys.executable,
                        bridge_root="", env_script=str(st / "config/container.ci.env.sh"))
    site = run / "site.source-only.toml"
    site.write_text(dump_toml(cfg), encoding="utf-8")
    spec = REPO / "tests/st/config/dependencies.toml"
    try:
        prepared = prepare(site, spec, run / "dependencies", evidence_dir=run / "bridge-evidence")
    finally:
        if (run / "bridge-evidence").exists():
            shutil.copytree(run / "bridge-evidence", artifacts / "bridge-evidence")
    (run / "prepare-result.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    lock = Path(prepared["bridge_root"]) / "3rdparty/Megatron-Bridge/uv.lock"
    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    if not any(p["name"] == "megatron-energon" and p["version"] == "7.4.0" for p in packages):
        raise ValueError("review the Energon runtime pin when changing the Bridge lockfile")
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
                    "--target", str(run / "python-deps"), "--report", str(artifacts / "runtime-pip-report.json"),
                    "megatron-energon==7.4.0"], check=True, timeout=180)
    revisions = [subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() for path in (source, REPO)]
    state = dict(run_root=str(run), namespace=namespace, image=image, nodes=nodes, pool=pool,
                 pool_template=profile, gpus=cards, source_commit=revisions[0], controller_commit=revisions[1],
                 bridge_commit=load_dependencies(spec)["commit"])
    (artifacts / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": str(run), "pool": pool, "cards": cards}), flush=True)


def validate_inputs(environ):
    nodes = environ.get("ST_K8S_NODES", "").split()
    if len(nodes) != 2 or len(set(nodes)) != 2 or any(not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", n) for n in nodes):
        raise ValueError("ST_K8S_NODES must explicitly name two distinct nodes: fault node, then standby")
    image = environ.get("ST_TRAINING_IMAGE", "")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("ST_TRAINING_IMAGE must use a fixed sha256 digest")
    shared = Path(environ.get("ST_SHARED_ROOT", ""))
    if not shared.is_absolute() or len(shared.parts) < 3 or shared == REPO or REPO in shared.parents:
        raise ValueError("ST_SHARED_ROOT must be a dedicated absolute shared directory outside the checkout")
    key = environ.get("ST_RUN_KEY", "")
    if not re.fullmatch(r"[0-9]+-[0-9]+", key):
        raise ValueError("ST_RUN_KEY must be the Actions run ID and attempt")
    return nodes, image, shared, key


def stage_source(source, destination):
    origin = source / "tests/st"
    if not origin.resolve().is_relative_to(source):
        raise ValueError("ST training source must stay inside its checkout")
    shutil.copytree(origin, destination, symlinks=True)


def validate_node(node, cards, allow_degraded):
    if node["spec"].get("unschedulable") or not any(c["type"] == "Ready" and c["status"] == "True" for c in node["status"]["conditions"]):
        raise ValueError("test node is not Ready and schedulable: " + node["metadata"]["name"])
    if int(node["status"].get("allocatable", {}).get("hygon.com/hcu", 0)) < cards:
        raise ValueError("test node has insufficient allocatable HCU devices")
    if node["metadata"].get("labels", {}).get("ci.hygon.cn/hcu-health") == "degraded" and not allow_degraded:
        raise ValueError("degraded test node requires explicit ST_ALLOW_DEGRADED=true")


if __name__ == "__main__":
    main()
