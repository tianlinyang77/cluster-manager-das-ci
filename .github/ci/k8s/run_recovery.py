# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native FT fault handling, node exclusion, Volcano migration and checkpoint resumption."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import secrets
import signal
import os
import subprocess
import sys
import time

import yaml
from transport import kube_api, exec_in_pod, cleanup_owned


def cancelled(signum, frame):
    raise KeyboardInterrupt("CI cancellation received")


signal.signal(signal.SIGTERM, cancelled)

scripts = Path(__file__).resolve().parent
trusted_repo = scripts.parents[2]
root = Path(os.environ["ST_K8S_ARTIFACTS"]).resolve()
state = json.loads((root / "state.json").read_text(encoding="utf-8"))
gpus = state["gpus"]
name = "st-crossnode-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + secrets.token_hex(3)
primary, standby = state["nodes"]
taint_key = "st.ft.hygon.io/" + name
ns = state["namespace"]
case = state["run_root"] + "/" + name
evidence = root / name
evidence.mkdir()
sys.path.insert(0, str(trusted_repo / "tests/st"))
from lib.training import parse_training_log

cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": name, "namespace": ns},
      "data": {"bootstrap.py": (scripts / "train.py").read_text(encoding="utf-8"),
               "snapshot.py": (scripts / "snapshot.py").read_text(encoding="utf-8")}}
job = {"apiVersion": "batch.volcano.sh/v1alpha1", "kind": "Job", "metadata": {"name": name, "namespace": ns},
       "spec": {"minAvailable": 1, "schedulerName": "volcano", "queue": "default", "maxRetry": 2, "ttlSecondsAfterFinished": 600,
                "policies": [{"event": "TaskCompleted", "action": "CompleteJob"}],
                "tasks": [{"name": "train", "replicas": 1,
                           "policies": [{"events": ["PodFailed", "PodEvicted"], "action": "RestartJob"}],
                           "template": {"metadata": {"labels": {"st-run": name}}, "spec": {
                    "restartPolicy": "Never", "activeDeadlineSeconds": 900, "automountServiceAccountToken": False,
                    "affinity": {"nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [
                            {"key": "kubernetes.io/hostname", "operator": "In", "values": [primary, standby]}]}]},
                        "preferredDuringSchedulingIgnoredDuringExecution": [{"weight": 100, "preference": {"matchExpressions": [
                            {"key": "kubernetes.io/hostname", "operator": "In", "values": [primary]}]}}]}},
                    "containers": [{"name": "train", "image": state["image"], "imagePullPolicy": "IfNotPresent",
                                    "command": ["python3", "-u", "/st-input/bootstrap.py"],
                                    "env": [{"name": "ST_RUN_ROOT", "value": state["run_root"]}, {"name": "ST_RECOVERY_ID", "value": name},
                                            {"name": "ST_GPUS", "value": str(gpus)},
                                            *[{"name": k, "valueFrom": {"fieldRef": {"fieldPath": v}}}
                                              for k, v in (("POD_NAME", "metadata.name"), ("POD_UID", "metadata.uid"), ("NODE_NAME", "spec.nodeName"))]],
                                    "resources": {"requests": {"cpu": "16", "memory": "64Gi", "hygon.com/hcu": gpus},
                                                  "limits": {"cpu": "32", "memory": "128Gi", "hygon.com/hcu": gpus}},
                                    "volumeMounts": [{"name": "source", "mountPath": "/st-input", "readOnly": True},
                                                     {"name": "shared", "mountPath": "/ci_public"}, {"name": "shm", "mountPath": "/dev/shm"}]}],
                    "volumes": [{"name": "source", "configMap": {"name": name}},
                                {"name": "shared", "persistentVolumeClaim": {"claimName": "ci-public"}},
                                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
                }}}]}}
profile = state["pool_template"]
job["spec"]["queue"] = profile["metadata"]["annotations"]["scheduling.volcano.sh/queue-name"]
template = job["spec"]["tasks"][0]["template"]
template["spec"]["nodeSelector"] = profile["spec"].get("nodeSelector", {})
template["spec"]["containers"][0]["resources"] = profile["spec"]["containers"][0]["resources"]
template["spec"]["volumes"][-1]["emptyDir"] = next(v["emptyDir"] for v in profile["spec"]["volumes"] if v["name"] == "dshm")
assert template["spec"]["containers"][0]["resources"]["requests"]["hygon.com/hcu"] == gpus
(evidence / "job.yaml").write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
receipt = {"status": "PREPARING", "name": name, "namespace": ns, "case": case, "image": state["image"],
           "source_commit": state["source_commit"], "bridge_commit": state["bridge_commit"],
           "world_size": gpus, "train_iters": 1000, "save_interval": 50, "local_checkpoint": "NOT_EXECUTED",
           "native_worker_helper_sha256": hashlib.sha256((trusted_repo / "tests/st/lib/training.py").read_bytes()).hexdigest(),
           "pool": state["pool"], "controller_commit": state["controller_commit"],
           "fault_transport": "one simulated NodeHealthCheckFailed FaultEvent; native FT Operator taint/delete actions",
           "primary_node": primary, "standby_node": standby, "taint_key": taint_key,
           "launch_mode": "torchrun PID1; native Python ST recipe; Kubernetes stdout without tee",
           "fault_injected": False, "recovery_observed": False, "pod_uids": []}


def save():
    (evidence / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def complete_workers(snapshot):
    workers = snapshot["workers"]
    attempt = snapshot.get("current", {}).get("attempt_id") if snapshot.get("current") else None
    return (len(workers) == gpus and {w["rank"] for w in workers} == set(range(gpus))
            and {w["local_rank"] for w in workers} == set(range(gpus))
            and all(w["world_size"] == gpus and str(w["max_restarts"]) == "0" and w["attempt_id"] == attempt for w in workers))


save()
(root / "active-crossnode.json").write_text(json.dumps({"evidence": str(evidence), "name": name, "namespace": ns, "case": case}, indent=2))
with kube_api() as (session, server):
    def request(method, path, **kwargs):
        response = session.request(method, server + path, timeout=(5, 25), **kwargs)
        if method == "DELETE" and response.status_code == 404:
            return {}
        if not response.ok:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code}: {response.text[:1000]}")
        return response.json()

    def snapshot(pod, parsed):
        saved = [n for n in parsed["global_saves"] if parsed["save_formats"].get(n) == "torch_dist"]
        result = exec_in_pod(session, server, ns, pod["metadata"]["name"], "train",
                             ["python3", "/st-input/snapshot.py", str(saved[-1] if saved else 0)])
        if result["status"].get("status") != "Success":
            raise RuntimeError("snapshot exec failed: " + str(result))
        data = json.loads(result["stdout"])
        data["parsed"] = parsed
        assert data["pod_uid"] == pod["metadata"]["uid"] and data["recovery_id"] == name
        return data

    jobs = f"/apis/batch.volcano.sh/v1alpha1/namespaces/{ns}/jobs"
    pods_path = f"/api/v1/namespaces/{ns}/pods"
    cm_path = f"/api/v1/namespaces/{ns}/configmaps"
    created_cm = request("POST", cm_path, json=cm)
    receipt["configmap_uid"] = created_cm["metadata"]["uid"]
    save()
    created_job = None
    created_crd = None
    created_fault = None
    operator = None
    operator_log = None
    crd_path = "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/faultevents.ft.hygon.io"
    fault_path = f"/apis/ft.hygon.io/v1alpha1/namespaces/{ns}/faultevents"
    try:
        crd_check = session.get(server + crd_path, timeout=(5, 20))
        if crd_check.status_code == 404:
            crd = yaml.safe_load((trusted_repo / "hygon-ft-k8s/manifests/crds/faultevents.yaml").read_text(encoding="utf-8"))
            created_crd = request("POST", crd_path.rsplit("/", 1)[0], json=crd)
            receipt["created_crd_uid"] = created_crd["metadata"]["uid"]
            save()
        else:
            crd_check.raise_for_status()
            raise RuntimeError("FaultEvent CRD now exists; review existing FT ownership before starting another watcher")
        for _ in range(30):
            current_crd = request("GET", crd_path)
            if any(c["type"] == "Established" and c["status"] == "True" for c in current_crd.get("status", {}).get("conditions", [])):
                break
            time.sleep(1)
        else:
            raise RuntimeError("FaultEvent CRD was not established")
        source = trusted_repo / "hygon-ft-k8s/hygon_ft/operator/controller.py"
        operator_env = dict(os.environ, POD_NAMESPACE=ns, POD_NAME=name,
                            FT_UNHEALTHY_TAINT_KEY=taint_key, FT_LEADER_ELECTION="false",
                            FT_ENABLE_NODE_NOT_READY_WATCH="false", FEISHU_WEBHOOK_URL="", FT_ALERT_COMMAND="")
        operator_log = (evidence / "operator.log").open("w", encoding="utf-8")
        operator_code = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['load_kube'](); m['FaultController']().run()"
        operator = subprocess.Popen([sys.executable, "-u", "-c", operator_code, str(source)], env=operator_env,
                                    stdout=operator_log, stderr=subprocess.STDOUT)
        receipt["operator_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        time.sleep(2)
        if operator.poll() is not None:
            raise RuntimeError("native FT Operator failed to start; see operator.log")
        created_job = request("POST", jobs, json=job)
        receipt.update(status="RUNNING", job_uid=created_job["metadata"]["uid"])
        save()
        print(json.dumps({"created": name, "case": case, "evidence": str(evidence)}), flush=True)
        deadline = time.monotonic() + 900
        previous = None
        while time.monotonic() < deadline:
            if operator.poll() is not None:
                raise RuntimeError("native FT Operator exited; see operator.log")
            if created_fault:
                fault = request("GET", fault_path + "/" + name)
                receipt["fault_event"] = fault
                (evidence / "fault-event.json").write_text(json.dumps(fault, indent=2), encoding="utf-8")
                actions = fault.get("status", {}).get("actions", [])
                if any(a.get("success") is False for a in actions):
                    raise RuntimeError("native FT Operator action failed: " + str(actions))
            current = request("GET", jobs + "/" + name)
            pods = request("GET", pods_path, params={"labelSelector": "st-run=" + name})["items"]
            phase = current.get("status", {}).get("state", {}).get("phase")
            summary = {"phase": phase, "pods": [{"name": p["metadata"]["name"], "uid": p["metadata"]["uid"],
                                                 "phase": p.get("status", {}).get("phase"), "node": p["spec"].get("nodeName")} for p in pods]}
            if summary != previous:
                print(json.dumps(summary), flush=True)
                with (evidence / "timeline.jsonl").open("a") as log:
                    log.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), **summary)) + "\n")
                previous = summary
            receipt["job_status"] = current.get("status", {})
            for pod in pods:
                assert any(r["uid"] == receipt["job_uid"] for r in pod["metadata"].get("ownerReferences", []))
                uid = pod["metadata"]["uid"]
                if uid not in receipt["pod_uids"]:
                    receipt["pod_uids"].append(uid)
                response = session.get(server + pods_path + "/" + pod["metadata"]["name"] + "/log", params={"timestamps": "true"}, timeout=(5, 20))
                if response.status_code == 200:
                    (evidence / ("pod-" + uid + ".log")).write_text(response.text, encoding="utf-8")
                if pod.get("status", {}).get("phase") != "Running":
                    continue
                if receipt["fault_injected"] and (uid == receipt["initial_pod_uid"] or receipt["recovery_observed"]):
                    continue
                if not any(c.get("state", {}).get("running") for c in pod.get("status", {}).get("containerStatuses", [])):
                    continue
                observed = snapshot(pod, parse_training_log(response.text if response.status_code == 200 else ""))
                if not receipt["fault_injected"]:
                    parsed = observed.get("parsed")
                    if not (parsed and parsed["global_saves"] and observed["checkpoint_files"] and complete_workers(observed)):
                        continue
                    assert pod["spec"]["nodeName"] == primary, "initial placement was not on the selected fault node"
                    receipt["before_injection"] = observed
                    receipt["initial_pod_uid"] = uid
                    receipt["injection_intent"] = {"pod_uid": uid, "node": primary, "state": "RECORDED_BEFORE_FAULT_EVENT"}
                    save()
                    fault = {"apiVersion": "ft.hygon.io/v1alpha1", "kind": "FaultEvent",
                             "metadata": {"name": name, "namespace": ns},
                             "spec": {"nodeName": primary, "podNamespace": ns, "podName": pod["metadata"]["name"],
                                      "jobName": name, "workloadKind": "VolcanoJob", "type": "NodeHealthCheckFailed",
                                      "faultClass": "explicit_node", "source": "crossnode-st-simulation", "severity": "Critical",
                                      "observedAt": datetime.now(timezone.utc).isoformat(),
                                      "reason": "Scoped cross-node checkpoint recovery simulation",
                                      "action": {"taintNode": True, "deletePod": True, "deletePods": False}}}
                    created_fault = request("POST", fault_path, json=fault)
                    receipt["injection"] = {"fault_event_uid": created_fault["metadata"]["uid"], "node": primary,
                                            "sent_epoch": time.time(), "kind": "simulated NodeHealthCheckFailed"}
                    receipt["fault_injected"] = True
                    save()
                    print(json.dumps({"injected": "FaultEvent", "node": primary, "pod_uid": uid,
                                      "saved_iteration": max(parsed["global_saves"])}), flush=True)
                elif uid != receipt["initial_pod_uid"] and not receipt["recovery_observed"]:
                    assert pod["spec"]["nodeName"] == standby, "replacement did not migrate to the standby node"
                    parsed = observed.get("parsed")
                    if not (parsed and parsed["loads"] and parsed["load_progress"] and complete_workers(observed)):
                        continue
                    loaded = parsed["loads"][-1]
                    assert PurePosixPath(loaded["path"]).is_relative_to(PurePosixPath(case) / "checkpoints/global")
                    assert observed["current"]["attempt_id"] != receipt["before_injection"]["current"]["attempt_id"]
                    if loaded["iteration"] > 0 and max(parsed["load_progress"]) > loaded["iteration"]:
                        receipt.update(recovery_observed=True, recovered_pod_uid=uid, recovered=observed)
                        save()
                        print(json.dumps({"recovered": True, "pod_uid": uid, "loaded_iteration": loaded["iteration"],
                                          "continued_to": max(parsed["load_progress"]), "workers": len(observed["workers"])}), flush=True)
            save()
            if phase == "Completed":
                assert receipt["fault_injected"] and receipt["recovery_observed"], "Job finished without proving injected recovery"
                actions = receipt["fault_event"].get("status", {}).get("actions", [])
                assert receipt["fault_event"].get("status", {}).get("processed") is True
                assert all(any(a["type"] == kind and a.get("success") is True for a in actions) for kind in ("TaintNode", "DeletePod"))
                receipt["completion_pods"] = [{"uid": p["metadata"]["uid"], "status": p.get("status", {})} for p in pods]
                receipt["status"] = "PASS"
                print("CROSS_NODE_RECOVERY_PASSED", flush=True)
                break
            if phase in ("Failed", "Terminated", "Aborted"):
                raise RuntimeError("Volcano Job ended before recovery: " + str(current.get("status")))
            time.sleep(1)
        else:
            raise RuntimeError("bounded worker recovery wait expired")
    except BaseException as error:
        receipt.update(status="FAIL", error=str(error) or type(error).__name__)
        raise
    finally:
        if operator is not None:
            operator.terminate()
            operator.wait(timeout=15)
        if operator_log is not None:
            operator_log.close()
        try:
            cleanup_owned(session, server, receipt, evidence)
        except Exception as error:
            receipt.update(status="FAIL", cleanup_error=str(error))
            raise
        finally:
            save()
        print(json.dumps({"cleanup_confirmed": receipt.get("cleanup_confirmed", False), "evidence": str(evidence)}), flush=True)
