# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CI input boundaries and the exact API operations used by compensating cleanup."""
import importlib.util
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
import sys
import subprocess
import threading

import pytest
import requests

SCRIPTS = Path(__file__).resolve().parents[3] / ".github/ci/k8s"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("k8s_prepare", SCRIPTS / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
import transport
from transport import cleanup_owned, finalize_run


def environment(tmp_path):
    return dict(ST_K8S_NODES="node-a node-b", ST_TRAINING_IMAGE="registry.example.org/train@sha256:" + "a" * 64,
                ST_SHARED_ROOT=str(tmp_path / "shared"), ST_RUN_KEY="123-1")


@pytest.mark.parametrize("field,value", [
    ("ST_K8S_NODES", ""), ("ST_K8S_NODES", "node-a"), ("ST_K8S_NODES", "node-a node-a"),
    ("ST_K8S_NODES", "node-a user@node-b"), ("ST_K8S_NODES", "node-a node-b node-c"),
    ("ST_TRAINING_IMAGE", "registry.example.org/train:latest"), ("ST_RUN_KEY", "../escape"),
    ("ST_SHARED_ROOT", "relative/path"),
])
def test_invalid_ci_inputs_are_rejected_without_cluster_access(tmp_path, field, value):
    env = environment(tmp_path)
    env[field] = value
    with pytest.raises(ValueError):
        prepare.validate_inputs(env)
    assert not (tmp_path / "shared").exists()


def test_explicit_fault_and_standby_order_is_preserved(tmp_path):
    nodes, image, shared, key = prepare.validate_inputs(environment(tmp_path))
    assert nodes == ["node-a", "node-b"] and key == "123-1"
    assert image.endswith("a" * 64) and shared == tmp_path / "shared"


def test_default_linux_shared_directory_is_accepted(tmp_path, monkeypatch):
    env = environment(tmp_path)
    env["ST_SHARED_ROOT"] = "/ci_public/cluster-manager-st"
    monkeypatch.setattr(prepare, "Path", PurePosixPath)
    assert str(prepare.validate_inputs(env)[2]) == env["ST_SHARED_ROOT"]


@pytest.mark.parametrize("root_link", [False, True])
def test_source_staging_does_not_follow_links_into_controller_files(tmp_path, root_link):
    source = tmp_path / "checkout"
    (source / "tests").mkdir(parents=True)
    external = tmp_path / "controller-files"
    external.mkdir()
    (external / "credential").write_text("test fixture")
    origin = source / "tests/st"
    try:
        if root_link:
            origin.symlink_to(external, target_is_directory=True)
        else:
            origin.mkdir()
            (origin / "credential").symlink_to(external / "credential")
    except OSError:
        pytest.skip("symlink creation requires local OS permission")
    destination = tmp_path / "staged"
    if root_link:
        with pytest.raises(ValueError, match="inside"):
            prepare.stage_source(source, destination)
        assert not destination.exists()
    else:
        prepare.stage_source(source, destination)
        assert (destination / "credential").is_symlink()


def test_degraded_override_does_not_bypass_readiness_or_cordon():
    node = dict(metadata=dict(name="node-a", labels={"ci.hygon.cn/hcu-health": "degraded"}), spec={},
                status=dict(allocatable={"hygon.com/hcu": "8"}, conditions=[dict(type="Ready", status="True")]))
    with pytest.raises(ValueError, match="explicit"):
        prepare.validate_node(node, 8, False)
    prepare.validate_node(node, 8, True)
    node["spec"]["unschedulable"] = True
    with pytest.raises(ValueError, match="schedulable"):
        prepare.validate_node(node, 8, True)


def test_cleanup_uses_recorded_uids_and_removes_only_the_exact_test_taint(tmp_path):
    foreign = dict(key="other.example.org/keep", effect="NoSchedule")
    own = dict(key="st.ft.hygon.io/case-1", value="NodeHealthCheckFailed", effect="NoSchedule")
    calls = []
    patched = []

    class API(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/v1/nodes/node-a":
                self.reply(200, {"spec": {"taints": [foreign] if patched else [foreign, own]}})
            elif "/events?" in self.path or "/pods?" in self.path:
                self.reply(200, {"items": []})
            else:
                self.reply(404, {})

        def do_PATCH(self):
            patched.append(True)
            self.reply(200, {})

        def do_DELETE(self):
            self.reply(404, {})

        def reply(self, status, body):
            content = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            calls.append((self.command, self.path, json.loads(content) if content else None))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), API)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    receipt = dict(name="case-1", namespace="test-ns", primary_node="node-a", taint_key=own["key"],
                   job_uid="job-uid", configmap_uid="cm-uid", injection={"fault_event_uid": "fault-uid"})
    try:
        with requests.Session() as session:
            session.trust_env = False
            cleanup_owned(session, "http://127.0.0.1:" + str(server.server_port), receipt, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    patch = next(body for method, _, body in calls if method == "PATCH")
    assert patch == [{"op": "test", "path": "/spec/taints/1", "value": own}, {"op": "remove", "path": "/spec/taints/1"}]
    deletes = {path: body for method, path, body in calls if method == "DELETE"}
    assert deletes["/apis/batch.volcano.sh/v1alpha1/namespaces/test-ns/jobs/case-1"]["preconditions"] == {"uid": "job-uid"}
    assert deletes["/api/v1/namespaces/test-ns/configmaps/case-1"]["preconditions"] == {"uid": "cm-uid"}
    assert deletes["/apis/ft.hygon.io/v1alpha1/namespaces/test-ns/faultevents/case-1"]["preconditions"] == {"uid": "fault-uid"}
    assert receipt["cleanup_confirmed"] and receipt["node_taint_removed"]


@pytest.mark.parametrize("failure_at", ["terminate", "wait"])
def test_operator_stop_failure_still_cleans_resources_and_saves_result(tmp_path, monkeypatch, failure_at):
    class Operator:
        def terminate(self):
            if failure_at == "terminate":
                raise OSError("stop failed")

        def wait(self, timeout):
            raise OSError("stop failed")

    calls = []
    def cleanup(session, server, receipt, evidence):
        calls.append(evidence)
        receipt["cleanup_confirmed"] = True
    monkeypatch.setattr(transport, "cleanup_owned", cleanup)
    receipt = dict(status="FAIL", error="original training failure")
    log = io.StringIO()
    with pytest.raises(OSError, match="stop failed"):
        finalize_run(None, None, receipt, tmp_path, Operator(), log)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert calls == [tmp_path] and log.closed
    assert saved["cleanup_confirmed"] and saved["status"] == "FAIL"
    assert saved["error"] == "original training failure" and saved["operator_stop_error"] == "stop failed"


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX SIGTERM semantics")
def test_operator_ignoring_term_is_killed_before_cleanup(tmp_path, monkeypatch):
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    operator = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, text=True)
    cleaned = []
    def cleanup(session, server, receipt, evidence):
        assert operator.poll() is not None
        cleaned.append(True)
        receipt["cleanup_confirmed"] = True
    monkeypatch.setattr(transport, "cleanup_owned", cleanup)
    try:
        assert operator.stdout.readline().strip() == "ready"
        finalize_run(None, None, {"status": "PASS"}, tmp_path, operator, stop_timeout=0.1)
        assert operator.returncode < 0 and cleaned == [True]
    finally:
        if operator.poll() is None:
            operator.kill()
            operator.wait()
        operator.stdout.close()


def test_cleanup_waits_for_fault_and_crd_deletion(tmp_path, monkeypatch):
    fault = "/apis/ft.hygon.io/v1alpha1/namespaces/test-ns/faultevents/case-1"
    crd = "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/faultevents.ft.hygon.io"
    calls = []
    pending = {fault: 2, crd: 2}
    deleting = set()
    gone = set()

    class API(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/v1/nodes/node-a":
                self.reply(200, {"spec": {"taints": []}})
            elif self.path in pending:
                if self.path in deleting:
                    pending[self.path] -= 1
                    if pending[self.path] <= 0:
                        gone.add(self.path)
                self.reply(404 if self.path in gone else 200, {"metadata": {"deletionTimestamp": "pending"}})
            elif self.path == "/apis/ft.hygon.io/v1alpha1/faultevents":
                self.reply(200, {"items": [] if fault in gone else [{"metadata": {"uid": "fault-uid"}}]})
            elif "/events?" in self.path or "/pods?" in self.path:
                self.reply(200, {"items": []})
            else:
                self.reply(404, {})

        def do_DELETE(self):
            deleting.add(self.path)
            self.reply(200, {})

        def reply(self, status, body):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            calls.append((self.command, self.path, status))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), API)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    receipt = dict(name="case-1", namespace="test-ns", primary_node="node-a", taint_key="st.ft.hygon.io/case-1",
                   created_crd_uid="crd-uid", injection={"fault_event_uid": "fault-uid"})
    try:
        with requests.Session() as session:
            session.trust_env = False
            cleanup_owned(session, "http://127.0.0.1:" + str(server.server_port), receipt, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert fault in gone and crd in gone and receipt["cleanup_confirmed"]
    assert calls.index(("GET", fault, 404)) < calls.index(("DELETE", crd, 200))
    assert ("GET", crd, 404) in calls
