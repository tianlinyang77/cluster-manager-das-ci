# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes authentication and exec transport for owned ST Pods."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode

import websocket


@contextmanager
def kube_api():
    import requests
    from kubernetes import client, config

    if os.environ.get("KUBECONFIG"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    cfg = client.Configuration.get_default_copy()
    with requests.Session() as session:
        session.trust_env = False
        session.verify = cfg.ssl_ca_cert if cfg.verify_ssl else False
        if cfg.cert_file and cfg.key_file:
            session.cert = cfg.cert_file, cfg.key_file
        token = cfg.get_api_key_with_prefix("authorization")
        if token:
            session.headers["Authorization"] = token
        yield session, cfg.host.rstrip("/")


def exec_in_pod(session, server, namespace, pod, container, command, timeout=30):
    query = urlencode({"container": container, "command": command, "stdout": "true", "stderr": "true", "stdin": "false", "tty": "false"}, doseq=True)
    url = server.replace("https://", "wss://") + f"/api/v1/namespaces/{namespace}/pods/{pod}/exec?" + query
    tls = {"ca_certs": session.verify}
    if session.cert:
        tls.update(certfile=session.cert[0], keyfile=session.cert[1])
    headers = ["Authorization: " + session.headers["Authorization"]] if "Authorization" in session.headers else []
    connection = websocket.create_connection(url, subprotocols=["v4.channel.k8s.io"], header=headers, sslopt=tls, timeout=timeout)
    output, errors, status = [], [], {}
    try:
        while True:
            data = connection.recv()
            if not data:
                break
            data = data.encode() if isinstance(data, str) else data
            channel, text = data[0], data[1:].decode("utf-8", "replace")
            if channel == 1:
                output.append(text)
            elif channel == 2:
                errors.append(text)
            elif channel == 3:
                status = json.loads(text)
    finally:
        connection.close()
    return {"stdout": "".join(output), "stderr": "".join(errors), "status": status}


def cleanup_owned(session, server, receipt, evidence):
    """Remove the recorded objects and exact test taint; usable after cancellation."""
    def request(method, path, **kwargs):
        response = session.request(method, server + path, timeout=(5, 25), **kwargs)
        if method == "DELETE" and response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()

    name, ns = receipt["name"], receipt["namespace"]
    node_path = "/api/v1/nodes/" + receipt["primary_node"]
    node = request("GET", node_path)
    for index, taint in enumerate(node.get("spec", {}).get("taints", [])):
        if taint["key"] == receipt["taint_key"]:
            request("PATCH", node_path, headers={"Content-Type": "application/json-patch+json"},
                    json=[{"op": "test", "path": f"/spec/taints/{index}", "value": taint},
                          {"op": "remove", "path": f"/spec/taints/{index}"}])
            break
    assert all(t["key"] != receipt["taint_key"] for t in request("GET", node_path).get("spec", {}).get("taints", []))
    receipt["node_taint_removed"] = True
    jobs = f"/apis/batch.volcano.sh/v1alpha1/namespaces/{ns}/jobs"
    pods = f"/api/v1/namespaces/{ns}/pods"
    cm = f"/api/v1/namespaces/{ns}/configmaps/{name}"
    crd = "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/faultevents.ft.hygon.io"
    fault = f"/apis/ft.hygon.io/v1alpha1/namespaces/{ns}/faultevents/{name}"
    if receipt.get("job_uid"):
        events = request("GET", f"/api/v1/namespaces/{ns}/events", params={"fieldSelector": "involvedObject.uid=" + receipt["job_uid"]})
        (Path(evidence) / "job-events.json").write_text(json.dumps(events["items"], indent=2), encoding="utf-8")
        request("DELETE", jobs + "/" + name, json={"propagationPolicy": "Foreground", "preconditions": {"uid": receipt["job_uid"]}})
    if receipt.get("configmap_uid"):
        request("DELETE", cm, json={"preconditions": {"uid": receipt["configmap_uid"]}})
    if receipt.get("injection", {}).get("fault_event_uid"):
        request("DELETE", fault, json={"preconditions": {"uid": receipt["injection"]["fault_event_uid"]}})
    if receipt.get("created_crd_uid") and session.get(server + crd, timeout=(5, 20)).status_code != 404:
        if not request("GET", "/apis/ft.hygon.io/v1alpha1/faultevents")["items"]:
            request("DELETE", crd, json={"preconditions": {"uid": receipt["created_crd_uid"]}})
            receipt["test_crd_delete_requested"] = True
    for _ in range(45):
        job_absent = session.get(server + jobs + "/" + name, timeout=(5, 20)).status_code == 404
        cm_absent = session.get(server + cm, timeout=(5, 20)).status_code == 404
        if job_absent and cm_absent and not request("GET", pods, params={"labelSelector": "st-run=" + name})["items"]:
            receipt["cleanup_confirmed"] = True
            return
        time.sleep(1)
    raise RuntimeError("owned resources were not confirmed removed")
