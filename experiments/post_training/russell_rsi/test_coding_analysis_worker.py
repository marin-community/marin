# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import httpx
import pytest
from fray.iris_backend import FrayIrisClient
from iris.client.client import IrisClient
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY
from iris.rpc import controller_pb2, job_pb2
from marin.execution.lazy import StepContext
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi import coding_eval_feedback, test_completed_coding_analysis
from experiments.post_training.russell_rsi import completed_coding_analysis as analysis
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_worker import (
    context_preflight,
    regional_analysis_request,
    submit_regional_coding_analysis,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import coding_analysis_request
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal
from experiments.post_training.russell_rsi.interrupted_calibration import PACKAGED_PYTHONPATH, BoundedIrisClient
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV

completed_evidence = test_completed_coding_analysis.completed_evidence


class CapturedSubmission(Exception):
    pass


@contextmanager
def served_tokenizer(mode, calls):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):  # noqa: A002
            pass

        def do_GET(self):
            calls.append(self.path)
            if self.path == "/v1/models":
                self.respond(
                    {"data": [{"id": "wrong" if mode == "wrong-model" else "glm-5.3", "max_model_len": 262144}]}
                )
            else:
                self.respond({"paths": {"/v1/tokenize": {"post": {}}} if mode == "fallback" else {}})

        def do_POST(self):
            calls.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["add_generation_prompt"] is True
            assert body["messages"]
            if self.path == "/tokenize" and mode in ("fallback", "missing"):
                self.respond({}, 404)
                return
            count = 262144 if mode == "oversized" else 12
            self.respond({"count": count, "tokens": [1] * count, "max_model_len": 262144})

        def respond(self, value, status=200):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("mode", ["correct", "fallback", "wrong-model", "oversized", "missing"])
def test_exact_served_context_gate_at_http_boundary(completed_evidence, tmp_path, monkeypatch, mode):
    config, step, pin, prefix = completed_evidence
    stages = analysis.completed_evidence_analysis_stages(
        config, PinnedFile(**pin(tmp_path / "input.json", config)), step
    )
    terminal = stages["terminal"]
    bound = terminal.build_config(StepContext.for_run(str(tmp_path / "analysis"), prefix, deps=terminal.deps))
    request = coding_analysis_request(PinnedFile(**config["evidence"]).read_json(), 64, 589824)
    monkeypatch.setenv(GLM_TOKEN_ENV, "synthetic-token")
    calls = []
    with served_tokenizer(mode, calls) as base_url:
        if mode in ("correct", "fallback"):
            asyncio.run(context_preflight(bound, base_url, request))
            result = json.loads((tmp_path / "analysis/context-preflight/result.json").read_text())
            assert result["prompt_tokens"] == 12
            assert result["max_output_tokens"] == 2048
        else:
            with pytest.raises(ValueError):
                asyncio.run(context_preflight(bound, base_url, request))
            assert not (tmp_path / "analysis/context-preflight/result.json").exists()
    assert "/v1/chat/completions" not in calls
    if mode == "fallback":
        assert calls == ["/v1/models", "/tokenize", "/openapi.json", "/v1/tokenize"]
    if mode == "wrong-model":
        assert calls == ["/v1/models"]


def test_refused_connection_is_durable_before_issuance(completed_evidence, tmp_path, monkeypatch):
    config, step, pin, prefix = completed_evidence
    stages = analysis.completed_evidence_analysis_stages(
        config, PinnedFile(**pin(tmp_path / "input.json", config)), step
    )
    terminal = stages["terminal"]
    bound = terminal.build_config(StepContext.for_run(str(tmp_path / "analysis"), prefix, deps=terminal.deps))
    request = coding_analysis_request(PinnedFile(**config["evidence"]).read_json(), 64, 589824)
    monkeypatch.setenv(GLM_TOKEN_ENV, "synthetic-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    with pytest.raises(httpx.ConnectError):
        asyncio.run(context_preflight(bound, f"http://127.0.0.1:{port}/v1", request))
    record = json.loads((tmp_path / "analysis/context-preflight/models.json").read_text())
    assert record["error_type"] == "ConnectError"


def test_completed_worker_replay_and_incomplete_refusal_need_no_client(completed_evidence, tmp_path):
    config, step, pin, prefix = completed_evidence
    stages = analysis.completed_evidence_analysis_stages(
        config, PinnedFile(**pin(tmp_path / "input.json", config)), step
    )
    terminal = stages["terminal"]
    bound = terminal.build_config(StepContext.for_run(str(tmp_path / "analysis"), prefix, deps=terminal.deps))
    attempt = AttemptJournal(StoragePath(bound.analysis.output_path) / "worker-submission", asdict(bound))
    write_once(attempt.directory / "reservation.json", attempt.binding)
    with pytest.raises(RuntimeError, match="incomplete"):
        submit_regional_coding_analysis(bound)

    write_once(attempt.directory / "result.json", {"binding": attempt.binding, "result": {"worker_completed": True}})
    submit_regional_coding_analysis(bound)


def test_regional_request_serializes_bounded_cpu_worker(completed_evidence, tmp_path, monkeypatch):
    config, step, pin, prefix = completed_evidence
    stages = analysis.completed_evidence_analysis_stages(
        config, PinnedFile(**pin(tmp_path / "input.json", config)), step
    )
    terminal = stages["terminal"]
    bound = terminal.build_config(StepContext.for_run(str(tmp_path / "analysis"), prefix, deps=terminal.deps))
    monkeypatch.setenv(GLM_TOKEN_ENV, "synthetic-token")
    captured = []

    def capture(request, **kwargs):
        captured.append(controller_pb2.Controller.LaunchJobRequest.FromString(request.SerializeToString()))
        raise CapturedSubmission

    cluster = RemoteClusterClient("http://controller.invalid", bundle_id="reviewed-source")
    monkeypatch.setattr(cluster._client, "launch_job", capture)
    client = BoundedIrisClient(IrisClient(cluster))
    fray = FrayIrisClient.from_iris_client(cast(IrisClient, client))
    try:
        with pytest.raises(CapturedSubmission):
            fray.submit(regional_analysis_request(bound))
    finally:
        cluster.shutdown()
    (wire,) = captured
    assert not any(constraint.key == CLUSTER_CONSTRAINT_KEY for constraint in wire.constraints)
    assert wire.timeout.milliseconds == 20 * 60 * 1000
    assert wire.max_retries_failure == wire.max_retries_preemption == wire.max_task_failures == 0
    assert wire.priority_band == job_pb2.PRIORITY_BAND_BATCH
    assert wire.environment.env_vars["PYTHONPATH"] == PACKAGED_PYTHONPATH


def test_analyst_context_failure_precedes_issued_marker(completed_evidence, tmp_path, monkeypatch):
    config, step, pin, prefix = completed_evidence
    stages = analysis.completed_evidence_analysis_stages(
        config, PinnedFile(**pin(tmp_path / "input.json", config)), step
    )
    terminal = stages["terminal"]
    bound = terminal.build_config(StepContext.for_run(str(tmp_path / "analysis"), prefix, deps=terminal.deps))
    monkeypatch.setenv(GLM_TOKEN_ENV, "synthetic-token")
    calls = []
    with served_tokenizer("wrong-model", calls) as base_url:
        monkeypatch.setattr(coding_eval_feedback, "resolve_glm_base_url", lambda relay_job: base_url)

        async def before_issue(url, request):
            await context_preflight(bound, url, request)

        with pytest.raises(ValueError, match="exact model"):
            asyncio.run(coding_eval_feedback.analyze_coding_failures(bound.analysis, before_issue=before_issue))
    assert calls == ["/v1/models"]
    assert not (tmp_path / "analysis/private-analysis-issued.json").exists()
