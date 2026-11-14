# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the native client's advertised context inside the production worker launch."""

import hashlib
import json
import subprocess
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

from finestore.rl.difficulty import DifficultySample, Solver
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation import difficulty_campaign
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask


def test_native_worker_keeps_full_completion_at_protocol_input_limit(tmp_path, monkeypatch):
    class Models(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.path == "/v1/models"
            body = json.dumps({"data": [{"id": "fixture", "max_model_len": 65536}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    real_run = subprocess.run
    results = []

    def native_worker(command, *, check, env):
        # Replace the full campaign boundary with its real native client in a fresh interpreter.
        program = """
import json, sys
from harbor.llms.openai_llm import OpenAILLM
from harbor.llms.vllm_tokenization import completion_budget_from_count
model = OpenAILLM('openai/fixture', api_key='fixture', api_base=sys.argv[1], model_info={
    'max_input_tokens': 65536, 'max_output_tokens': 16384})
context = model.get_model_context_limit()
print(json.dumps([context, completion_budget_from_count(
    prompt_tokens=49152, context_window=context, max_output_tokens=model.get_model_output_limit())]))
"""
        result = real_run(
            [*command[: command.index("python") + 1], "-c", program, endpoint],
            check=check,
            capture_output=True,
            text=True,
            env=env,
        )
        results.append(json.loads(result.stdout))
        return result

    monkeypatch.setattr(difficulty_campaign, "forwarded_secrets", lambda: {})
    monkeypatch.setattr(difficulty_campaign, "require_secret", lambda name: "fixture")
    monkeypatch.setattr(difficulty_campaign.subprocess, "run", native_worker)
    try:
        difficulty_campaign.run_trial_worker(
            tmp_path / "prepared.json",
            tmp_path / "candidate.json",
            tmp_path,
            StoragePath(str(tmp_path / "output")),
            "fixture",
            recovery_arguments=[],
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert results == [[65536, 16384]]


TEMPLATE_SHA256 = hashlib.sha256(b"fixture chat template").hexdigest()


def _campaign_probe_test(tmp_path, monkeypatch, judge_model):
    """Drive one campaign through patched boundaries and record every judge qualification call."""

    class StubSource:
        @classmethod
        def model_validate(cls, row):
            return row

    probes = []
    judge_urls = []
    secrets = []
    templates = []
    workers = []

    def probe(solver, *args, **kwargs):
        probes.append(solver)
        return {"tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}}}

    fake_pin = SimpleNamespace(revision="fixture-pin", template_sha256=TEMPLATE_SHA256, gpus=0)

    def fake_template(model, filename, revision):
        path = tmp_path / "template.jinja"
        path.write_bytes(b"fixture chat template")
        templates.append(model)
        return str(path)

    @contextmanager
    def fake_solver_endpoint(solver, settings):
        yield "http://solver.fixture/v1", "fixture-model", "fixture-key", [{"job": "fixture"}]

    monkeypatch.setattr(difficulty_campaign, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(difficulty_campaign, "file_sha256", lambda path: "1" * 64)
    monkeypatch.setattr(difficulty_campaign, "DifficultySource", StubSource)
    monkeypatch.setattr(
        difficulty_campaign, "worker_limits", lambda root: (root / "worker-cgroup.json").write_text("{}") or {}
    )
    monkeypatch.setattr(difficulty_campaign, "PINS", {Solver.SMALL: fake_pin, Solver.HOSTED: fake_pin})
    monkeypatch.setattr(difficulty_campaign, "hf_hub_download", fake_template)
    monkeypatch.setattr(
        difficulty_campaign,
        "prepare_source",
        lambda source, root: PreparedSource(
            sample=DifficultySample(
                source_id="fixture:probe",
                release={"repo": "fixture", "revision": "pin", "url": "https://fixture.invalid/pin"},
                converter_version="fixture",
                converter_revision="a" * 40,
                population_count=1,
                task_specs_sha256="b" * 64,
                drop_manifest={"uri": "fixture:drop", "sha256": "c" * 64},
                seed=0,
                pre_oracle_filter=False,
                resample_after_oracle_filter=False,
                tasks=[{"task_id": "probe-task", "task_sha256": "d" * 64}],
            ),
            sample_receipt={"uri": "fixture:sample", "sha256": "e" * 64},
            tasks=[
                PreparedTask(
                    task_id="probe-task",
                    runtime_sha256="0" * 64,
                    path=str(tmp_path / "task"),
                    workload="task_spec",
                    backend="local",
                    reason=None,
                    judge_model=judge_model,
                )
            ],
        ),
    )
    monkeypatch.setattr(difficulty_campaign, "solver_endpoint", fake_solver_endpoint)
    monkeypatch.setattr(difficulty_campaign, "probe_endpoint", probe)
    monkeypatch.setattr(
        difficulty_campaign, "resolve_glm_base_url", lambda job: judge_urls.append(job) or "http://judge.fixture/v1"
    )
    monkeypatch.setattr(difficulty_campaign, "require_secret", lambda name: secrets.append(name) or "fixture-secret")
    monkeypatch.setattr(
        difficulty_campaign,
        "run_trial_worker",
        lambda *args, **kwargs: workers.append(args),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "difficulty_campaign",
            "--solver",
            "small",
            "--sources",
            str(tmp_path / "sources.json"),
            "--sources-sha256",
            "1" * 64,
            "--output-root",
            str(tmp_path / "output"),
            "--local-root",
            str(tmp_path / "local"),
            "--workers",
            "1",
            "--attempts-per-task",
            "1",
        ],
    )
    (tmp_path / "sources.json").write_text("[{}]")
    difficulty_campaign.main()

    serving = json.loads((tmp_path / "output" / "serving.json").read_text())
    config = json.loads((tmp_path / "local" / "candidate.json").read_text())
    return probes, judge_urls, secrets, templates, workers, serving, config


def test_campaign_probes_judge_service_only_when_tasks_require_it(tmp_path, monkeypatch):
    result = _campaign_probe_test(tmp_path, monkeypatch, judge_model=None)
    probes, judge_urls, secrets, templates, workers, serving, config = result
    assert probes == [Solver.SMALL]
    assert judge_urls == []
    assert secrets == []
    assert templates == ["Qwen/Qwen3-Coder-30B-A3B-Instruct"]
    assert len(workers) == 1
    assert serving["judge"]["serving_probe"] is None
    assert serving["judge"]["revision"] == "unqualified:no-judge-tasks"
    assert config["judge_endpoint"] == ""
    assert config["judge_identity"]["revision"] == "unqualified:no-judge-tasks"


def test_campaign_qualifies_judge_service_when_a_task_requires_it(tmp_path, monkeypatch):
    result = _campaign_probe_test(tmp_path, monkeypatch, judge_model="")
    probes, judge_urls, secrets, templates, workers, serving, config = result
    assert probes == [Solver.SMALL, Solver.HOSTED]
    assert judge_urls == [difficulty_campaign.GLM_RELAY_JOB]
    assert secrets == [difficulty_campaign.GLM_BULK_TOKEN_ENV]
    assert len(templates) == 2
    assert len(workers) == 1
    assert serving["judge"]["serving_probe"] == {
        "tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}}
    }
    assert serving["judge"]["revision"] == "unexposed:fixture-pin"
    assert config["judge_endpoint"] == "http://judge.fixture/v1"
    assert config["judge_identity"]["revision"] == "unexposed:fixture-pin"
