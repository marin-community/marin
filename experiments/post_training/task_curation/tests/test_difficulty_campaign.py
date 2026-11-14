# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the native client's advertised context inside the production worker launch."""

import hashlib
import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation import difficulty_campaign
from experiments.post_training.task_curation.difficulty_config import CandidateRun
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


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

    def unavailable_secret(name):
        raise AssertionError(f"Judgeless worker looked up {name}")

    monkeypatch.setattr(difficulty_campaign, "require_secret", unavailable_secret)
    monkeypatch.setattr(difficulty_campaign.subprocess, "run", native_worker)
    try:
        difficulty_campaign.run_trial_worker(
            tmp_path / "prepared.json",
            tmp_path / "candidate.json",
            tmp_path,
            StoragePath(str(tmp_path / "output")),
            "fixture",
            recovery_arguments=[],
            judge=None,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert results == [[65536, 16384]]


@pytest.mark.parametrize("with_judge", [False, True])
def test_campaign_qualifies_only_required_judge_routes(tmp_path, monkeypatch, difficulty_source, with_judge):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("selected-task")["lowered_task_spec"])["task"])
    if with_judge:
        task = task.model_copy(
            update={
                "grader": verifyit_package(JudgeSpec(model=difficulty_campaign.GLM_MODEL, references=("answer",))).grader
            }
        )
    source = difficulty_source(task)
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps([source.model_dump(mode="json")]))
    template = tmp_path / "template.jinja"
    template.write_text("Fixture chat template")
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    for solver in (Solver.SMALL, Solver.HOSTED):
        monkeypatch.setitem(
            difficulty_campaign.PINS, solver, replace(difficulty_campaign.PINS[solver], template_sha256=digest)
        )
    downloads, probes, secrets, launches = [], [], [], []

    def download(model, *args, **kwargs):
        downloads.append(model)
        return str(template)

    def limits(root):
        (root / "worker-cgroup.json").write_text("{}")
        return {}

    @contextmanager
    def endpoint(solver, settings):
        yield "https://solver.fixture/v1", "fixture-small", "fixture-key", []

    def probe(solver, endpoint, api_model, api_key, settings):
        probes.append(solver)
        return {"tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}}}

    def secret(name):
        secrets.append(name)
        assert with_judge and name == difficulty_campaign.GLM_BULK_TOKEN_ENV
        return "fixture-judge-key"

    def launch(prepared_path, config_path, local_root, output, api_key, recovery_arguments, judge):
        config = CandidateRun.model_validate_json(config_path.read_text())
        assert config.judge == judge
        launches.append(config)

    prepared = PreparedSource(
        sample=source.sample,
        sample_receipt=source.sample_receipt,
        tasks=[
            PreparedTask(
                task_id=task.id,
                runtime_sha256="a" * 64,
                path=str(tmp_path / "packet"),
                workload="task_spec",
                backend="local",
                reason=None,
                judge_model=difficulty_campaign.GLM_MODEL if with_judge else None,
            )
        ],
    )
    monkeypatch.setattr(difficulty_campaign, "hf_hub_download", download)
    monkeypatch.setattr(difficulty_campaign, "worker_limits", limits)
    monkeypatch.setattr(difficulty_campaign, "prepare_source", lambda source, root: prepared)
    monkeypatch.setattr(difficulty_campaign, "solver_endpoint", endpoint)
    monkeypatch.setattr(difficulty_campaign, "probe_endpoint", probe)
    monkeypatch.setattr(difficulty_campaign, "require_secret", secret)
    monkeypatch.setattr(difficulty_campaign, "resolve_glm_base_url", lambda job: "https://judge.fixture/v1")
    monkeypatch.setattr(difficulty_campaign, "run_trial_worker", launch)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "difficulty_campaign",
            "--solver",
            "small",
            "--sources",
            str(source_path),
            "--sources-sha256",
            hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "--output-root",
            str(output),
            "--local-root",
            str(tmp_path / "run"),
            "--workers",
            "2",
            "--attempts-per-task",
            "3",
        ],
    )
    difficulty_campaign.main()
    assert probes == ([Solver.SMALL, Solver.HOSTED] if with_judge else [Solver.SMALL])
    assert set(downloads) == {difficulty_campaign.SOLVER_MODELS[solver] for solver in probes}
    assert secrets == ([difficulty_campaign.GLM_BULK_TOKEN_ENV] if with_judge else [])
    (config,) = launches
    serving = json.loads((output / "serving.json").read_text())
    if with_judge:
        assert serving["judge_provider_limits"] == {
            "max_in_flight": config.judge.provider_limits.max_in_flight,
            "attempts": config.judge.provider_limits.attempts,
        }
        assert serving["judge"]["revision"] == config.judge.identity.revision
    else:
        assert config.judge is serving["judge"] is serving["judge_provider_limits"] is None
