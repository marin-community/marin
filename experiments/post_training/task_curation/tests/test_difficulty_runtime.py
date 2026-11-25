# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the native client's advertised context inside the production worker launch."""

import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import httpx
import pytest
from finestore.rl.difficulty import Solver
from iris.rpc import job_pb2
from iris.rpc.proto_display import priority_band_value
from marin.inference.vllm_backend import vllm_launcher
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.difficulty import runtime as difficulty_runtime
from experiments.post_training.task_curation.difficulty_config import CandidateRun
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.difficulty_sampling import sampling_settings
from experiments.post_training.task_curation.judges import JUDGE_TIERS
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

    monkeypatch.setattr(difficulty_runtime, "forwarded_secrets", lambda: {})

    def unavailable_secret(name):
        raise AssertionError(f"Judgeless worker looked up {name}")

    monkeypatch.setattr(difficulty_runtime, "require_secret", unavailable_secret)
    bundle_secrets = tmp_path / "job-bundle"
    bundle_secrets.mkdir()
    (bundle_secrets / difficulty_runtime.TOGETHER_API_KEY_ENV).write_text("fixture-provider-key")
    monkeypatch.setenv(difficulty_runtime.SECRETS_DIR_ENV, str(bundle_secrets))
    monkeypatch.setattr(difficulty_runtime.subprocess, "run", native_worker)
    try:
        difficulty_runtime.run_trial_worker(
            tmp_path / "prepared.json",
            tmp_path / "candidate.json",
            tmp_path,
            StoragePath(str(tmp_path / "output")),
            "fixture",
            ledger=str(tmp_path / "ledger"),
            action_manifest=str(tmp_path / "action.json"),
            judge=None,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert results == [[65536, 16384]]
    forwarded = tmp_path / "framework-secrets" / difficulty_runtime.TOGETHER_API_KEY_ENV
    # The judgeless worker forwards the job bundle's provider credential file for
    # the dispatch scan without resolving it in this process.
    assert forwarded.is_file() and forwarded.read_text() == "fixture-provider-key"


def test_worker_receives_daytona_connection_through_secret_files(tmp_path, monkeypatch):
    monkeypatch.delenv("DAYTONA_DATA_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_URL", "https://app.daytona.io/api")
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setenv(difficulty_runtime.SECRETS_DIR_ENV, str(bundle))
    real_run = subprocess.run
    connections = []

    def worker(command, *, check, env):
        assert "DAYTONA_DATA_API_KEY" not in env
        program = """
import json, os
from rigging.secrets import require_secret
from harbor.host_capabilities import host_capabilities_from_environment
capabilities = host_capabilities_from_environment({
    **os.environ, 'DAYTONA_API_KEY': require_secret('DAYTONA_DATA_API_KEY')}, model_authentication={})
connection = capabilities.daytona
connection.client_factory()
print(json.dumps([connection.api_key, connection.api_url, connection.target]))
"""
        result = real_run(
            [*command[: command.index("python") + 1], "-c", program],
            check=check,
            capture_output=True,
            text=True,
            env=env,
        )
        connections.append(json.loads(result.stdout))

    monkeypatch.setattr(difficulty_runtime.subprocess, "run", worker)
    difficulty_runtime.run_trial_worker(
        tmp_path / "prepared.json",
        tmp_path / "candidate.json",
        tmp_path,
        StoragePath(str(tmp_path / "output")),
        "fixture-candidate",
        None,
        "ledger",
        "action",
        daytona_api_key="fixture-data",
    )
    assert connections == [["fixture-data", "https://app.daytona.io/api", "us"]]
    assert (tmp_path / "framework-secrets" / "DAYTONA_DATA_API_KEY").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("pooled_snowball", [False, True])
@pytest.mark.parametrize("judge_model", [None, difficulty_runtime.GLM_MODEL, "Qwen/Qwen3.5-9B"])
@pytest.mark.parametrize("tpu_campaign", [False, True])
def test_campaign_qualifies_only_required_judge_routes(
    tmp_path, monkeypatch, difficulty_source, judge_model, pooled_snowball, tpu_campaign
):
    candidate_solver = Solver.SNOWBALL if pooled_snowball else Solver.SMALL
    with_judge = judge_model is not None
    pooled = pooled_snowball or tpu_campaign
    task = TaskSpec.model_validate(json.loads(prebuilt_task("selected-task")["lowered_task_spec"])["task"])
    if with_judge:
        task = task.model_copy(
            update={"grader": verifyit_package(JudgeSpec(model=judge_model, references=("answer",))).grader}
        )
    source = difficulty_source(task)
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps([source.model_dump(mode="json")]))
    template = tmp_path / "template.jinja"
    template.write_text("Fixture chat template")
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    for solver in (candidate_solver, Solver.HOSTED):
        monkeypatch.setitem(
            difficulty_runtime.PINS, solver, replace(difficulty_runtime.PINS[solver], template_sha256=digest)
        )
    downloads, probes, secrets, launches = [], [], [], []

    def download(model, *args, **kwargs):
        downloads.append(model)
        return str(template)

    def limits(root):
        (root / "worker-cgroup.json").write_text("{}")
        return {}

    @contextmanager
    def endpoint(solver, settings, priority):
        yield "https://solver.fixture/v1", "fixture-small", "fixture-key", []

    def probe(solver, endpoint, api_model, api_key, settings):
        probes.append(solver)
        return {"tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}}}

    declared_probes = []

    def declared_probe(endpoint, api_model, api_key, temperature, pin):
        declared_probes.append(api_model)
        return {"tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}}}

    def secret(name):
        secrets.append(name)
        assert with_judge and name == difficulty_runtime.TOGETHER_API_KEY_ENV
        return "fixture-judge-key"

    def launch(
        prepared_path, config_path, local_root, output, api_key, judge, ledger, action_manifest, daytona_api_key=None
    ):
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
                judge_model=difficulty_runtime.SUBSTITUTE_JUDGE_MODEL if tpu_campaign and with_judge else judge_model,
            )
        ],
    )
    monkeypatch.setattr(difficulty_runtime, "hf_hub_download", download)
    monkeypatch.setattr(difficulty_runtime, "worker_limits", limits)
    monkeypatch.setattr(difficulty_runtime, "prepare_source", lambda source, root, **kwargs: prepared)
    if tpu_campaign:
        monkeypatch.setattr(difficulty_runtime, "campaign_record", lambda ledger: {"campaign_id": "fixture-tpu-version"})
    monkeypatch.setattr(difficulty_runtime, "solver_endpoint", endpoint)
    monkeypatch.setattr(difficulty_runtime, "probe_endpoint", probe)
    monkeypatch.setattr(difficulty_runtime, "probe_declared_judge_endpoint", declared_probe)
    monkeypatch.setattr(difficulty_runtime, "require_secret", secret)
    monkeypatch.setattr(difficulty_runtime, "run_trial_worker", launch)
    if pooled:
        monkeypatch.setattr(difficulty_runtime, "solver_pool_record", lambda solver, registry: {"effective_serving": {}})
        monkeypatch.setattr(
            difficulty_runtime,
            "solver_pool_connection",
            lambda record, *args: difficulty_runtime.OpenAIEndpoint(
                base_url="https://solver.fixture/v1", model="fixture-snowball"
            ),
        )
    manifest = tmp_path / "action.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [{"source_id": source.sample.source_id, "tasks": [[task.id, "a" * 64, 1]]}],
                "vllm_release_tag": (
                    difficulty_runtime.TPU_SERVING_ID
                    if tpu_campaign
                    else difficulty_runtime.VLLM_GPU_RELEASE.release_tag
                ),
            }
        )
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "difficulty_runtime",
            "--solver",
            candidate_solver.value,
            *(["--solver-pool", "s3://fixtures/pool"] if pooled else []),
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
            "--priority",
            "interactive",
            "--ledger",
            str(tmp_path / "ledger"),
            "--action-manifest",
            str(manifest),
        ],
    )
    difficulty_runtime.main()
    declared = judge_model in difficulty_runtime.DECLARED_JUDGE_PINS and not tpu_campaign
    expected_declared = (
        [difficulty_runtime.SUBSTITUTE_JUDGE_MODEL] if tpu_campaign and with_judge else [judge_model] if declared else []
    )
    assert declared_probes == expected_declared
    assert probes == (
        [candidate_solver, Solver.HOSTED] if with_judge and not declared and not tpu_campaign else [candidate_solver]
    )
    expected_solvers = {candidate_solver, Solver.HOSTED} if with_judge else {candidate_solver}
    assert set(downloads) == {difficulty_runtime.SOLVER_MODELS[solver] for solver in expected_solvers}
    assert secrets == ([difficulty_runtime.TOGETHER_API_KEY_ENV] if with_judge else [])
    (config,) = launches
    serving = json.loads((output / "serving.json").read_text())
    if tpu_campaign:
        assert serving["placement"]["root_target_cluster"] == "marin"
        assert serving["placement"]["model_cache_root"] == "gs://marin-us-central2"
        assert serving["placement"]["child_region_constraint"] == "us-central2"
        assert serving["compilation_requested"] and not serving["cuda_graphs_requested"]
        assert config.gpu_count == 0
        if with_judge:
            assert config.judge.identity.revision == "opaque-provider"
            assert serving["judge"]["judge_version"] == difficulty_runtime.JUDGE_VERSION
    if with_judge:
        assert config.judge.provider_limits.request_timeout == (30 if declared else 600)
        assert serving["judge_provider_limits"] == {
            "max_in_flight": config.judge.provider_limits.max_in_flight,
            "attempts": config.judge.provider_limits.attempts,
            "request_timeout": config.judge.provider_limits.request_timeout,
        }
        assert serving["judge"]["revision"] == config.judge.identity.revision
    else:
        assert config.judge is serving["judge"] is serving["judge_provider_limits"] is None


@pytest.mark.parametrize("solver", [Solver.SMALL, Solver.SNOWBALL])
def test_solver_endpoint_passes_the_declared_band_and_release_to_serving_children(monkeypatch, solver):
    captured = []

    @contextmanager
    def remote_inference(config):
        captured.append(config)
        endpoint = SimpleNamespace(base_url="http://solver.fixture/v1", model="fixture-model", api_key="fixture-key")
        session = SimpleNamespace(
            ready_direct_session=lambda: SimpleNamespace(
                effective_serving=SimpleNamespace(max_model_len=settings.context_tokens),
                model=SimpleNamespace(endpoint=endpoint),
            ),
            jobs=[SimpleNamespace(job_id="fixture-job")],
        )
        yield session

    monkeypatch.setattr(difficulty_runtime, "remote_inference", remote_inference)
    monkeypatch.setattr("marin.evaluation.serving_config._checkpoint_file_sizes", lambda *args: ())
    monkeypatch.setattr("marin.inference.vllm_release.platform.machine", lambda: "x86_64")
    settings = sampling_settings(solver, [])
    with difficulty_runtime.solver_endpoint(solver, settings, priority_band_value("interactive")) as (
        endpoint,
        _api_model,
        _api_key,
        _jobs,
    ):
        assert endpoint.startswith("http")
    assert len(captured) == 1
    assert captured[0].iris.priority == job_pb2.PRIORITY_BAND_INTERACTIVE
    assert captured[0].iris.worker_timeout == Duration.from_hours(72)
    assert captured[0].engine.startup_timeout_seconds == 259200
    assert captured[0].iris.endpoint_ready_timeout_seconds == 259200
    if solver == Solver.SNOWBALL:
        launcher = vllm_launcher(captured[0].engine)
        command = launcher.command()
        provenance = json.loads(command[-1])
        assert provenance["source_commit"] == "01911be34fac8715347962a8d791ca34a879e01e"
        assert provenance["wheel_sha256"] == "d04fa6af4a7161f8995154cbdbbc8dddc61b9a9ff198b3947985b2d998ef60e0"
        requirements = [value for index, value in enumerate(command) if index and command[index - 1] == "--with"]
        assert "torch==2.13.0+cu132" in requirements
        assert provenance["wheel_url"] in launcher.cache_identity()


def test_every_judge_tier_carries_an_immutable_checkpoint_pin():
    """Local hosting probes an immutable model revision; a tier without one cannot serve."""
    for tier, definition in JUDGE_TIERS.items():
        assert definition.model_revision, f"{tier.value} has no immutable checkpoint pin"
        assert len(definition.model_revision) >= 40, f"{tier.value} pins a shortened revision"


def test_hosted_endpoint_liveness_fails_fast_and_sends_the_hosted_model(monkeypatch):
    sent = {}

    def dead_post(self, url, **kwargs):
        sent["url"] = url
        sent["model"] = kwargs["json"]["model"]
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.Client, "post", dead_post)
    monkeypatch.setattr(difficulty_runtime, "TOGETHER_BASE_URL", "https://hosted.fixture/v1")
    monkeypatch.setattr(difficulty_runtime, "require_secret", lambda name: "fixture-key")
    try:
        with difficulty_runtime.solver_endpoint(Solver.HOSTED, sampling_settings(Solver.HOSTED, []), 0):
            raise AssertionError("The dead endpoint must fail the endpoint context")
    except RuntimeError as error:
        assert "liveness" in str(error) and "hosted.fixture" in str(error)
    assert sent["model"] == difficulty_runtime.TOGETHER_MODEL


def test_campaign_refuses_daytona_sources_without_the_connection_environment(monkeypatch):
    """The Daytona connection resolves from the worker's own environment at the
    first Daytona trial (only the API url defaults), so the campaign requires
    every name up front instead of dying hours into the run (large-v22-relay)."""
    monkeypatch.setattr(difficulty_runtime.os, "environ", {"DAYTONA_API_KEY": "fixture-key"})
    with pytest.raises(ValueError, match="DAYTONA_API_URL, DAYTONA_TARGET"):
        difficulty_runtime.require_daytona_connection_environment("fixture-key")

    monkeypatch.setattr(
        difficulty_runtime.os,
        "environ",
        {"DAYTONA_API_URL": "https://app.daytona.io/api", "DAYTONA_TARGET": "us"},
    )
    with pytest.raises(ValueError, match="DAYTONA_DATA_API_KEY"):
        difficulty_runtime.require_daytona_connection_environment(None)

    monkeypatch.setattr(
        difficulty_runtime.os,
        "environ",
        {
            "DAYTONA_API_KEY": "fixture-key",
            "DAYTONA_API_URL": "https://app.daytona.io/api",
            "DAYTONA_TARGET": "us",
        },
    )
    difficulty_runtime.require_daytona_connection_environment("fixture-key")


def _declared_judge_packet(tmp_path, judge: str = "together_ai/Qwen/Qwen3.5-9B"):
    directory = tmp_path / "packet"
    directory.mkdir()
    (directory / "task.toml").write_text(
        "[verifier]\n"
        "timeout_sec = 600.0\n"
        f'env = {{ REWARDKIT_JUDGE = "{judge}", TOGETHER_API_KEY = "${{TOGETHER_API_KEY}}" }}\n'
    )
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    return directory


def _fixture_source(difficulty_source):
    task_spec = TaskSpec.model_validate(json.loads(prebuilt_task("selected-task")["lowered_task_spec"])["task"])
    return difficulty_source(task_spec)


def _prepared_with_tasks(source, tasks):
    return PreparedSource(sample=source.sample, sample_receipt=source.sample_receipt, tasks=tasks)


def _native_task(path, **updates):
    task = PreparedTask(
        task_id="declared-task",
        runtime_sha256="a" * 64,
        path=str(path),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=None,
    )
    return task.model_copy(update=updates) if updates else task


def test_declared_judge_serves_its_own_shared_route(tmp_path, difficulty_source):
    source = _prepared_with_tasks(_fixture_source(difficulty_source), [_native_task(_declared_judge_packet(tmp_path))])
    assert difficulty_runtime.batch_judge_route([source]) == (None, "Qwen/Qwen3.5-9B")


def test_substitute_judge_route_serves_its_own_shared_route(tmp_path, difficulty_source):
    """The standing-ruling substitute is a declared-judge-class route: its own queue, one model."""
    source = _prepared_with_tasks(
        _fixture_source(difficulty_source),
        [_native_task(_declared_judge_packet(tmp_path), judge_model=difficulty_runtime.SUBSTITUTE_JUDGE_MODEL)],
    )
    assert difficulty_runtime.batch_judge_route([source]) == (None, "zai-org/GLM-5.3-Flash")


def test_substitute_judge_cannot_share_the_glm_queue(tmp_path, difficulty_source):
    packet = _declared_judge_packet(tmp_path)
    source = _prepared_with_tasks(
        _fixture_source(difficulty_source),
        [
            _native_task(packet, judge_model=difficulty_runtime.SUBSTITUTE_JUDGE_MODEL),
            _native_task(packet, judge_model=difficulty_runtime.GLM_MODEL),
        ],
    )
    with pytest.raises(ValueError, match="own queue"):
        difficulty_runtime.batch_judge_route([source])


def test_glm_judged_route_is_unchanged(tmp_path, difficulty_source):
    source = _prepared_with_tasks(
        _fixture_source(difficulty_source),
        [_native_task(_declared_judge_packet(tmp_path), judge_model=difficulty_runtime.GLM_MODEL)],
    )
    assert difficulty_runtime.batch_judge_route([source]) == (None, None)


def test_hosted_tier_route_is_unchanged(tmp_path, difficulty_source):
    tier_model = next(iter(JUDGE_TIERS.values())).spec.model
    source = _prepared_with_tasks(
        _fixture_source(difficulty_source),
        [_native_task(_declared_judge_packet(tmp_path), judge_model=tier_model)],
    )
    tier, shared = difficulty_runtime.batch_judge_route([source])
    assert shared is None and tier is not None and JUDGE_TIERS[tier].spec.model == tier_model


def test_declared_judge_cannot_share_the_glm_queue(tmp_path, difficulty_source):
    packet = _declared_judge_packet(tmp_path)
    source = _prepared_with_tasks(
        _fixture_source(difficulty_source),
        [
            _native_task(packet),
            _native_task(packet, judge_model=difficulty_runtime.GLM_MODEL),
        ],
    )
    with pytest.raises(ValueError, match="own queue"):
        difficulty_runtime.batch_judge_route([source])


def test_unqualified_judge_model_fails_the_queue(tmp_path, difficulty_source):
    source = _prepared_with_tasks(
        _fixture_source(difficulty_source),
        [_native_task(_declared_judge_packet(tmp_path, "together_ai/Qwen/Unqualified-9B"))],
    )
    with pytest.raises(ValueError, match="has not qualified"):
        difficulty_runtime.batch_judge_route([source])


def test_runtime_imports_without_harbor():
    """The controller and runtime environments have no Harbor package; the import graph must not need it.

    A declared-judge helper briefly imported difficulty_trials from the runtime,
    which imports Harbor, and every campaign crashed before submission. This
    regression imports the runtime in a fresh interpreter with Harbor forbidden.
    """
    program = (
        "import sys\n"
        "class _HarborBlocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'harbor' or name.startswith('harbor.'):\n"
        "            return self\n"
        "    def load_module(self, name):\n"
        "        raise ModuleNotFoundError(f'{name} is forbidden in the base environment')\n"
        "sys.meta_path.insert(0, _HarborBlocker())\n"
        "import experiments.post_training.task_curation.difficulty.runtime\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True, cwd=Path(__file__).parents[4]
    )
    assert result.stdout.strip() == "ok"
