# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line evaluation launch and relaunch behavior."""

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from evalrecords.records import (
    CuratedAgentRef,
    EvaluationSelection,
    RunStatus,
    TaskCoverage,
    TaskEvidence,
    TaskMachineBackend,
    TaskMachineSettings,
)
from iris.client.client import IrisClient
from iris.cluster.client import remote_client
from iris.rpc import controller_pb2, job_pb2
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import AgentConfig
from marin.evaluation.runner import (
    EvaluationBatch,
    EvaluationError,
    SubmittedEvaluation,
    SubmittedEvaluationBatch,
    _record,
)
from marin.evaluation.task_worker import DAYTONA_API_KEY, DAYTONA_EGRESS, EPISODE_HARNESS_REVISIONS
from marin.execution.step_runner import StepRunner
from marin.inference.iris import RemoteInferenceStartupError
from taskcompendium.evaluation.spec import PiAgent

import experiments.evaluation.launch as launch_module
import experiments.evaluation.policy_finalization as policy_module
from experiments.evaluation.benchmark import fixed_benchmarks
from experiments.evaluation.benchmarks import micro210
from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.cli import cli
from experiments.evaluation.curated import load_evaluations
from experiments.evaluation.evals import DAYTONA_SECRET_ENV, HarborDefinition, resolve_evaluations
from experiments.evaluation.launch import LaunchSpec, build_evaluation_steps, prepare_evaluation_batch
from experiments.evaluation.models import models
from experiments.post_training.task_curation.harbor_hub import REGISTRY_HOST
from tests.rl.micro210_fixture import policy_fixture_reference


@pytest.fixture
def submitted_batches(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "local-test-key")
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "local-test-key")
    batches: list[EvaluationBatch] = []

    class Client:
        def submit(self, *, entrypoint, **_kwargs):
            _, args, _ = entrypoint.resolve()
            batches.append(args[0])
            return SimpleNamespace(job_id="/test/eval")

    @contextmanager
    def client(**_kwargs):
        yield Client()

    monkeypatch.setattr("experiments.evaluation.cli.open_iris_client", client)
    return batches


def _command(tmp_path: Path) -> list[str]:
    return [
        "launch",
        "--model",
        "qwen3-8b",
        "--evals",
        "gsm8k-smoke,arc-easy",
        "--platform",
        "gpu",
        "--accelerator",
        "H100x1",
        "--records-prefix",
        str(tmp_path),
        "--run-id",
        "repeatable-evals",
        "--no-wait",
    ]


@pytest.fixture
def harbor_task_workspace():
    with tempfile.TemporaryDirectory(prefix=".harbor-launch-", dir=Path(__file__).parents[2]) as directory:
        root = Path(directory)
        task = root / "dataset" / "task"
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
        (task / "instruction.md").write_text("Complete the task.\n")
        (task / "task.toml").write_text(
            '[task]\nname = "test/launch"\n[environment]\n'
            f'docker_image = "example.test/task@sha256:{"a" * 64}"\n'
            'allow_internet = false\nworkdir = "/workspace"\n'
        )
        yield root


def test_launch_preflights_prepared_task_with_hosted_judge(
    tmp_path, submitted_batches, monkeypatch, harbor_task_workspace
):
    monkeypatch.setattr(launch_module, "_capability_origin", lambda _cluster: "https://iris.example")
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "TOGETHER_API_KEY", "CUSTOM_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    root = harbor_task_workspace
    task_config = root / "dataset" / "task" / "task.toml"
    task_config.write_text(
        task_config.read_text() + '[environment.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
        'OPENAI_BASE_URL = "${OPENAI_BASE_URL}"\nTAU2_USER_MODEL = "${TAU2_USER_MODEL}"\n'
        '[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
        'OPENAI_BASE_URL = "${OPENAI_BASE_URL}"\n'
        'TAU2_NL_ASSERTIONS_MODEL = "${TAU2_NL_ASSERTIONS_MODEL}"\n'
    )
    policy = root / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "environment": {"type": "local", "kwargs": {"judge_access": "runtime"}},
                "agents": [{"name": "terminus-2", "kwargs": {"llm_backend": "openai"}}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    command = [
        "launch",
        "--model",
        "qwen3-8b",
        "--harbor-config",
        str(policy),
        "--platform",
        "gpu",
        "--accelerator",
        "H100x1",
        "--federated_cluster",
        "cw-rno2a",
        "--records-prefix",
        str(tmp_path),
        "--no-wait",
    ]
    runner = CliRunner()
    rejected = runner.invoke(cli, command)
    assert rejected.exit_code != 0
    assert submitted_batches == []

    accepted = runner.invoke(cli, [*command, "--judge-model", "qwen3-0.6b", "--judge-accelerator", "H100x1"])

    assert accepted.exit_code == 0, accepted.output
    batch = submitted_batches[0]
    config = batch.evaluations[0].executor.config
    assert config.benchmark.n_benchmark == 1
    assert config.verifier_env_keys == ()
    assert json.loads(config.stable_policy_json)["environment"]["kwargs"] == {"judge_access": "runtime"}
    assert batch.judge.api_model == "qwen3-0.6b"


def test_judge_deadline_change_preserves_launch_identity_and_saved_results(tmp_path, harbor_task_workspace):
    policy = harbor_task_workspace / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "environment": {"type": "local"},
                "agents": [{"name": "terminus-2", "kwargs": {"llm_backend": "openai"}}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    judge = models()["qwen3-0.6b"]
    spec = replace(
        _pipeline_spec(tmp_path),
        evals=(HarborDefinition(policy.stem, policy),),
        judge_model=judge,
        judge_accelerator="H100x1",
        cache_key=None,
        run_id="judge-resume",
    )
    before = prepare_evaluation_batch(spec)
    identity = replace(before.evaluations[0].identity, output_dir=str(tmp_path / "saved-results"))
    record_path = Path(_record(before, identity, RunStatus.INFRA_FAILED, "interrupted", {}, {}, {}))
    record = json.loads(record_path.read_text())
    record["judge"]["model"]["config"].pop("evaluation_request_timeout", None)
    record_path.write_text(json.dumps(record))

    after = prepare_evaluation_batch(replace(spec, judge_model=replace(judge, evaluation_request_timeout=21600)))

    assert after.evaluations[0].identity.run_id == identity.run_id
    assert after.evaluations[0].identity.output_dir == identity.output_dir


@pytest.mark.parametrize("dry_run", [True, False])
def test_coreweave_launch_without_storage_credentials_fails_before_submission(
    tmp_path, submitted_batches, monkeypatch, dry_run
):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path / "no-secrets"))
    monkeypatch.delenv("CW_KEY_ID")
    command = [*_command(tmp_path), "--federated_cluster", "cw-rno2a", *(["--dry-run"] if dry_run else [])]

    result = CliRunner().invoke(cli, command)

    assert result.exit_code != 0
    assert "CW_KEY_ID" in result.output
    assert submitted_batches == []


@pytest.mark.parametrize(
    "status,exit_code",
    [(RunStatus.SUCCEEDED, 0), (RunStatus.INFRA_FAILED, 1), (None, 1)],
    ids=["succeeded", "failed-record", "no-record"],
)
def test_waiting_launch_exits_nonzero_unless_every_evaluation_succeeds(tmp_path, monkeypatch, status, exit_code):
    job = SimpleNamespace(
        job_id="/test/eval",
        wait=lambda **_kwargs: SimpleNamespace(state=job_pb2.JOB_STATE_FAILED, error="peer rejected the handoff"),
    )

    def launch(batch, _client):
        if status is not None:
            for evaluation in batch.evaluations:
                _record(batch, evaluation.identity, status, None, {}, {}, {})
        return SubmittedEvaluationBatch(
            group_id=batch.group_id,
            job=job,
            records_prefix=batch.records_prefix,
            model_name=batch.model.name,
            evaluations=tuple(
                SubmittedEvaluation(evaluation.identity.run_id, evaluation.identity.eval_ref.name)
                for evaluation in batch.evaluations
            ),
        )

    @contextmanager
    def client(**_kwargs):
        yield None

    monkeypatch.setattr("experiments.evaluation.cli.open_iris_client", client)
    monkeypatch.setattr("experiments.evaluation.cli.launch_group", launch)
    command = [argument for argument in _command(tmp_path) if argument != "--no-wait"]

    result = CliRunner().invoke(cli, command)

    assert result.exit_code == exit_code, result.output
    if status is None:
        assert "peer rejected the handoff" in result.output


def test_launch_selects_across_documents_and_adds_harbor_configs(
    tmp_path, submitted_batches, monkeypatch, harbor_task_workspace
):
    ifeval = tmp_path / "ifeval.yaml"
    ifeval.write_text("evaluations:\n  ifeval:\n    benchmark: ifeval\n")
    math = tmp_path / "math.yaml"
    math.write_text("evaluations:\n  bounded-math:\n    benchmark: math500\n  omitted:\n    benchmark: gsm8k\n")
    policies = []
    for name in ("first-policy", "second-policy"):
        policy = harbor_task_workspace / f"{name}.json"
        policy.write_text(
            json.dumps(
                {
                    "environment": {"type": "local"},
                    "agents": [{"name": "terminus-2", "kwargs": {"llm_backend": "openai"}}],
                    "datasets": [{"path": "dataset"}],
                }
            )
        )
        policies.append(policy)
    monkeypatch.setattr(launch_module, "_capability_origin", lambda _cluster: "https://iris.example")

    command = _command(tmp_path)
    command[command.index("--evals") + 1] = "ifeval,bounded-math"
    command.extend(
        [
            "--evaluation-config",
            str(ifeval),
            "--evaluation-config",
            str(math),
            "--harbor-config",
            str(policies[0]),
            "--harbor-config",
            str(policies[1]),
        ]
    )
    result = CliRunner().invoke(cli, command)

    assert result.exit_code == 0, result.output
    assert {evaluation.identity.eval_ref.name for evaluation in submitted_batches[0].evaluations} == {
        "ifeval",
        "bounded-math",
        "first-policy",
        "second-policy",
    }


def test_daytona_launch_uses_the_eval_org_key_not_an_exported_daytona_key(tmp_path, monkeypatch, harbor_task_workspace):
    monkeypatch.setenv("DAYTONA_API_KEY", "rl-org-key")
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "eval-org-key")
    monkeypatch.setattr(launch_module, "_capability_origin", lambda _cluster: "https://iris.example")
    submitted_env = []

    class Client:
        def submit(self, *, environment, **_kwargs):
            submitted_env.append(environment.env_vars)
            return SimpleNamespace(job_id="/test/eval")

    @contextmanager
    def client(**_kwargs):
        yield Client()

    monkeypatch.setattr("experiments.evaluation.cli.open_iris_client", client)
    policy = harbor_task_workspace / "daytona-policy.json"
    policy.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": "terminus-2", "kwargs": {"llm_backend": "openai"}}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    command = _command(tmp_path)
    command.extend(["--harbor-config", str(policy)])

    result = CliRunner().invoke(cli, command)

    assert result.exit_code == 0, result.output
    assert submitted_env[0]["DAYTONA_API_KEY"] == "eval-org-key"


@pytest.mark.parametrize("explicit_run_id", [True, False])
def test_relaunch_reuses_completed_steps_and_preserves_partial_archive(tmp_path, submitted_batches, explicit_run_id):
    runner = CliRunner()
    command = _command(tmp_path)
    if not explicit_run_id:
        index = command.index("--run-id")
        del command[index : index + 2]
    first = runner.invoke(cli, command)
    assert first.exit_code == 0, first.output
    batch = submitted_batches[-1]
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        path = _record(batch, evaluation.identity, RunStatus.SUCCEEDED, None, {}, {}, {})
        return {"record_path": path}

    steps = build_evaluation_steps(batch, execute)
    StepRunner().run(steps[:1])
    pending = batch.evaluations[1]
    _record(batch, pending.identity, RunStatus.ARTIFACT_FAILED, "interrupted", {}, {}, {})
    partial = Path(pending.identity.output_dir) / "saved-request.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"response":"already generated"}')
    executed.clear()

    second = runner.invoke(cli, [*command, "--run-id", batch.group_id, "--instances", "8", "--max-concurrent", "8"])
    assert second.exit_code == 0, second.output
    restored = submitted_batches[-1]
    StepRunner().run(build_evaluation_steps(restored, execute))

    assert executed == [pending.identity.run_id]
    assert restored.evaluations[1].identity.output_dir == pending.identity.output_dir
    assert partial.read_text() == '{"response":"already generated"}'


@pytest.mark.parametrize("changed_args", [["--limit", "2"], ["--model", "qwen3-0.6b"]])
def test_relaunch_changed_scoring_settings_use_distinct_steps(tmp_path, submitted_batches, changed_args):
    runner = CliRunner()
    command = _command(tmp_path)
    first = runner.invoke(cli, command)
    assert first.exit_code == 0, first.output

    changed = runner.invoke(cli, [*command, *changed_args])

    assert changed.exit_code == 0, changed.output
    original = {evaluation.identity.run_id for evaluation in submitted_batches[0].evaluations}
    revised = {evaluation.identity.run_id for evaluation in submitted_batches[1].evaluations}
    assert original.isdisjoint(revised)


@pytest.mark.parametrize("cache_key", [None, "pipeline-artifact"])
def test_relaunch_changed_request_concurrency_and_task_evidence_preserves_identity(tmp_path, cache_key):
    config = tmp_path / "custom.yaml"
    config.write_text("evaluations:\n  custom:\n    benchmark: gsm8k\n")
    model = models()["qwen3-8b"]
    spec = LaunchSpec(
        evals=(load_evaluations(config)[0],),
        model=model,
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=0,
        cache_key=cache_key,
        run_id="repeatable-evals" if cache_key is None else None,
    )
    before = prepare_evaluation_batch(spec)
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        return {"finished": evaluation.identity.run_id}

    StepRunner().run(build_evaluation_steps(before, execute))
    executed.clear()

    model = replace(model, evaluation_num_concurrent=64, evaluation_request_timeout=21600)
    tuned = replace(model.serve, max_num_batched_tokens=8192, max_num_seqs=128, gpu_memory_utilization=0.9)
    config.write_text("evaluations:\n  custom:\n    benchmark: gsm8k\n    run:\n      concurrency: 16\n")
    bounded = load_evaluations(config)[0]
    after = prepare_evaluation_batch(
        replace(spec, model=replace(model, serve=tuned), evals=(bounded,), task_evidence=TaskEvidence.REFERENCE)
    )
    StepRunner().run(build_evaluation_steps(after, execute))

    assert executed == []
    # The evaluation's bound is below the model's, so it applies.
    assert after.evaluations[0].identity.eval_ref.curated.settings.concurrency == 16


@pytest.mark.parametrize(
    "platform, launch_seed, expected",
    [
        (Platform.GPU, None, {"default": 0, "configured": 5}),
        (Platform.GPU, 7, {"default": 7, "configured": 7}),
        # TPU vLLM rejects seeded sampling, so only an explicit seed is sent there.
        (Platform.TPU, None, {"default": None, "configured": 5}),
        (Platform.TPU, 7, {"default": 7, "configured": 7}),
    ],
)
def test_run_seed_defaults_only_where_the_backend_accepts_request_seeds(tmp_path, platform, launch_seed, expected):
    config = tmp_path / "custom.yaml"
    config.write_text(
        "evaluations:\n"
        "  default:\n    benchmark: gsm8k\n"
        "  configured:\n    benchmark: gsm8k\n    num_fewshot: 0\n    run:\n      seed: 5\n"
    )
    gpu = platform is Platform.GPU
    spec = LaunchSpec(
        evals=load_evaluations(config),
        model=models()["qwen3-8b"],
        platform=platform,
        accelerator="H100x1" if gpu else None,
        limit=None,
        seed=launch_seed,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster="cw-rno2a" if gpu else None,
        priority_band=0,
        run_id="seed-defaults",
    )

    batch = prepare_evaluation_batch(spec)

    seeds = {
        evaluation.identity.eval_ref.name: evaluation.identity.eval_ref.curated.settings.seed
        for evaluation in batch.evaluations
    }
    assert seeds == expected


@pytest.mark.parametrize(
    "launch_choice, expected",
    [
        (None, {"code": TaskEvidence.REFERENCE, "math": TaskEvidence.EMBEDDED}),
        (TaskEvidence.EMBEDDED, {"code": TaskEvidence.EMBEDDED, "math": TaskEvidence.EMBEDDED}),
    ],
)
def test_task_evidence_follows_the_benchmark_unless_the_launch_chooses(tmp_path, launch_choice, expected):
    config = tmp_path / "custom.yaml"
    config.write_text("evaluations:\n  code:\n    benchmark: livecodebench\n  math:\n    benchmark: gsm8k\n")
    spec = LaunchSpec(
        evals=load_evaluations(config),
        model=models()["qwen3-8b"],
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        task_evidence=launch_choice,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=0,
        run_id="task-evidence",
    )

    batch = prepare_evaluation_batch(spec)

    modes = {
        evaluation.identity.eval_ref.name: evaluation.identity.eval_ref.curated.settings.task_evidence
        for evaluation in batch.evaluations
    }
    assert modes == expected


def _pipeline_spec(tmp_path):
    return LaunchSpec(
        evals=resolve_evaluations("gsm8k-smoke,arc-easy"),
        model=models()["qwen3-8b"],
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=1,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=0,
        cache_key="caller-owned-immutable-artifact",
        version="2026.10.19",
    )


@pytest.mark.parametrize("request_timeout", [1800, 21600])
def test_relaunch_matches_records_without_a_request_deadline(tmp_path, request_timeout):
    spec = replace(_pipeline_spec(tmp_path), cache_key=None, run_id="legacy-record")
    before = prepare_evaluation_batch(spec)
    identity = replace(before.evaluations[0].identity, output_dir=str(tmp_path / "saved-results"))
    record_path = Path(_record(before, identity, RunStatus.INFRA_FAILED, "timed out", {}, {}, {}))
    record = json.loads(record_path.read_text())
    record["model"]["config"].pop("evaluation_request_timeout")
    record_path.write_text(json.dumps(record))

    after = prepare_evaluation_batch(
        replace(spec, model=replace(spec.model, evaluation_request_timeout=request_timeout))
    )

    assert after.evaluations[0].identity.run_id == identity.run_id
    assert after.evaluations[0].identity.output_dir == identity.output_dir


@pytest.mark.parametrize(
    ("agent_selection", "expected_hosts"),
    [
        (None, set()),
        (EvaluationSelection(benchmark="terminal-bench-2.1"), {REGISTRY_HOST}),
        (
            EvaluationSelection(benchmark="ot-tblite", protocol="campaign-2026-09-25"),
            {"codeload.github.com"},
        ),
    ],
)
def test_curated_group_submission_declares_resources_and_source_egress(
    tmp_path, monkeypatch, agent_selection, expected_hosts
):
    requests = []

    def submit(request, **_kwargs):
        requests.append(request)
        return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

    monkeypatch.setattr(
        remote_client,
        "ControllerServiceClientSync",
        lambda **_kwargs: SimpleNamespace(launch_job=submit, close=lambda: None),
    )
    spec = _pipeline_spec(tmp_path)
    if agent_selection is not None:
        spec = replace(
            spec,
            evals=(agent_selection,),
            model=_agent_model(),
            task_machine_backend=TaskMachineBackend.DOCKER,
        )
    batch = prepare_evaluation_batch(spec)
    with IrisClient.remote("http://controller.invalid") as client:
        launch_module.launch_group(batch, client)

    (request,) = requests
    assert request.container_profile == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert request.constraints[0].value.string_value == "cw-rno2a"
    assert {rule.domain for rule in request.egress_allow} == expected_hosts


@pytest.mark.parametrize("cache_key", [None, "caller-owned-immutable-artifact"])
def test_changed_runtime_creates_new_steps_and_preserves_partial_archive(tmp_path, monkeypatch, cache_key):
    spec = replace(
        _pipeline_spec(tmp_path), cache_key=cache_key, run_id="runtime-relaunch" if cache_key is None else None
    )
    before = prepare_evaluation_batch(spec)
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        return {"finished": evaluation.identity.run_id}

    StepRunner().run(build_evaluation_steps(before, execute)[:1])
    pending = before.evaluations[1]
    _record(before, pending.identity, RunStatus.ARTIFACT_FAILED, "interrupted", {}, {}, {})
    partial = Path(pending.identity.output_dir) / "saved-request.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"response":"already generated"}')
    executed.clear()

    monkeypatch.setattr(launch_module, "EVALUATION_EXECUTION_REVISION", "changed-execution")
    after = prepare_evaluation_batch(spec)
    if cache_key is None:
        # A previous launcher can stamp the new runtime onto a record under its old run ID.
        stale = replace(
            after.evaluations[1].identity, run_id=pending.identity.run_id, output_dir=pending.identity.output_dir
        )
        _record(after, stale, RunStatus.INFRA_FAILED, "startup interrupted", {}, {}, {})
        resumed = prepare_evaluation_batch(spec)
        assert resumed.evaluations[1].identity.run_id == after.evaluations[1].identity.run_id
    StepRunner().run(build_evaluation_steps(after, execute))

    assert after.group_id == before.group_id
    assert after.evaluations[0].identity.eval_runtime != before.evaluations[0].identity.eval_runtime
    assert {evaluation.identity.run_id for evaluation in before.evaluations}.isdisjoint(executed)
    assert sorted(executed) == sorted(evaluation.identity.run_id for evaluation in after.evaluations)
    assert after.evaluations[1].identity.output_dir != pending.identity.output_dir
    assert partial.read_text() == '{"response":"already generated"}'


@pytest.mark.parametrize("platform,accelerator", [("tpu", "v5litepod-4"), ("gpu", "GB200x1")])
def test_policy_judge_rejects_incompatible_placement_before_submission(
    tmp_path, submitted_batches, platform, accelerator
):
    config = Path(__file__).parent / "campaigns/eval-campaign-09-25-unlabeled/finalization.yaml"
    command = [
        *_command(tmp_path),
        "--evals",
        "financebench",
        "--policy-finalization-config",
        str(config),
        "--platform",
        platform,
        "--accelerator",
        accelerator,
    ]
    result = CliRunner().invoke(cli, command)
    assert result.exit_code != 0
    assert submitted_batches == []


@pytest.mark.parametrize(
    "records_prefix",
    ["s3://marin-rno2a/marin/evals", "s3://marin-us-east-02a/marin/evals/ablations/regrade-20261009"],
)
def test_policy_launch_rejects_records_prefix_outside_evaldash(tmp_path, submitted_batches, records_prefix):
    command = _command(tmp_path)
    command[command.index("--records-prefix") + 1] = records_prefix
    command[command.index("--evals") + 1] = "financebench"
    config = Path(__file__).parent / "campaigns/eval-campaign-09-25-unlabeled/finalization.yaml"
    command.extend(["--policy-finalization-config", str(config)])
    result = CliRunner().invoke(cli, command)
    assert result.exit_code != 0
    assert "not catalogued by EvalDash" in result.output
    assert submitted_batches == []


def test_policy_snapshot_uses_project_python_and_preserves_judge_configuration(tmp_path):
    campaign = Path(__file__).parent / "campaigns/eval-campaign-09-25-unlabeled"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    system_python = bin_dir / "python3"
    system_python.write_text("#!/bin/sh\necho 'system Python has no project dependencies' >&2\nexit 1\n")
    system_python.chmod(0o755)
    snapshots = tmp_path / "snapshots"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; CONFIG_SNAPSHOT_DIR="$2"; ' 'snapshot_launch_configs --policy-finalization-config "$3"',
            "snapshot",
            str(campaign / "common.sh"),
            str(snapshots),
            str(campaign / "finalization.yaml"),
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "UV_NO_SYNC": "1",
            "MARIN_DIR": str(Path(__file__).resolve().parents[2]),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    snapshot = snapshots / result.stdout.strip()
    manifest = json.loads((snapshot / "launch.json").read_text())
    assert {row["kind"] for row in manifest["files"]} == {
        "policy-finalization-config",
        "judge_model_config",
    }


@pytest.mark.parametrize(
    "benchmark,label", [("financebench", "financebench-smoke"), ("olympiadbench", "olympiad-smoke")]
)
def test_policy_judge_startup_failure_preserves_infrastructure_evidence(
    tmp_path, submitted_batches, monkeypatch, benchmark, label
):
    campaign = Path(__file__).parent / "campaigns/eval-campaign-09-25-unlabeled"
    command = _command(tmp_path)
    selection = tmp_path / "selection.yaml"
    selection.write_text(f"evaluations:\n  {label}:\n    benchmark: {benchmark}\n")
    command[command.index("--evals") + 1] = label
    command.extend(["--evaluation-config", str(selection)])
    command.extend(["--policy-finalization-config", str(campaign / "finalization.yaml")])
    result = CliRunner().invoke(cli, command)
    assert result.exit_code == 0, result.output
    batch = submitted_batches[-1]
    # The batch prepares the policy judge's weights beside the candidate; the evaluation serves it.
    assert batch.judge is None
    assert [judge.model.location for judge in batch.prepared_judges] == ["MiniMaxAI/MiniMax-M3-MXFP8"]
    executor = batch.evaluations[0].executor

    class FailedJob:
        job_id = "/test/policy-judge"

        def logs(self, *, max_lines):
            return ("endpoint registration failed",)

    @contextmanager
    def failed_inference(_config):
        raise RemoteInferenceStartupError("endpoint registration failed", (FailedJob(),))
        yield

    monkeypatch.setattr(policy_module, "remote_inference", failed_inference)
    with pytest.raises(EvaluationError) as caught:
        executor(None, str(tmp_path / "output"), {}, backend_state=lambda: None)
    assert caught.value.status is RunStatus.INFRA_FAILED
    assert caught.value.jobs == {"policy-judge": "/test/policy-judge"}
    assert caught.value.log_tails == {"policy-judge": ("endpoint registration failed",)}


def _agent_model():
    model = models()["qwen3-8b"]
    return replace(
        model,
        serve=replace(model.serve, max_model_len=32768),
        generation=replace(model.generation, max_gen_toks=8192),
    )


@pytest.mark.parametrize(("pi_scored", "reward"), [(94, "reward 0.3917"), (50, "(requires 90%)")])
def test_micro210_scores_pool_rows_after_coverage_floor(tmp_path, monkeypatch, pi_scored, reward):
    monkeypatch.setattr(micro210, "POLICY_REFERENCE", policy_fixture_reference())
    model = replace(_agent_model(), agent=AgentConfig(agent_kwargs={"thinking_format": "chat-template"}))
    spec = replace(_pipeline_spec(tmp_path), model=model, evals=resolve_evaluations("micro210"))
    batch = prepare_evaluation_batch(spec)
    paths = []
    for evaluation, (population, scored, mean) in zip(
        batch.evaluations, ((340, 340, 0.5), (94, pi_scored, 0.0)), strict=True
    ):
        name = evaluation.identity.eval_ref.name
        paths.append(
            _record(
                batch,
                evaluation.identity,
                RunStatus.SUCCEEDED,
                None,
                {name: {"reward": mean}},
                {},
                {},
                coverage={name: TaskCoverage(n_benchmark=population, n_attempted=population, n_scored=scored)},
            )
        )
    result = CliRunner().invoke(cli, ["scores", *paths])
    assert result.exit_code == 0, result.output
    summary = next(line for line in result.output.splitlines() if line.startswith(f"{batch.group_id}  micro210 "))
    assert f"scored {340 + pi_scored}/434" in summary
    assert reward in summary


@pytest.fixture
def agent_selection(tmp_path, agent_source, monkeypatch) -> Path:
    """An evaluation file selecting the fixture agent benchmark, added to the curated catalog."""
    catalog = {
        **evaluation_benchmarks(),
        **{benchmark.name: benchmark for benchmark in fixed_benchmarks([agent_source])},
    }
    monkeypatch.setattr("experiments.evaluation.curated.evaluation_benchmarks", lambda: catalog)
    path = tmp_path / "agent.yaml"
    path.write_text("evaluations:\n  terminal-agent:\n    benchmark: terminal-agent\n")
    return path


@pytest.fixture
def agent_command(tmp_path, agent_selection, monkeypatch) -> list[str]:
    monkeypatch.setattr("experiments.evaluation.cli.models", lambda: {"agent-model": _agent_model()})
    return [
        "launch",
        "--model",
        "agent-model",
        "--evaluation-config",
        str(agent_selection),
        "--evals",
        "terminal-agent",
        "--platform",
        "gpu",
        "--accelerator",
        "H100x1",
        "--records-prefix",
        str(tmp_path / "records"),
        "--no-wait",
    ]


@pytest.mark.parametrize("backend", ["docker", "local"])
def test_a_backend_override_drops_the_daytona_image_build(agent_command, agent_selection, submitted_batches, backend):
    agent_selection.write_text(
        "evaluations:\n  terminal-agent:\n    benchmark: terminal-agent\n"
        "    run:\n      task_machine:\n        image_build: declarative\n        cpus: 2\n"
    )

    result = CliRunner().invoke(cli, [*agent_command, "--task-machine-backend", backend])

    assert result.exit_code == 0, result.output
    (batch,) = submitted_batches
    (evaluation,) = batch.evaluations
    settings = evaluation.identity.eval_ref.curated.settings.task_machine
    assert settings == TaskMachineSettings(backend=TaskMachineBackend(backend), cpus=2)


@pytest.mark.parametrize("backend", [None, "docker", "local"])
def test_agent_launch_binds_daytona_credentials_only_to_daytona_task_machines(agent_command, submitted_batches, backend):
    command = agent_command if backend is None else [*agent_command, "--task-machine-backend", backend]

    result = CliRunner().invoke(cli, command)

    assert result.exit_code == 0, result.output
    (batch,) = submitted_batches
    (evaluation,) = batch.evaluations
    curated = evaluation.identity.eval_ref.curated
    assert curated.settings.task_machine == TaskMachineSettings(backend=TaskMachineBackend(backend or "daytona"))
    # The protocol caps the agent context below the served window and each reply below the model budget.
    assert curated.agent == CuratedAgentRef(
        harness="terminus-2",
        revision=EPISODE_HARNESS_REVISIONS["terminus-2"],
        max_input_tokens=24576,
        max_output_tokens=4096,
    )
    daytona = backend is None
    assert (batch.secret_env.get(DAYTONA_API_KEY) == DAYTONA_SECRET_ENV[DAYTONA_API_KEY]) is daytona
    assert (DAYTONA_API_KEY in evaluation.secret_env_keys) is daytona
    assert set(DAYTONA_EGRESS).issubset(batch.egress_allow) is daytona


def test_agent_launch_budgets_the_window_the_server_runs(agent_command, submitted_batches, monkeypatch):
    # The checkpoint's native window is smaller than the model YAML's, so the server clamps it.
    monkeypatch.setattr(
        "experiments.evaluation.launch.resolved_serve_config",
        lambda model: replace(model.serve, max_model_len=16384, auto_overrides=False),
    )

    result = CliRunner().invoke(cli, agent_command)

    assert result.exit_code == 0, result.output
    (batch,) = submitted_batches
    (evaluation,) = batch.evaluations
    # The agent context limit is the served window; for the YAML window of 32768, the protocol caps it at 24576.
    assert evaluation.identity.eval_ref.curated.agent.max_input_tokens == 16384


def test_agent_launch_requires_the_served_context_window(agent_command, submitted_batches, monkeypatch):
    monkeypatch.setattr("experiments.evaluation.cli.models", lambda: {"agent-model": models()["qwen3-8b"]})

    result = CliRunner().invoke(cli, agent_command)

    assert result.exit_code != 0
    assert "serve.max_model_len" in result.output
    assert submitted_batches == []


@pytest.mark.parametrize("thinking_format", [None, "qwen-chat-template"])
def test_pi_benchmarks_require_the_models_pi_thinking_format(agent_command, agent_source, monkeypatch, thinking_format):
    dataset = agent_source.config
    pi = dataset.agent.model_copy(update={"harness": PiAgent(version="0.87.0")})
    pi_source = replace(
        agent_source, config=replace(dataset, agent=pi, item_builder=partial(dataset.item_builder, agent=pi))
    )
    catalog = {**evaluation_benchmarks(), **{benchmark.name: benchmark for benchmark in fixed_benchmarks([pi_source])}}
    monkeypatch.setattr("experiments.evaluation.curated.evaluation_benchmarks", lambda: catalog)
    model = _agent_model()
    kwargs = {} if thinking_format is None else {"thinking_format": thinking_format}
    model = replace(model, agent=replace(model.agent, agent_kwargs={**model.agent.agent_kwargs, **kwargs}))
    monkeypatch.setattr("experiments.evaluation.cli.models", lambda: {"agent-model": model})

    result = CliRunner().invoke(cli, [*agent_command, "--dry-run"])

    if thinking_format is None:
        assert result.exit_code != 0
        assert "agent.agent_kwargs.thinking_format" in result.output
    else:
        assert result.exit_code == 0, result.output
        assert f"agent=pi@{EPISODE_HARNESS_REVISIONS['pi']}" in result.output


def test_agent_backend_and_harness_revision_are_part_of_run_identity(tmp_path, agent_selection, monkeypatch):
    spec = LaunchSpec(
        evals=load_evaluations(agent_selection),
        model=_agent_model(),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=0,
        run_id="agent-identity",
    )

    def run_id(launch: LaunchSpec) -> str:
        return prepare_evaluation_batch(launch).evaluations[0].identity.run_id

    original = run_id(spec)
    assert run_id(replace(spec, model=replace(spec.model, evaluation_num_concurrent=8))) == original
    assert run_id(replace(spec, task_machine_backend=TaskMachineBackend.DOCKER)) != original
    monkeypatch.setattr("experiments.evaluation.curated.EPISODE_HARNESS_REVISIONS", {"terminus-2": "changed"})
    assert run_id(spec) != original
