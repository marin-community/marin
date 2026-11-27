# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty CLI receipts retain the distinction between scores and execution blockers."""

import hashlib
import json
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx
import pytest
import yaml
from finestore.reader import ReadView
from harbor.environments.docker_translate import ComposeDocument
from harbor.host_capabilities import HostCapabilities
from harbor.llms.openai_llm import OpenAILLM
from harbor.utils.logger import logger as harbor_logger
from harbor.llms.vllm_tokenization import completion_budget_from_count, tokenize_url
from finestore.rl.difficulty import (
    MEASUREMENTS,
    PROTOCOL_ID,
    SOLVER_MODELS,
    DifficultyGap,
    DifficultyMeasurement,
    DifficultySample,
    SamplingSettings,
    Solver,
    SolverIdentity,
    difficulty_gap_reason,
    measurement_from_attempts,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.runtime import TaskSpecRuntimeBinding
from verifyit.modes.grade_math import (
    MATH_DEADLINE_CANDIDATE_PARSE,
    MATH_DEADLINE_COMPARISON,
    MATH_DEADLINE_REFERENCE_PARSE,
    MATH_VERIFIER_DEADLINE,
)
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import UnsupportedMachineSpec

from experiments.post_training.task_curation import difficulty_relay, difficulty_trials
from experiments.post_training.task_curation.difficulty import ledger_worker
from experiments.post_training.task_curation.difficulty.importer import population_revision
from experiments.post_training.task_curation.difficulty.ledger import digest as ledger_digest
from experiments.post_training.task_curation.difficulty.state import CatalogSource, CoveragePolicy, CellStatus, cell_state
from experiments.post_training.task_curation.difficulty.accounting import GpuAccounting
from experiments.post_training.task_curation.difficulty.ledger import read_rows as read_ledger_rows
from experiments.post_training.task_curation.difficulty.ledger_worker import record_attempt as record_ledger_attempt
from experiments.post_training.task_curation.difficulty.state import GapReason, Outcome as LedgerOutcome
from experiments.post_training.task_curation.difficulty_config import CandidateRun, JudgeRun
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.difficulty_sampling import request_parameters
from experiments.post_training.task_curation.difficulty_provider import ProviderLimits
from experiments.post_training.task_curation.difficulty_trials import (
    ModelTraffic,
    RunSealState,
    iris_root_job_id,
    worker_host_capabilities,
)


@pytest.fixture(autouse=True)
def daytona_connection_environment(monkeypatch):
    monkeypatch.setenv("DAYTONA_API_URL", "https://fixture.invalid")
    monkeypatch.setenv("DAYTONA_TARGET", "us")


def batch_manifest(root, source):
    sample = source.sample.model_dump(mode="json")
    payload = {"kind": "AttemptBatch", "solver": "hosted", "n_attempts": len(source.tasks), "max_gpu_hours": 40.0,
               "sources": [{"source_id": source.sample.source_id, "registry_revision": population_revision(sample),
                            "tasks": [[task.task_id, task.task_sha256, 1] for task in source.sample.tasks],
                            "queue_uri": "fixture:queue", "queue_sha256": "1" * 64, "evidence_ids": []}]}
    path = root / (ledger_digest(payload) + ".json")
    path.write_text(json.dumps(payload))
    return path


def math_deadline_trial(phase):
    """A trial fixture whose verifier verdict carries one math-deadline phase."""

    async def deadline_trial(configuration, _authentication, _guest_environment, _host_capabilities):
        trial_dir = configuration.trials_dir / "candidate"
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True)
        (verifier_dir / "verdict.json").write_text(json.dumps({
            "status": "infra_error", "reward": 0.0,
            "detail": {"code": MATH_VERIFIER_DEADLINE, "error": "math verifier deadline exhausted", "phase": phase},
        }))
        (trial_dir / "network-policy.json").write_text(json.dumps({
            "source_runtime_sha256": canonical_sha256({"task_machine": {"network": "allow"}, "verifier_machine": {"network": "allow"}}),
            "actor_declared": "allow", "verifier_declared": "allow", "actor_required": True,
            "actor_effective": "allow", "verifier_effective": "allow",
        }))
        result = SimpleNamespace(
            finished_at=datetime.now(UTC), completion="completed", verifier_result=None,
            exception_info=None, model_dump=lambda **_kwargs: {"verifier_result": None},
        )
        trial = SimpleNamespace(paths=SimpleNamespace(trial_dir=trial_dir, verifier_dir=verifier_dir))
        return trial, result

    return deadline_trial


def scored_trial():
    async def scored(configuration, _authentication, _guest_environment, _host_capabilities):
        trial_dir = configuration.trials_dir / "candidate"
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True)
        (verifier_dir / "verdict.json").write_text(json.dumps({"status": "scored", "reward": 1.0}))
        (trial_dir / "network-policy.json").write_text(json.dumps({
            "source_runtime_sha256": canonical_sha256({"task_machine": {"network": "allow"}, "verifier_machine": {"network": "allow"}}),
            "actor_declared": "allow", "verifier_declared": "allow", "actor_required": True,
            "actor_effective": "allow", "verifier_effective": "allow",
        }))
        result = SimpleNamespace(
            finished_at=datetime.now(UTC), completion="completed",
            verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
            exception_info=None, model_dump=lambda **_kwargs: {"verifier_result": {"rewards": {"reward": 1.0}}},
        )
        trial = SimpleNamespace(paths=SimpleNamespace(trial_dir=trial_dir, verifier_dir=verifier_dir))
        return trial, result

    return scored


def deadline_fixture(root, source_id="fixture:hardmath", task_id="hardmath"):
    """The prepared task plus the sample and identity the measurement accounting needs."""
    runtime = {"task_machine": {"network": "allow"}, "verifier_machine": {"network": "allow"}}
    runtime_sha256 = canonical_sha256(runtime)
    directory = root
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task-spec.json").write_text(json.dumps({"resources": {}}))
    (directory / "source-runtime.json").write_text(TaskSpecRuntimeBinding(
        task_sha256="d" * 64, runtime_sha256=runtime_sha256, runtime=runtime,
    ).model_dump_json())
    task = PreparedTask(
        task_id=task_id, runtime_sha256=runtime_sha256, path=str(directory),
        workload="task_spec", backend="local", reason=None, judge_model=None,
    )
    settings = SamplingSettings(
        context_tokens=65536, input_tokens=49152, output_tokens=16384,
        temperature=0.7, top_p=0.95, top_k=20, min_p=0,
        repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
        thinking="low", deviations=[],
    )
    config = SimpleNamespace(
        attempts_per_task=6, settings=settings, environment_options={}, solver_protocol_id=PROTOCOL_ID,
        daytona_provisioning="snapshot",
        identity=SimpleNamespace(model="fixture", revision="pin"), api_model="fixture",
    )
    sample = DifficultySample(
        source_id=source_id, release={"repo": "fixture", "revision": "pin", "url": "https://fixture.invalid/pin"},
        converter_version="fixture", converter_revision="a" * 40, population_count=1,
        task_specs_sha256="b" * 64, drop_manifest={"uri": "fixture:drop", "sha256": "c" * 64},
        seed=0, pre_oracle_filter=False, resample_after_oracle_filter=False,
        tasks=[{"task_id": task_id, "task_sha256": "d" * 64}],
    )
    identity = SolverIdentity(
        model=SOLVER_MODELS[Solver.HOSTED], revision="unexposed:fixture-pin",
        tokenizer=SOLVER_MODELS[Solver.HOSTED], tokenizer_revision="a" * 40,
        chat_template_sha256="f" * 64,
        serving_receipt={"uri": "fixture:serving", "sha256": "0" * 64},
    )
    measurement = DifficultyMeasurement(
        measurement_id="fixture", protocol_id="atlas-difficulty-v3", created_at=datetime.now(UTC).isoformat(),
        source_id=sample.source_id, release=sample.release, converter_version=sample.converter_version,
        solver=Solver.HOSTED, status="pending", reason="Awaiting scored coverage", sample=sample,
        sample_receipt={"uri": "fixture:sample", "sha256": "e" * 64}, model=identity, settings=settings,
        n=0, solved=0, wilson_95=None, infrastructure_attempts=0, pending_tasks=1,
        attempts_receipt={"uri": "fixture:attempts", "sha256": "1" * 64},
        gpu_hours=0, input_tokens=0, output_tokens=0,
    )
    return task, config, sample, identity, measurement


def test_reference_phase_deadline_blocks_after_streak(tmp_path, monkeypatch):
    task, config, sample, identity, measurement = deadline_fixture(tmp_path / "source-task")
    monkeypatch.setattr(difficulty_trials, "execute_trial", math_deadline_trial(MATH_DEADLINE_REFERENCE_PARSE))
    output = tmp_path / "receipts"
    outcome = difficulty_trials.run_task(
        task, config, "https://fixture.invalid", tmp_path / "local" / "source", str(output), None, 0,
        host_capabilities=HostCapabilities(),
    )
    limit = difficulty_trials.MATH_DEADLINE_LIMIT
    assert [attempt.status for attempt in outcome.attempts] == ["infrastructure"] * (limit - 1) + ["blocked"]
    assert all(attempt.reward is None for attempt in outcome.attempts)
    assert all(
        attempt.reason == difficulty_gap_reason(DifficultyGap.MATH_VERIFIER_DEADLINE, MATH_DEADLINE_REFERENCE_PARSE)
        for attempt in outcome.attempts
    )
    receipts = sorted(output.glob("*/*/attempt.json"))
    assert len(receipts) == limit

    final = measurement_from_attempts(measurement, outcome.attempts)
    assert (final.status, final.n, final.pending_tasks, final.blocked_tasks, final.infrastructure_attempts) == (
        "blocked", 0, 0, 1, limit,
    )
    assert MATH_VERIFIER_DEADLINE in final.reason




def test_comparison_phase_deadlines_do_not_block_and_report_the_gap(tmp_path, monkeypatch):
    task, config, sample, identity, measurement = deadline_fixture(tmp_path / "source-task")
    monkeypatch.setattr(difficulty_trials, "execute_trial", math_deadline_trial(MATH_DEADLINE_COMPARISON))
    output = tmp_path / "receipts"
    outcome = difficulty_trials.run_task(
        task, config, "https://fixture.invalid", tmp_path / "local" / "source", str(output), None, 0,
        host_capabilities=HostCapabilities(),
    )
    assert [attempt.status for attempt in outcome.attempts] == ["infrastructure"] * config.attempts_per_task
    assert all(
        attempt.reason == difficulty_gap_reason(DifficultyGap.MATH_VERIFIER_DEADLINE, MATH_DEADLINE_COMPARISON)
        for attempt in outcome.attempts
    )

    final = measurement_from_attempts(measurement, outcome.attempts)
    assert (final.status, final.n, final.pending_tasks, final.blocked_tasks) == ("pending", 0, 1, 0)
    assert final.reason == difficulty_gap_reason(DifficultyGap.MATH_VERIFIER_DEADLINE, MATH_DEADLINE_COMPARISON)
    assert final.infrastructure_attempts == config.attempts_per_task




def test_scored_attempt_clears_the_deadline_streak(tmp_path, monkeypatch):
    task, config, sample, identity, measurement = deadline_fixture(tmp_path / "source-task")
    phases = [MATH_DEADLINE_REFERENCE_PARSE, MATH_DEADLINE_REFERENCE_PARSE]

    async def mixed_trial(configuration, _authentication, _guest_environment, host_capabilities):
        if phases:
            return await math_deadline_trial(phases.pop(0))(configuration, _authentication, {}, host_capabilities)
        return await scored_trial()(configuration, _authentication, {}, host_capabilities)

    monkeypatch.setattr(difficulty_trials, "execute_trial", mixed_trial)
    outcome = difficulty_trials.run_task(
        task, config, "https://fixture.invalid", tmp_path / "local" / "source", str(tmp_path / "receipts"), None, 0,
        host_capabilities=HostCapabilities(),
    )
    assert [attempt.status for attempt in outcome.attempts] == ["infrastructure", "infrastructure", "scored"]
    assert outcome.attempts[-1].reward == 1.0

    final = measurement_from_attempts(measurement, outcome.attempts)
    assert (final.status, final.n, final.solved) == ("measured", 1, 1)
    assert final.reason is None


@pytest.mark.parametrize("graded_reward,upstream_failure,workload,legacy_buffer,with_judge", [
    *[(reward, failure, "task_spec", 0, False) for reward in (0.0, 1.0) for failure in (None, "json", "html")],
    *[(reward, "nonretryable", "task_spec", 0, False) for reward in (0.0, 1.0)],
    *[(reward, "actor_recovered", "task_spec", 0, False) for reward in (0.0, 1.0)],
    *[(reward, "judge_recovered", "task_spec", 0, True) for reward in (0.0, 1.0)],
    (0.0, "policy_violation", "task_spec", 0, False),
    (0.0, None, "task_spec", 0, True),
    (0.0, "json", "task_spec", 0, True),
    *[(reward, None, "native", buffer, False) for reward in (0.0, 1.0) for buffer in (0, 8)],
    (1.0, "input_tokenize", "native", 0, False),
    (0.0, "missing_image", "native", 0, False),
    (0.0, "missing_images", "native", 0, False),
    (0.0, "compose_unsupported", "native", 0, False),
])
def test_mixed_scored_and_blocked_source_keeps_its_real_pending_reason(tmp_path, monkeypatch, upstream_failure, graded_reward, workload, legacy_buffer, with_judge):
    infrastructure_first = upstream_failure not in {
        None, "missing_image", "missing_images", "compose_unsupported", "policy_violation",
        "actor_recovered", "judge_recovered",
    }
    unserved = upstream_failure == "input_tokenize"
    provider_wait = upstream_failure in {"json", "html"}
    runtime = {"task_machine": {"network": "allow"}, "verifier_machine": {"network": "allow"}}
    runtime_sha256 = canonical_sha256(runtime)
    blocker = "missing_environment_artifact: fixture/original/requirements.lock"
    sample = DifficultySample(
        source_id="fixture:mixed", release={"repo": "fixture", "revision": "pin", "url": "https://fixture.invalid/pin"},
        converter_version="fixture", converter_revision="a" * 40, population_count=2,
        task_specs_sha256="b" * 64, drop_manifest={"uri": "fixture:drop", "sha256": "c" * 64},
        seed=0, pre_oracle_filter=False, resample_after_oracle_filter=False,
        tasks=[{"task_id": name, "task_sha256": "d" * 64} for name in ("scored", "blocked")],
    )
    source = PreparedSource(
        sample=sample, sample_receipt={"uri": "fixture:sample", "sha256": "e" * 64},
        tasks=[
            PreparedTask(task_id="scored", runtime_sha256=runtime_sha256, path=str(tmp_path / "source-task"),
                         workload=workload, backend="local", reason=None, judge_model="glm-5.3" if with_judge else None),
            PreparedTask(task_id="blocked", runtime_sha256=runtime_sha256, path=None,
                         workload="task_spec", backend="local", reason=blocker, judge_model="glm-5.3" if with_judge else None),
        ],
    )
    directory = tmp_path / "source-task"
    directory.mkdir()
    if workload == "task_spec":
        (directory / "task-spec.json").write_text(json.dumps({"resources": {}}))
    (directory / "task.toml").write_text('[environment]\nworkdir = "/app"\n')
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    (directory / "source-runtime.json").write_text(TaskSpecRuntimeBinding(
        task_sha256="d" * 64, runtime_sha256=runtime_sha256, runtime=runtime,
    ).model_dump_json())
    if upstream_failure in {"missing_image", "missing_images"}:
        (directory / "task.toml").write_text('[environment]\ndocker_image = "example.test/native:latest"\nworkdir = "/app"\n')
        (directory / "instruction.md").write_text("Original native task")
        (directory / "tests").mkdir(exist_ok=True)
        (directory / "tests/test.sh").write_text("exit 0\n")
        source = source.model_copy(update={"tasks": [
            source.tasks[0], source.tasks[1].model_copy(update={"reason": None, "path": str(directory),
                                               "workload": "native" if upstream_failure == "missing_images" else "task_spec"}),
        ]})
        if upstream_failure == "missing_image":
            blocked_directory = tmp_path / "blocked-task"
            shutil.copytree(directory, blocked_directory)
            (blocked_directory / "task-spec.json").write_text(json.dumps({"resources": {}}))
            source = source.model_copy(update={"tasks": [
                source.tasks[0], source.tasks[1].model_copy(update={"path": str(blocked_directory)}),
            ]})
    if upstream_failure == "compose_unsupported":
        (directory / "task.toml").write_text('[environment]\ndocker_image = "example.test/native@sha256:' + "a" * 64 + '"\nworkdir = "/app"\n')
        (directory / "instruction.md").write_text("Original native task")
        (directory / "tests").mkdir(exist_ok=True)
        (directory / "tests/test.sh").write_text("exit 0\n")
        (directory / "docker-compose.yml").write_text(
            "services:\n"
            "  main:\n"
            "    image: example.test/native@sha256:" + "a" * 64 + "\n"
            "    platform: linux/amd64\n"
            "    cpus: 0.1\n"
            "    mem_limit: 128m\n"
        )
        (directory / "shellbox-images.json").write_text(json.dumps(
            {"main": {"image": "example.test/native@sha256:" + "a" * 64, "workdir": "/app"}}))
        source = source.model_copy(update={"tasks": [
            source.tasks[0], source.tasks[1].model_copy(update={"reason": None, "path": str(directory),
                                                               "workload": "native"}),
        ]})
    config = CandidateRun(
        solver=Solver.HOSTED,
        identity=SolverIdentity(model=SOLVER_MODELS[Solver.HOSTED], revision="unexposed:fixture-pin",
                                tokenizer=SOLVER_MODELS[Solver.HOSTED], tokenizer_revision="a" * 40,
                                chat_template_sha256="f" * 64,
                                serving_receipt={"uri": "fixture:serving", "sha256": "0" * 64}),
        settings=SamplingSettings(context_tokens=65536, input_tokens=49152, output_tokens=16384,
                                  temperature=0.7, top_p=0.95, top_k=20, min_p=0, repetition_penalty=1,
                                  presence_penalty=0, frequency_penalty=0, thinking="low", deviations=[]),
        endpoint="https://model.fixture/v1", api_model="glm-5.3", api_key_secret=None,
        provider_limits=ProviderLimits(max_in_flight=2, attempts=2, request_timeout=600),
        environment_options={}, workers=2, gpu_count=0, attempts_per_task=2,
        judge=JudgeRun(endpoint="https://judge.fixture/v1", api_key_secret="unused", api_model="glm-5.3",
                       provider_limits=ProviderLimits(max_in_flight=1, attempts=1, request_timeout=600),
                       identity=SolverIdentity(model=SOLVER_MODELS[Solver.HOSTED], revision="unexposed:fixture-pin",
                                     tokenizer=SOLVER_MODELS[Solver.HOSTED], tokenizer_revision="a" * 40,
                                     chat_template_sha256="f" * 64,
                                     serving_receipt={"uri": "fixture:serving", "sha256": "0" * 64})) if with_judge else None,
    )
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps([source.model_dump(mode="json")]))
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    output = campaign / "trials"
    serving = {
        "protocol": "atlas-difficulty-v3", "solver": "hosted", "model": config.identity.model,
        "native_client_context_safety_buffer_tokens": legacy_buffer,
        "tokenizer_revision": config.identity.tokenizer_revision,
        "chat_template_sha256": config.identity.chat_template_sha256, "sampling": request_parameters(config.settings),
        "tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}},
        "judge": None if config.judge is None else {**config.judge.identity.model_dump(exclude={"serving_receipt"}),
                  "api_model": config.judge.api_model, "effective_temperature": 0.7},
    }
    serving_bytes = json.dumps(serving).encode()
    (campaign / "serving.json").write_bytes(serving_bytes)
    (campaign / "prepared.json").write_text(json.dumps({
        "source_queue_sha256": "1" * 64, "sources": [source.model_dump(mode="json")],
    }))

    judge_requests, candidate_requests = [], []
    real_client = httpx.Client

    def model_response(request):
        body = json.loads(request.content)
        if request.url.host == "judge.fixture":
            judge_requests.append(body)
            if upstream_failure == "judge_recovered" and len(judge_requests) == 1:
                return httpx.Response(503, json={"error": {"message": "Transient judge failure"}})
            return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 20000}})
        candidate_requests.append(body)
        if upstream_failure == "actor_recovered":
            return httpx.Response(401 if len(candidate_requests) == 1 else 200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
        if upstream_failure == "nonretryable":
            return httpx.Response(401, json={"error": {"message": "Credential rejected"}})
        if upstream_failure == "json":
            return httpx.Response(503, json={"error": {"message": "Transient serving failure"}})
        return httpx.Response(502, text="<html>Temporary gateway failure</html>")

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1] * (49153 if unserved else 1)

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(model_response)))
    monkeypatch.setattr(ledger_worker, "require_secret", lambda name: "fixture-host-key")

    physical_attempts = []
    continuing = False
    original_execute_trial = difficulty_trials.execute_trial

    async def physical_trial(configuration, authentication, _guest_environment, host_capabilities):
        # Replace the machine/model boundary; the CLI still dispatches, archives and counts real receipts.
        if upstream_failure in {"missing_image", "missing_images"} and configuration.agent.name == "terminus-2":
            return await original_execute_trial(configuration, authentication, {}, host_capabilities)
        if upstream_failure == "compose_unsupported" and configuration.agent.name == "terminus-2":
            # The local backend cannot reach compose validation off Linux; raise the
            # genuine translator error the prepared sandbox would raise at construction.
            compose_content = yaml.safe_load((Path(configuration.task.path) / "docker-compose.yml").read_text())
            ComposeDocument.model_validate(compose_content)
        if configuration.agent.name == "terminus-2":
            llm = OpenAILLM(model_name="openai/fixture", api_key="fixture",
                           api_base=configuration.agent.kwargs["api_base"],
                           model_info=configuration.agent.kwargs["model_info"])
            assert llm.get_model_context_limit() == 65536
            assert (
                llm.get_model_context_limit() - (config.settings.input_tokens + 1)
                < configuration.agent.kwargs["proactive_summarization_threshold"]
            )
            assert completion_budget_from_count(prompt_tokens=40000, context_window=llm.get_model_context_limit(),
                                                max_output_tokens=llm.get_model_output_limit()) == 16384
        failed = infrastructure_first and not physical_attempts
        reward = 1.0 if continuing else graded_reward
        physical_attempts.append(configuration.trial_name)
        directory = configuration.trials_dir / configuration.trial_name
        agent = directory / "agent"
        verifier = directory / "verifier"
        agent.mkdir(parents=True)
        verifier.mkdir()
        if with_judge and not failed:
            # The machine boundary submits its declared temperature through the CLI's real host relay.
            endpoint = configuration.environment.kwargs["model_endpoint"]
            request = Request(endpoint + "/chat/completions", method="POST",
                              data=json.dumps({"model": "glm-5.3", "temperature": 0.0, "max_tokens": 32768,
                                               "messages": [{"role": "user", "content": "Original rubric"}]}).encode(),
                              headers={"Content-Type": "application/json"})
            if upstream_failure == "judge_recovered" and len(judge_requests) == 0:
                with pytest.raises(HTTPError) as error:
                    urlopen(request, timeout=10)
                assert error.value.code == 503
            with urlopen(request, timeout=10) as response:
                assert response.status == 200
        if upstream_failure == "actor_recovered" and not candidate_requests:
            request = Request(configuration.agent.kwargs["api_base"] + "/chat/completions", method="POST",
                              data=json.dumps({"messages": [{"role": "user", "content": "Original task"}]}).encode(),
                              headers={"Content-Type": "application/json"})
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=10)
            assert error.value.code == 401
            with urlopen(request, timeout=10) as response:
                assert response.status == 200
        if failed:
            endpoint = configuration.agent.kwargs["api_base"]
            request = Request(tokenize_url(endpoint) if unserved else endpoint + "/chat/completions", method="POST",
                              data=json.dumps({"messages": [{"role": "user", "content": "Original task"}]}).encode(),
                              headers={"Content-Type": "application/json"})
            if unserved:
                with urlopen(request, timeout=10) as response:
                    assert json.load(response)["count"] > config.settings.input_tokens
                request = Request(endpoint + "/chat/completions", method="POST", data=request.data,
                                  headers={"Content-Type": "application/json"})
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=10)
            assert error.value.code == (400 if unserved else (401 if upstream_failure == "nonretryable" else 503))
        # A source may grade empty output zero even though its model request was never served.
        verdict_reward = reward
        if not unserved:
            (verifier / "verdict.json").write_text(json.dumps({"status": "scored", "reward": verdict_reward}))
        (directory / "network-policy.json").write_text(json.dumps({
            "source_runtime_sha256": runtime_sha256, "actor_declared": "allow", "verifier_declared": "allow",
            "actor_required": True,
            "actor_effective": "deny" if upstream_failure == "policy_violation" else "allow",
            "verifier_effective": None if unserved else "allow",
        }))
        payload = {"exception_info": {"exception_type": "AgentTimeoutError"} if failed else None,
                   "verifier_result": None if unserved else {"rewards": {"reward": verdict_reward}}}
        (directory / "result.json").write_text(json.dumps(payload))
        result = SimpleNamespace(finished_at=datetime.now(UTC), completion=None if unserved else "completed",
                                 verifier=SimpleNamespace(started_at=datetime.now(UTC), finished_at=datetime.now(UTC)),
                                 exception_info=SimpleNamespace(exception_type="AgentTimeoutError") if failed else None,
                                 verifier_result=None if unserved else SimpleNamespace(rewards={"reward": verdict_reward}),
                                 model_dump=lambda **kwargs: payload)
        trial = SimpleNamespace(paths=SimpleNamespace(trial_dir=directory, agent_dir=agent, verifier_dir=verifier))
        return trial, result

    monkeypatch.setattr(difficulty_trials, "execute_trial", physical_trial)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(output), "--local-root", str(tmp_path / "local"),
                                    "--ledger", str(tmp_path / "ledger"), "--action-manifest", str(batch_manifest(tmp_path, source))])
    ledger_worker.main()

    rows = read_ledger_rows(str(tmp_path / "ledger"))
    scored = [row for row in rows if row.outcome == LedgerOutcome.SCORED]
    unscored = [row for row in rows if row.outcome in {LedgerOutcome.UNSCORED, LedgerOutcome.OBSERVATION}]
    if upstream_failure in {"missing_image", "missing_images", "compose_unsupported"}:
        expected_reason = GapReason.UNSUPPORTED_COMPOSE_DOCUMENT if upstream_failure == "compose_unsupported" else GapReason.ENVIRONMENT_ARTIFACT_MISSING
        assert any(row.reason == expected_reason for row in unscored)
        assert len(scored) == (1 if upstream_failure == "missing_image" else 0)
        evidence = json.loads(next(output.glob("*/attempts/*/*/construction-error.json")).read_text())
        assert evidence["runtime_sha256"] == runtime_sha256
        assert not candidate_requests
        return
    if upstream_failure == "policy_violation":
        assert not scored and any(row.reason == GapReason.NETWORK_POLICY_VIOLATION for row in unscored)
        return
    assert len(scored) == (0 if unserved or provider_wait else 1)
    assert any(row.reason == GapReason.ENVIRONMENT_ARTIFACT_MISSING and row.task_id == "blocked" for row in unscored)
    if unserved:
        assert any(row.reason == GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED for row in unscored)
    elif provider_wait:
        assert any(row.reason == GapReason.PROVIDER_CAPACITY_WAIT and row.outcome == LedgerOutcome.OBSERVATION for row in unscored)
    else:
        assert scored[0].reward == graded_reward
    assert len(list(output.glob("*/attempts/*/*/trial-artifacts.tar.gz"))) == 1 + int(upstream_failure == "nonretryable")
    if with_judge and not unserved and not provider_wait:
        assert all(request["temperature"] == 0.7 for request in judge_requests)



def test_recovered_transport_retry_preserves_the_valid_grade(tmp_path, monkeypatch):
    runtime = {"task_machine": {"network": "allow"}, "verifier_machine": {"network": "allow"}}
    runtime_sha256 = canonical_sha256(runtime)
    sample = DifficultySample(
        source_id="fixture:recovered", release={"repo": "fixture", "revision": "pin", "url": "https://fixture.invalid/pin"},
        converter_version="fixture", converter_revision="a" * 40, population_count=1,
        task_specs_sha256="b" * 64, drop_manifest={"uri": "fixture:drop", "sha256": "c" * 64},
        seed=0, pre_oracle_filter=False, resample_after_oracle_filter=False,
        tasks=[{"task_id": "scored", "task_sha256": "d" * 64}],
    )
    source = PreparedSource(
        sample=sample, sample_receipt={"uri": "fixture:sample", "sha256": "e" * 64},
        tasks=[PreparedTask(task_id="scored", runtime_sha256=runtime_sha256, path=str(tmp_path / "source-task"),
                            workload="task_spec", backend="local", reason=None, judge_model=None)],
    )
    directory = tmp_path / "source-task"
    directory.mkdir()
    (directory / "task-spec.json").write_text(json.dumps({"resources": {}}))
    (directory / "task.toml").write_text('[environment]\nworkdir = "/app"\n')
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    (directory / "source-runtime.json").write_text(TaskSpecRuntimeBinding(
        task_sha256="d" * 64, runtime_sha256=runtime_sha256, runtime=runtime,
    ).model_dump_json())
    config = CandidateRun(
        solver=Solver.HOSTED,
        identity=SolverIdentity(model=SOLVER_MODELS[Solver.HOSTED], revision="unexposed:fixture-pin",
                                tokenizer=SOLVER_MODELS[Solver.HOSTED], tokenizer_revision="a" * 40,
                                chat_template_sha256="f" * 64,
                                serving_receipt={"uri": "fixture:serving", "sha256": "0" * 64}),
        settings=SamplingSettings(context_tokens=65536, input_tokens=49152, output_tokens=16384,
                                  temperature=0.7, top_p=0.95, top_k=20, min_p=0, repetition_penalty=1,
                                  presence_penalty=0, frequency_penalty=0, thinking="low", deviations=[]),
        endpoint="https://model.fixture/v1", api_model="glm-5.3", api_key_secret=None,
        provider_limits=ProviderLimits(max_in_flight=2, attempts=2, request_timeout=600),
        environment_options={}, workers=2, gpu_count=0, attempts_per_task=2,
        judge=None,
    )
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps([source.model_dump(mode="json")]))
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    output = tmp_path / "output" / "trials"

    candidate_requests = []
    real_client = httpx.Client

    def model_response(request):
        body = json.loads(request.content)
        candidate_requests.append(body)
        if len(candidate_requests) <= 2:
            # Both provider admission attempts of the first request fail transiently.
            return httpx.Response(503, json={"error": {"message": "Transient serving failure"}})
        return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 200}})

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(model_response)))
    monkeypatch.setattr(ledger_worker, "require_secret", lambda name: "fixture-host-key")

    async def physical_trial(configuration, authentication, _guest_environment, _host_capabilities):
        # Replace the machine/model boundary; the CLI still dispatches, archives and counts real receipts.
        # The trial's first model request is transiently rejected and its retry succeeds through the relay.
        endpoint = configuration.agent.kwargs["api_base"]
        request = Request(endpoint + "/chat/completions", method="POST",
                          data=json.dumps({"messages": [{"role": "user", "content": "Original task"}]}).encode(),
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=10)
        assert error.value.code == 503
        with urlopen(request, timeout=10) as response:
            assert response.status == 200
        trial_directory = configuration.trials_dir / configuration.trial_name
        agent = trial_directory / "agent"
        verifier = trial_directory / "verifier"
        agent.mkdir(parents=True)
        verifier.mkdir()
        (verifier / "verdict.json").write_text(json.dumps({"status": "scored", "reward": 1.0}))
        (trial_directory / "network-policy.json").write_text(json.dumps({
            "source_runtime_sha256": runtime_sha256, "actor_declared": "allow", "verifier_declared": "allow",
            "actor_required": True, "actor_effective": "allow", "verifier_effective": "allow",
        }))
        payload = {"exception_info": None, "verifier_result": {"rewards": {"reward": 1.0}}}
        (trial_directory / "result.json").write_text(json.dumps(payload))
        result = SimpleNamespace(finished_at=datetime.now(UTC), completion="completed", exception_info=None,
                                 verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                                 model_dump=lambda **kwargs: payload)
        trial = SimpleNamespace(paths=SimpleNamespace(trial_dir=trial_directory, agent_dir=agent, verifier_dir=verifier))
        return trial, result

    monkeypatch.setattr(difficulty_trials, "execute_trial", physical_trial)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                     "--output-root", str(output), "--local-root", str(tmp_path / "local"),
                                     "--ledger", str(tmp_path / "ledger"), "--action-manifest", str(batch_manifest(tmp_path, source))])
    ledger_worker.main()

    rows = read_ledger_rows(str(tmp_path / "ledger"))
    scored = [row for row in rows if row.outcome == LedgerOutcome.SCORED]
    assert [(row.task_id, row.reward) for row in scored] == [("scored", 1.0)]
    assert not [row for row in rows if row.reason == GapReason.PROVIDER_CAPACITY_WAIT]













def test_worker_host_capabilities_binds_data_without_an_eval_fallback():
    capabilities = worker_host_capabilities(
        {},
        environment={"DAYTONA_DATA_API_KEY": "data-key", "DAYTONA_API_KEY": "eval-key",
                     "DAYTONA_API_URL": "https://fixture.invalid", "DAYTONA_TARGET": "us"},
    )
    assert capabilities.daytona is not None
    capabilities.daytona.client_factory()  # an incomplete connection raises here
    assert capabilities.daytona.api_key == "data-key"

    with pytest.raises(UnsupportedMachineSpec, match="missing api_key, target"):
        worker_host_capabilities({}, environment={"DAYTONA_API_KEY": "eval-key"})




def test_run_seal_checkpoint_keeps_own_windows_and_deltas(tmp_path):
    """The checkpointed run seal survives a killed worker with exact own accounting."""

    seal = RunSealState(solver="large", selection_source_ids=("Src:a", "Src:b"), started_at="2026-11-18T00:00:00+00:00")
    traffic = ModelTraffic(receipt=None, input_tokens=120, output_tokens=45)
    judge = ModelTraffic(receipt=None, input_tokens=7, output_tokens=3)
    seal.record(hashlib.sha256("Src:a".encode()).hexdigest(), "Src:a", 0.25, traffic, judge)
    seal.record(hashlib.sha256("Src:b".encode()).hexdigest(), "Src:b", 0.5, ModelTraffic(receipt=None, input_tokens=10, output_tokens=5), None)

    payload = seal.payload()
    assert payload["solver"] == "large"
    assert payload["iris_job_id"] is None  # not running under Iris in this process
    assert payload["selection_source_ids"] == ["Src:a", "Src:b"]
    assert payload["sources_aggregated"] == 2
    assert payload["own_gpu_hours"] == 0.75
    assert payload["own_input_tokens"] == 130 and payload["own_output_tokens"] == 50
    assert payload["judge_own_input_tokens"] == 7 and payload["judge_own_output_tokens"] == 3
    key = hashlib.sha256("Src:a".encode()).hexdigest()
    assert payload["sources"][key] == {
        "source_id": "Src:a",
        "own_gpu_hours": 0.25,
        "own_input_tokens": 120,
        "own_output_tokens": 45,
        "judge_own_input_tokens": 7,
        "judge_own_output_tokens": 3,
    }


def test_iris_root_job_id_derives_the_job_from_the_task_wire(monkeypatch):

    monkeypatch.setenv("IRIS_TASK_ID", "/benfeuer/atlas-difficulty-hosted-20261118-v13c/0:0")
    assert iris_root_job_id() == "/benfeuer/atlas-difficulty-hosted-20261118-v13c"
    monkeypatch.delenv("IRIS_TASK_ID")
    assert iris_root_job_id() is None


def test_ledger_worker_records_zero_token_budget_refusal_without_a_source_aggregate(tmp_path):
    task, config, sample, identity, measurement = deadline_fixture(tmp_path / "source-task")
    config.solver = Solver.LARGE
    source = PreparedSource(sample=sample, sample_receipt={"uri": "fixture:sample", "sha256": "e" * 64}, tasks=[task])
    path = StoragePath(str(tmp_path / "attempt.json"))
    path.write_text(json.dumps({"task_id": task.task_id, "status": "pending", "reward": None,
        "reason": "protocol_input_budget_exceeded", "result": {
            "finished_at": "2026-11-20T10:25:00+00:00", "agent_result": {"n_input_tokens": 0, "n_output_tokens": 0}}}))
    now = [0.0]
    meter = GpuAccounting(8, lambda: now[0])
    meter.start("task")
    now[0] = 3600
    ledger = str(tmp_path / "ledger")
    record_ledger_attempt(path, source=source, config=config, output=StoragePath(str(tmp_path / "run" / "trials")),
                          ledger=ledger, meter=meter, task_key="task")
    rows = read_ledger_rows(ledger)
    assert len(rows) == 1
    assert rows[0].outcome == LedgerOutcome.UNSCORED
    assert rows[0].reason == GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED
    assert rows[0].gpu_hours == 8.0 and rows[0].reward is None
