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

from experiments.post_training.task_curation import difficulty_relay, difficulty_trials
from experiments.post_training.task_curation.difficulty_config import CandidateRun, JudgeRun
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.difficulty_sampling import request_parameters
from experiments.post_training.task_curation.difficulty_provider import ProviderLimits
from experiments.post_training.task_curation.difficulty_recovery import PreviousCampaign, recovered_source


def math_deadline_trial(phase):
    """A trial fixture whose verifier verdict carries one math-deadline phase."""

    async def deadline_trial(configuration, _authentication):
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
    async def scored(configuration, _authentication):
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
        attempts_per_task=6, settings=settings, environment_options={},
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
        task, config, "https://fixture.invalid", tmp_path / "local" / "source", str(output), None, 0
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

    campaign = tmp_path / "campaign"
    source_key = hashlib.sha256(sample.source_id.encode()).hexdigest()
    shutil.copytree(output, campaign / "trials" / source_key / "attempts")
    source = PreparedSource(sample=sample, sample_receipt=measurement.sample_receipt, tasks=[task])
    previous = PreviousCampaign(
        root=StoragePath(str(campaign)), sources={sample.source_id: source},
        serving={
            "protocol": PROTOCOL_ID, "solver": Solver.HOSTED.value, "model": identity.model,
            "tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}},
            "tokenizer_revision": identity.tokenizer_revision,
            "chat_template_sha256": identity.chat_template_sha256,
            "sampling": request_parameters(config.settings),
        },
        parent=None,
    )
    recovered = recovered_source(previous, source, Solver.HOSTED, identity, config.settings, request_parameters(config.settings), None)
    assert recovered.math_deadline_streaks == {task.task_id: limit}
    assert recovered.attempts[-1].status == "blocked"

    resumed = difficulty_trials.run_task(
        task, config, "https://fixture.invalid", tmp_path / "continued" / "source",
        str(tmp_path / "continued-receipts"), None, prior_math_deadlines=limit - 1,
    )
    assert len(resumed.attempts) == 1 and resumed.attempts[0].status == "blocked"


def test_comparison_phase_deadlines_do_not_block_and_report_the_gap(tmp_path, monkeypatch):
    task, config, sample, identity, measurement = deadline_fixture(tmp_path / "source-task")
    monkeypatch.setattr(difficulty_trials, "execute_trial", math_deadline_trial(MATH_DEADLINE_COMPARISON))
    output = tmp_path / "receipts"
    outcome = difficulty_trials.run_task(
        task, config, "https://fixture.invalid", tmp_path / "local" / "source", str(output), None, 0
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

    # Old receipts from the pre-#11511 streak rule (bare reason, no phase) are re-read
    # as retryable: the blocked attempt is rewritten to infrastructure, the task leaves
    # the terminal set, and the streak does not continue.
    legacy_dir = tmp_path / "legacy"
    legacy_key = hashlib.sha256(sample.source_id.encode()).hexdigest()
    legacy_attempts = legacy_dir / "trials" / legacy_key / "attempts"
    legacy_attempts.mkdir(parents=True)
    receipt = legacy_attempts / "00" / "00" / "attempt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({
        "task_id": task.task_id, "status": "blocked", "reward": None, "reason": MATH_VERIFIER_DEADLINE,
    }))
    source = PreparedSource(sample=sample, sample_receipt=measurement.sample_receipt, tasks=[task])
    previous = PreviousCampaign(
        root=StoragePath(str(legacy_dir)), sources={sample.source_id: source},
        serving={
            "protocol": PROTOCOL_ID, "solver": Solver.HOSTED.value, "model": identity.model,
            "tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}},
            "tokenizer_revision": identity.tokenizer_revision,
            "chat_template_sha256": identity.chat_template_sha256,
            "sampling": request_parameters(config.settings),
        },
        parent=None,
    )
    recovered = recovered_source(previous, source, Solver.HOSTED, identity, config.settings, request_parameters(config.settings), None)
    assert recovered.math_deadline_streaks == {task.task_id: 0}
    assert [attempt.status for attempt in recovered.attempts] == ["infrastructure"]

    # The re-read task re-enters the attempt budget from a zero streak.
    resumed = difficulty_trials.run_task(
        task, config, "https://fixture.invalid", tmp_path / "continued" / "source",
        str(tmp_path / "continued-receipts"), None, prior_math_deadlines=0,
    )
    assert [attempt.status for attempt in resumed.attempts] == ["infrastructure"] * config.attempts_per_task


def test_scored_attempt_clears_the_deadline_streak(tmp_path, monkeypatch):
    task, config, sample, identity, measurement = deadline_fixture(tmp_path / "source-task")
    phases = [MATH_DEADLINE_REFERENCE_PARSE, MATH_DEADLINE_REFERENCE_PARSE]

    async def mixed_trial(configuration, _authentication):
        if phases:
            return await math_deadline_trial(phases.pop(0))(configuration, _authentication)
        return await scored_trial()(configuration, _authentication)

    monkeypatch.setattr(difficulty_trials, "execute_trial", mixed_trial)
    outcome = difficulty_trials.run_task(
        task, config, "https://fixture.invalid", tmp_path / "local" / "source", str(tmp_path / "receipts"), None, 0
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
    (directory / "source-runtime.json").write_text(TaskSpecRuntimeBinding(
        task_sha256="d" * 64, runtime_sha256=runtime_sha256, runtime=runtime,
    ).model_dump_json())
    if upstream_failure in {"missing_image", "missing_images"}:
        (directory / "task.toml").write_text('[environment]\ndocker_image = "example.test/native:latest"\nworkdir = "/app"\n')
        (directory / "instruction.md").write_text("Original native task")
        (directory / "tests").mkdir()
        (directory / "tests/test.sh").write_text("exit 0\n")
        source = source.model_copy(update={"tasks": [
            source.tasks[0], source.tasks[1].model_copy(update={"reason": None, "path": str(directory),
                                               "workload": "native" if upstream_failure == "missing_images" else "task_spec"}),
        ]})
    if upstream_failure == "compose_unsupported":
        (directory / "task.toml").write_text('[environment]\ndocker_image = "example.test/native@sha256:' + "a" * 64 + '"\nworkdir = "/app"\n')
        (directory / "instruction.md").write_text("Original native task")
        (directory / "tests").mkdir()
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
        provider_limits=ProviderLimits(max_in_flight=2, attempts=2),
        environment_options={}, workers=2, gpu_count=0, attempts_per_task=2, task_limit=None,
        judge=JudgeRun(endpoint="https://judge.fixture/v1", api_key_secret="unused", api_model="glm-5.3",
                       provider_limits=ProviderLimits(max_in_flight=1, attempts=1),
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
    monkeypatch.setattr(difficulty_trials, "require_secret", lambda name: "fixture-host-key")

    physical_attempts = []
    continuing = False
    original_execute_trial = difficulty_trials.execute_trial

    async def physical_trial(configuration, authentication):
        # Replace the machine/model boundary; the CLI still dispatches, archives and counts real receipts.
        if upstream_failure in {"missing_image", "missing_images"} and configuration.agent.name == "terminus-2":
            return await original_execute_trial(configuration, authentication)
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
                                 exception_info=SimpleNamespace(exception_type="AgentTimeoutError") if failed else None,
                                 verifier_result=None if unserved else SimpleNamespace(rewards={"reward": verdict_reward}),
                                 model_dump=lambda **kwargs: payload)
        trial = SimpleNamespace(paths=SimpleNamespace(trial_dir=directory, agent_dir=agent, verifier_dir=verifier))
        return trial, result

    monkeypatch.setattr(difficulty_trials, "execute_trial", physical_trial)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(output), "--local-root", str(tmp_path / "local")])
    difficulty_trials.main()

    rows = list(ReadView(str(output / "catalog")).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    if upstream_failure in {"missing_image", "missing_images", "compose_unsupported"}:
        all_missing = upstream_failure == "missing_images"
        compose_unsupported = upstream_failure == "compose_unsupported"
        assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == (
            ("not_measured", 0, 0, 2) if all_missing or compose_unsupported else ("pending", 1, 0, 1))
        expected_reason = (
            difficulty_trials.UNSUPPORTED_COMPOSE_DOCUMENT_REASON if compose_unsupported
            else difficulty_trials.MISSING_PREPARED_IMAGE_REASON
        )
        assert measurement.reason == expected_reason
        assert measurement.infrastructure_attempts == 0
        evidence = json.loads(next(output.glob("*/attempts/*/*/construction-error.json")).read_text())
        assert evidence["runtime_sha256"] == runtime_sha256
        assert evidence["exception_type"] == ("ValidationError" if compose_unsupported else "MissingPreparedImageError")
        assert evidence["reason"] == expected_reason
        if compose_unsupported:
            assert evidence["message"].startswith("compose document declares keys")
        else:
            assert "@sha256" in evidence["message"]
        ledger = json.loads(next(output.glob("*/attempts.json")).read_text())
        assert [(attempt["task_id"], attempt["status"], attempt["reward"]) for attempt in ledger["attempts"]] == [
            ("scored", "pending", None), ("blocked", "pending" if all_missing or compose_unsupported else "scored",
                                          None if all_missing or compose_unsupported else 0.0),
        ]
        assert not candidate_requests
        assert len(physical_attempts) == (0 if all_missing or compose_unsupported else 1)
        if not compose_unsupported:
            # The real construction path still creates the trial log before failing.
            failed_log = next((tmp_path / "local").rglob("trial.log"))
            size = failed_log.stat().st_size
            harbor_logger.getChild("harbor.trial.trial.candidate").warning("Later candidate log after construction failure")
            assert failed_log.stat().st_size == size
        return
    if upstream_failure == "policy_violation":
        assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == ("not_measured", 0, 0, 2)
        assert difficulty_trials.NETWORK_POLICY_REASON in measurement.reason
        assert blocker in measurement.reason
        return
    assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == (
        "pending", 0 if unserved or provider_wait else 1, 0 if unserved or provider_wait else int(graded_reward), 2 if unserved or provider_wait else 1)
    assert blocker in measurement.reason
    assert measurement.infrastructure_attempts == int(upstream_failure == "nonretryable")
    attempts = json.loads(next(output.glob("*/attempts.json")).read_text())
    if not with_judge:
        assert all(attempts[name] is None for name in (
            "judge_model_requests", "judge_input_tokens", "judge_output_tokens", "judge_temperature", "judge_token_basis",
        ))
        assert not list(output.glob("*/judge-requests.tar.gz"))
    pending = {"blocked": blocker}
    if unserved:
        pending["scored"] = difficulty_trials.INPUT_BUDGET_REASON
    elif provider_wait:
        pending["scored"] = difficulty_trials.PROVIDER_WAIT_REASON
    elif upstream_failure == "policy_violation":
        pending["scored"] = difficulty_trials.NETWORK_POLICY_REASON
    assert attempts["pending_reasons"] == pending
    assert len(list(output.glob("*/attempts/*/*/trial-artifacts.tar.gz"))) == 1 + int(upstream_failure == "nonretryable")
    assert len(candidate_requests) == (2 if provider_wait or upstream_failure == "actor_recovered" else int(upstream_failure == "nonretryable"))
    if unserved or provider_wait or upstream_failure == "policy_violation":
        return
    if infrastructure_first:
        aggregate = next(output.glob("*/attempts.json"))
        aggregate.rename(aggregate.with_name("preserved-before-interruption-attempts.json"))

    # A recovered environment makes the unfinished task executable. The first valid grade
    # must survive even though this model boundary would return a different answer now.
    continuing = True
    available = source.model_copy(update={"tasks": [
        source.tasks[0], source.tasks[1].model_copy(update={"reason": None, "path": str(directory)}),
    ]})
    source_path.write_text(json.dumps([available.model_dump(mode="json")]))
    previous_root = campaign
    if workload == "task_spec" and upstream_failure is None:
        original_relay = difficulty_trials.budget_relay

        def broken_relay(**kwargs):
            raise RuntimeError("framework failure before visiting sources")

        async def broken_dispatch(configuration, authentication):
            raise RuntimeError("framework failure before source aggregate")

        for phase in ("relay", "dispatch"):
            interrupted = tmp_path / f"interrupted-{phase}"
            interrupted.mkdir()
            (interrupted / "prepared.json").write_text(json.dumps({
                "source_queue_sha256": "1" * 64, "sources": [available.model_dump(mode="json")],
            }))
            (interrupted / "serving.json").write_bytes(serving_bytes)
            monkeypatch.setattr(difficulty_trials, "budget_relay", broken_relay if phase == "relay" else original_relay)
            monkeypatch.setattr(difficulty_trials, "execute_trial", broken_dispatch)
            monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                            "--output-root", str(interrupted / "trials"),
                                            "--local-root", str(tmp_path / f"interrupted-{phase}-local"),
                                            "--previous-campaign", str(previous_root),
                                            "--previous-serving-sha256", hashlib.sha256(serving_bytes).hexdigest(),
                                            "--source-queue-sha256", "1" * 64])
            with pytest.raises(RuntimeError, match="framework failure"):
                difficulty_trials.main()
            assert (interrupted / "trials/model-requests-final.json").exists()
            assert not list((interrupted / "trials").glob("*/attempts.json"))
            previous_root = interrupted
        monkeypatch.setattr(difficulty_trials, "budget_relay", original_relay)
        monkeypatch.setattr(difficulty_trials, "execute_trial", physical_trial)
    second = tmp_path / "continuation"
    second.mkdir()
    (second / "prepared.json").write_text(json.dumps({
        "source_queue_sha256": "1" * 64, "sources": [available.model_dump(mode="json")],
    }))
    (second / "serving.json").write_bytes(serving_bytes)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(second / "trials"), "--local-root", str(tmp_path / "continued-local"),
                                    "--previous-campaign", str(previous_root), "--previous-serving-sha256", hashlib.sha256(serving_bytes).hexdigest(),
                                    "--source-queue-sha256", "1" * 64])
    if workload == "native" and legacy_buffer:
        with pytest.raises(ValueError, match="Native continuation requires"):
            difficulty_trials.main()
        assert len(physical_attempts) == 1
        return
    difficulty_trials.main()
    rows = list(ReadView(str(second / "trials" / "catalog")).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (continued,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (continued.status, continued.n, continued.solved, continued.pending_tasks) == ("measured", 2, 1 + int(graded_reward), 0)
    assert continued.infrastructure_attempts == int(infrastructure_first)
    assert len(physical_attempts) == 2 + int(infrastructure_first)
    if with_judge:
        assert len(judge_requests) == (3 if upstream_failure == "judge_recovered" else 2)
        assert {request["temperature"] for request in judge_requests} == {0.7}
        assert {request["model"] for request in judge_requests} == {"glm-5.3"}
        assert {request["max_tokens"] for request in judge_requests} == {32768}
        assert continued.input_tokens == continued.output_tokens == 0
        ledger = json.loads(next((second / "trials").glob("*/attempts.json")).read_text())
        assert (ledger["judge_input_tokens"], ledger["judge_output_tokens"]) == (1, 20000)
    if workload == "task_spec" and upstream_failure is None and graded_reward == 0:
        for corruption in ("cycle", "serving", "queue"):
            invalid = tmp_path / f"invalid-{corruption}"
            (invalid / "trials").mkdir(parents=True)
            (invalid / "prepared.json").write_bytes((second / "prepared.json").read_bytes())
            (invalid / "serving.json").write_bytes(serving_bytes)
            (invalid / "trials/model-requests-final.json").write_bytes(
                (second / "trials/model-requests-final.json").read_bytes())
            parent = invalid if corruption == "cycle" else campaign
            (invalid / "trials/parent-campaign.json").write_text(json.dumps({
                "root": str(parent), "source_queue_sha256": "2" * 64 if corruption == "queue" else "1" * 64,
                "serving_receipt": {"uri": str(parent / "serving.json"),
                                    "sha256": "f" * 64 if corruption == "serving" else hashlib.sha256(serving_bytes).hexdigest()},
            }))
            monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                            "--output-root", str(tmp_path / f"invalid-{corruption}-output"),
                                            "--local-root", str(tmp_path / f"invalid-{corruption}-local"),
                                            "--previous-campaign", str(invalid),
                                            "--previous-serving-sha256", hashlib.sha256(serving_bytes).hexdigest(),
                                            "--source-queue-sha256", "1" * 64])
            with pytest.raises(ValueError, match="cycle|ancestry|serving"):
                difficulty_trials.main()
            assert len(physical_attempts) == 2
    assert len(list((second / "trials").glob("*/attempts/*/*/trial-artifacts.tar.gz"))) == 1
    assert json.loads(next(output.glob("*/measurement.json")).read_text())["n"] == 1

    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(tmp_path / "third"), "--local-root", str(tmp_path / "third-local"),
                                    "--previous-campaign", str(second), "--previous-serving-sha256", hashlib.sha256(serving_bytes).hexdigest(),
                                    "--source-queue-sha256", "1" * 64])
    difficulty_trials.main()
    final = DifficultyMeasurement.model_validate(json.loads(next((tmp_path / "third").glob("*/measurement.json")).read_text()))
    assert (final.n, final.solved, final.infrastructure_attempts) == (2, 1 + int(graded_reward), int(infrastructure_first))
    assert len(physical_attempts) == 2 + int(infrastructure_first)
    if with_judge:
        assert len(judge_requests) == (3 if upstream_failure == "judge_recovered" else 2)
        changed_judge = config.judge.identity.model_copy(update={"revision": "unexposed:changed"})
        config_path.write_text(config.model_copy(update={"judge": config.judge.model_copy(update={"identity": changed_judge})}).model_dump_json())
        monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                        "--output-root", str(tmp_path / "changed-judge"),
                                        "--local-root", str(tmp_path / "changed-judge-local"),
                                        "--previous-campaign", str(second),
                                        "--previous-serving-sha256", hashlib.sha256(serving_bytes).hexdigest(),
                                        "--source-queue-sha256", "1" * 64])
        with pytest.raises(ValueError, match="Continuation cannot mix judge identities"):
            difficulty_trials.main()
        assert not list((tmp_path / "changed-judge").glob("*/measurement.json"))
        assert len(physical_attempts) == 2 + int(infrastructure_first)


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
        provider_limits=ProviderLimits(max_in_flight=2, attempts=2),
        environment_options={}, workers=2, gpu_count=0, attempts_per_task=2, task_limit=None,
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
    monkeypatch.setattr(difficulty_trials, "require_secret", lambda name: "fixture-host-key")

    async def physical_trial(configuration, authentication):
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
                                     "--output-root", str(output), "--local-root", str(tmp_path / "local")])
    difficulty_trials.main()

    rows = list(ReadView(str(output / "catalog")).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    # The wait was recovered by the successful retry, so the valid grade is preserved and counted.
    assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == ("measured", 1, 1, 0)
    assert measurement.infrastructure_attempts == 0
    attempts = json.loads(next(output.glob("*/attempts.json")).read_text())
    assert [(attempt["task_id"], attempt["status"], attempt["reward"]) for attempt in attempts["attempts"]] == [
        ("scored", "scored", 1.0),
    ]
    assert attempts["pending_reasons"] == {}
    assert len(candidate_requests) == 3


def _sealed_resume_setup(tmp_path, monkeypatch, physical_attempts):
    """One clean scored task behind the real relay; returns the pieces both seal tests share."""
    runtime = {"task_machine": {"network": "allow"}, "verifier_machine": {"network": "allow"}}
    runtime_sha256 = canonical_sha256(runtime)
    sample = DifficultySample(
        source_id="fixture:sealed-resume", release={"repo": "fixture", "revision": "pin", "url": "https://fixture.invalid/pin"},
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
        provider_limits=ProviderLimits(max_in_flight=2, attempts=2),
        environment_options={}, workers=2, gpu_count=0, attempts_per_task=2, task_limit=None,
        judge=None,
    )
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps([source.model_dump(mode="json")]))
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    serving = {
        "protocol": "atlas-difficulty-v3", "solver": "hosted", "model": config.identity.model,
        "native_client_context_safety_buffer_tokens": 0,
        "tokenizer_revision": config.identity.tokenizer_revision,
        "chat_template_sha256": config.identity.chat_template_sha256, "sampling": request_parameters(config.settings),
        "tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}},
        "judge": None,
    }
    serving_bytes = json.dumps(serving).encode()
    (tmp_path / "serving.json").write_bytes(serving_bytes)

    candidate_requests = []
    real_client = httpx.Client

    def model_response(request):
        candidate_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 200}})

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(model_response)))
    monkeypatch.setattr(difficulty_trials, "require_secret", lambda name: "fixture-host-key")

    async def physical_trial(configuration, authentication):
        # Replace the machine/model boundary; the CLI still dispatches, archives and counts real receipts.
        physical_attempts.append(configuration.trial_name)
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
    return source, config, source_path, config_path, serving_bytes, physical_attempts


def _run_trials(tmp_path, output, *, previous=None, serving_sha=None, local=None):
    argv = ["difficulty_trials", "--sources", str(tmp_path / "sources.json"),
            "--config", str(tmp_path / "config.json"), "--output-root", str(output),
            "--local-root", str(local if local is not None else output.parent / "local")]
    if previous is not None:
        argv += ["--previous-campaign", str(previous), "--previous-serving-sha256", serving_sha,
                 "--source-queue-sha256", "1" * 64]
    monkeypatch_argv = pytest.MonkeyPatch()
    monkeypatch_argv.setattr("sys.argv", argv)
    try:
        difficulty_trials.main()
    finally:
        monkeypatch_argv.undo()


@pytest.mark.parametrize("seal_during_run", [True, False])
def test_killed_worker_resume_recovers_completed_trials_without_rerunning(tmp_path, monkeypatch, seal_during_run):
    physical_attempts = []
    source, config, source_path, config_path, serving_bytes, physical_attempts = _sealed_resume_setup(
        tmp_path, monkeypatch, physical_attempts
    )
    killed = tmp_path / "killed"
    killed.mkdir()
    output = killed / "trials"

    @contextmanager
    def no_exit_seal(root, out):
        # The worker dies before its exit-time seal; nothing writes a final archive at exit.
        yield

    monkeypatch.setattr(difficulty_trials, "preserved_model_traffic", no_exit_seal)
    if not seal_during_run:
        # The worker is killed before even the first incremental seal fires.
        monkeypatch.setattr(difficulty_trials, "seal_model_traffic", lambda root, out: None)
    _run_trials(tmp_path, output)

    seal_path = output / "model-requests-final.json"
    assert seal_path.exists() is seal_during_run
    ledger = json.loads(next(output.glob("*/attempts.json")).read_text())
    assert [(attempt["task_id"], attempt["status"], attempt["reward"]) for attempt in ledger["attempts"]] == [
        ("scored", "scored", 1.0),
    ]
    assert len(physical_attempts) == 1
    # The killed root still needs its parent binding evidence for the continuation.
    (killed / "serving.json").write_bytes(serving_bytes)
    (killed / "prepared.json").write_text(json.dumps({
        "source_queue_sha256": "1" * 64, "sources": [source.model_dump(mode="json")],
    }))
    serving_sha = hashlib.sha256(serving_bytes).hexdigest()

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    _run_trials(tmp_path, resumed / "trials", previous=killed, serving_sha=serving_sha)

    rows = list(ReadView(str(resumed / "trials" / "catalog")).iter_rows(
        MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == ("measured", 1, 1, 0)
    # The completed trial was recovered from the killed root, never re-executed.
    assert len(physical_attempts) == 1
    if not seal_during_run:
        manifest = json.loads(seal_path.read_text())
        assert manifest["reconstructed"] is True
        assert len(manifest["sources"]) == 1
        receipt = manifest["sources"][0]["model_requests"]
        assert file_sha256(StoragePath(receipt["uri"])) == receipt["sha256"]


def test_same_root_resume_retries_from_its_own_output_without_rerunning(tmp_path, monkeypatch):
    """A preempted root retried with identical arguments resumes its own trials output."""
    physical_attempts = []
    source, config, source_path, config_path, serving_bytes, physical_attempts = _sealed_resume_setup(
        tmp_path, monkeypatch, physical_attempts
    )
    root = tmp_path / "root"
    root.mkdir()
    output = root / "trials"
    _run_trials(tmp_path, output)
    assert len(physical_attempts) == 1
    (root / "serving.json").write_bytes(serving_bytes)
    (root / "prepared.json").write_text(json.dumps({
        "source_queue": str(source_path), "source_queue_sha256": "1" * 64,
        "sources": [source.model_dump(mode="json")],
    }))
    serving_sha = hashlib.sha256(serving_bytes).hexdigest()

    # The retry binds the root as its own recovery parent: the existing output is
    # the chain being continued, not a conflicting prior campaign. Iris retries
    # in a fresh container, so the local root is fresh too.
    _run_trials(tmp_path, output, previous=root, serving_sha=serving_sha,
                local=tmp_path / "retried-local")

    # The scored trial was recovered from the same root, never re-executed.
    assert len(physical_attempts) == 1
    rows = list(ReadView(str(output / "catalog")).iter_rows(
        MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    assert len(rows) == 1
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == ("measured", 1, 1, 0)


def test_same_root_resume_replaces_an_interrupted_aggregate_and_keeps_its_tokens(tmp_path, monkeypatch):
    """A kill between the aggregate and its measurement resumes without losing totals."""
    physical_attempts = []
    source, config, source_path, config_path, serving_bytes, physical_attempts = _sealed_resume_setup(
        tmp_path, monkeypatch, physical_attempts
    )
    root = tmp_path / "root"
    root.mkdir()
    output = root / "trials"

    real_seal = difficulty_trials.seal_model_traffic

    def seal_then_die(root_, out):
        # The worker dies right after the first source's aggregate and seal.
        real_seal(root_, out)
        raise KeyboardInterrupt

    # A separate patch instance: undoing it must not strip the fixture's own
    # tokenizer and transport patches before the resumed run.
    seal_patch = pytest.MonkeyPatch()
    seal_patch.setattr(difficulty_trials, "seal_model_traffic", seal_then_die)
    with pytest.raises(KeyboardInterrupt):
        _run_trials(tmp_path, output)
    seal_patch.undo()
    assert not (output / "catalog").exists()
    aggregate_path = next(output.glob("*/attempts.json"))
    killed_ledger = json.loads(aggregate_path.read_text())
    # Simulate nonzero prior traffic: the fixture's mocked trial records none.
    killed_ledger["input_tokens"] = 4242
    killed_ledger["output_tokens"] = 17
    aggregate_path.write_text(json.dumps(killed_ledger))

    (root / "serving.json").write_bytes(serving_bytes)
    (root / "prepared.json").write_text(json.dumps({
        "source_queue": str(source_path), "source_queue_sha256": "1" * 64,
        "sources": [source.model_dump(mode="json")],
    }))
    serving_sha = hashlib.sha256(serving_bytes).hexdigest()

    _run_trials(tmp_path, output, previous=root, serving_sha=serving_sha,
                local=tmp_path / "retried-local")

    assert len(physical_attempts) == 1  # the scored attempt was recovered, not re-run
    rows = list(ReadView(str(output / "catalog")).iter_rows(
        MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (measurement.status, measurement.n, measurement.solved) == ("measured", 1, 1)
    assert measurement.input_tokens == 4242
    # The resumed aggregate carries the killed attempt's token totals forward.
    ledger = json.loads(aggregate_path.read_text())
    assert ledger["input_tokens"] == killed_ledger["input_tokens"]
    assert [(attempt["task_id"], attempt["status"], attempt["reward"]) for attempt in ledger["attempts"]] == [
        ("scored", "scored", 1.0),
    ]


def test_worker_host_capabilities_resolves_the_daytona_connection_from_the_environment():
    """The trial worker inherits DAYTONA_* from the campaign; the capabilities builder
    must receive that environment so Daytona attempts do not fail at client creation."""
    from experiments.post_training.task_curation.difficulty_trials import worker_host_capabilities

    capabilities = worker_host_capabilities(
        {},
        environment={"DAYTONA_API_KEY": "fixture-key", "DAYTONA_API_URL": "https://fixture.invalid", "DAYTONA_TARGET": "us"},
    )
    assert capabilities.daytona is not None
    factory = capabilities.daytona.client_factory()  # an incomplete connection raises here
    assert callable(factory)

    incomplete = worker_host_capabilities({}, environment={})
    assert (incomplete.daytona.api_key, incomplete.daytona.target) == (None, None)
    with pytest.raises(ValueError, match="explicit api_key"):
        incomplete.daytona.client_factory()


def test_previous_campaign_reconstructs_an_empty_seal_for_an_attempt_with_no_completed_source(tmp_path):
    """A root killed before any source aggregate still binds as its own parent."""
    from experiments.post_training.task_curation.difficulty_recovery import previous_campaign

    root = tmp_path / "root"
    (root / "trials").mkdir(parents=True)
    (root / "serving.json").write_bytes(b"{}\n")
    (root / "prepared.json").write_text(json.dumps({"source_queue_sha256": "1" * 64, "sources": []}))

    previous = previous_campaign(str(root), hashlib.sha256(b"{}\n").hexdigest(), "1" * 64)

    assert str(previous.root) == str(root)
    manifest = json.loads((root / "trials" / "model-requests-final.json").read_text())
    assert manifest["reconstructed"] is True
    assert manifest["sources"] == []


def test_run_seal_checkpoint_keeps_own_windows_and_deltas(tmp_path):
    """The checkpointed run seal survives a killed worker with exact own accounting."""
    from experiments.post_training.task_curation.difficulty_trials import ModelTraffic, RunSealState

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
    from experiments.post_training.task_curation.difficulty_trials import iris_root_job_id

    monkeypatch.setenv("IRIS_TASK_ID", "/benfeuer/atlas-difficulty-hosted-20261118-v13c/0:0")
    assert iris_root_job_id() == "/benfeuer/atlas-difficulty-hosted-20261118-v13c"
    monkeypatch.delenv("IRIS_TASK_ID")
    assert iris_root_job_id() is None
