# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty CLI receipts retain the distinction between scores and execution blockers."""

import json
import hashlib
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.request import Request, urlopen

import httpx
import pytest
from finestore.reader import ReadView
from finestore.rl.difficulty import (
    MEASUREMENTS,
    SOLVER_MODELS,
    DifficultyMeasurement,
    DifficultySample,
    SamplingSettings,
    Solver,
    SolverIdentity,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.harbor.runtime import TaskSpecRuntimeBinding

from experiments.post_training.task_curation import difficulty_relay, difficulty_trials
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask


@pytest.mark.parametrize("infrastructure_first", [False, True])
@pytest.mark.parametrize("graded_reward", [0.0, 1.0])
@pytest.mark.parametrize("with_judge", [False, True])
def test_mixed_scored_and_blocked_source_keeps_its_real_pending_reason(tmp_path, monkeypatch, infrastructure_first, graded_reward, with_judge):
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
                         workload="task_spec", backend="local", reason=None, judge_model="glm-5.3" if with_judge else None),
            PreparedTask(task_id="blocked", runtime_sha256=runtime_sha256, path=None,
                         workload="task_spec", backend="local", reason=blocker, judge_model="glm-5.3" if with_judge else None),
        ],
    )
    directory = tmp_path / "source-task"
    directory.mkdir()
    (directory / "source-runtime.json").write_text(TaskSpecRuntimeBinding(
        task_sha256="d" * 64, runtime_sha256=runtime_sha256, runtime=runtime,
    ).model_dump_json())
    config = difficulty_trials.CandidateRun(
        solver=Solver.HOSTED,
        identity=SolverIdentity(model=SOLVER_MODELS[Solver.HOSTED], revision="unexposed:fixture-pin",
                                tokenizer=SOLVER_MODELS[Solver.HOSTED], tokenizer_revision="a" * 40,
                                chat_template_sha256="f" * 64,
                                serving_receipt={"uri": "fixture:serving", "sha256": "0" * 64}),
        settings=SamplingSettings(context_tokens=65536, input_tokens=49152, output_tokens=16384,
                                  temperature=0.7, top_p=0.95, top_k=20, min_p=0, repetition_penalty=1,
                                  presence_penalty=0, frequency_penalty=0, thinking="low", deviations=[]),
        endpoint="https://model.fixture/v1", api_model="glm-5.3", api_key_secret=None,
        request_parameters={}, environment_options={}, judge_endpoint="https://judge.fixture/v1",
        judge_api_key_secret="unused", workers=2, gpu_count=0, attempts_per_task=2, task_limit=None,
        judge_api_model="glm-5.3",
        judge_identity=SolverIdentity(model=SOLVER_MODELS[Solver.HOSTED], revision="unexposed:fixture-pin",
                                     tokenizer=SOLVER_MODELS[Solver.HOSTED], tokenizer_revision="a" * 40,
                                     chat_template_sha256="f" * 64,
                                     serving_receipt={"uri": "fixture:serving", "sha256": "0" * 64}),
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
        "tokenizer_revision": config.identity.tokenizer_revision,
        "chat_template_sha256": config.identity.chat_template_sha256, "sampling": config.request_parameters,
        "tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}},
        "judge": {**config.judge_identity.model_dump(exclude={"serving_receipt"}),
                  "api_model": config.judge_api_model, "effective_temperature": 0.7},
    }
    serving_bytes = json.dumps(serving).encode()
    (campaign / "serving.json").write_bytes(serving_bytes)
    (campaign / "prepared.json").write_text(json.dumps({
        "source_queue_sha256": "1" * 64, "sources": [source.model_dump(mode="json")],
    }))

    judge_requests = []
    real_budget_relay = difficulty_relay.budget_relay
    real_client = httpx.Client

    def judge_response(request):
        judge_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(judge_response)))
    monkeypatch.setattr(difficulty_trials, "require_secret", lambda name: "fixture-host-key")

    @contextmanager
    def model_service(**kwargs):
        if kwargs["parameters"] == {"temperature": 0.7}:
            with real_budget_relay(**kwargs) as relay:
                yield relay
        else:
            yield "https://model.fixture"

    physical_attempts = []
    continuing = False

    async def physical_trial(configuration, authentication):
        # Replace the machine/model boundary; the CLI still dispatches, archives and counts real receipts.
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
                              data=json.dumps({"model": "glm-5.3", "temperature": 0.0,
                                               "messages": [{"role": "user", "content": "Original rubric"}]}).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=10) as response:
                assert response.status == 200
        if not failed:
            (verifier / "verdict.json").write_text(json.dumps({"status": "scored", "reward": reward}))
        (directory / "network-policy.json").write_text(json.dumps({
            "source_runtime_sha256": runtime_sha256, "actor_declared": "allow", "verifier_declared": "allow",
            "actor_required": True, "actor_effective": "allow", "verifier_effective": None if failed else "allow",
        }))
        payload = {"exception_info": {"exception_type": "RolloutInterrupted"} if failed else None,
                   "verifier_result": None if failed else {"rewards": {"reward": reward}}}
        (directory / "result.json").write_text(json.dumps(payload))
        result = SimpleNamespace(finished_at=datetime.now(UTC), completion=None if failed else "completed",
                                 exception_info=SimpleNamespace(exception_type="RolloutInterrupted") if failed else None,
                                 verifier_result=None if failed else SimpleNamespace(rewards={"reward": reward}),
                                 model_dump=lambda **kwargs: payload)
        trial = SimpleNamespace(paths=SimpleNamespace(trial_dir=directory, agent_dir=agent, verifier_dir=verifier))
        return trial, result

    monkeypatch.setattr(difficulty_trials, "budget_relay", model_service)
    monkeypatch.setattr(difficulty_trials, "execute_trial", physical_trial)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(output), "--local-root", str(tmp_path / "local")])
    difficulty_trials.main()

    rows = list(ReadView(str(output / "catalog")).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == ("pending", 1, int(graded_reward), 1)
    assert measurement.reason == blocker
    assert measurement.infrastructure_attempts == int(infrastructure_first)
    attempts = json.loads(next(output.glob("*/attempts.json")).read_text())
    assert attempts["pending_reasons"] == {"blocked": blocker}
    assert len(list(output.glob("*/attempts/*/*/trial-artifacts.tar.gz"))) == 1 + int(infrastructure_first)
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
    second = tmp_path / "continuation"
    second.mkdir()
    (second / "prepared.json").write_text(json.dumps({
        "source_queue_sha256": "1" * 64, "sources": [available.model_dump(mode="json")],
    }))
    (second / "serving.json").write_bytes(serving_bytes)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(second / "trials"), "--local-root", str(tmp_path / "continued-local"),
                                    "--previous-campaign", str(campaign), "--previous-serving-sha256", hashlib.sha256(serving_bytes).hexdigest(),
                                    "--source-queue-sha256", "1" * 64])
    difficulty_trials.main()
    rows = list(ReadView(str(second / "trials" / "catalog")).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (continued,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (continued.status, continued.n, continued.solved, continued.pending_tasks) == ("measured", 2, 1 + int(graded_reward), 0)
    assert continued.infrastructure_attempts == int(infrastructure_first)
    assert len(physical_attempts) == 2 + int(infrastructure_first)
    if with_judge:
        assert len(judge_requests) == 2
        assert {request["temperature"] for request in judge_requests} == {0.7}
        assert {request["model"] for request in judge_requests} == {"glm-5.3"}
        assert continued.input_tokens == continued.output_tokens == 0
        ledger = json.loads(next((second / "trials").glob("*/attempts.json")).read_text())
        assert (ledger["judge_input_tokens"], ledger["judge_output_tokens"]) == (1, 2)
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
        assert len(judge_requests) == 2
        changed_judge = config.judge_identity.model_copy(update={"revision": "unexposed:changed"})
        config_path.write_text(config.model_copy(update={"judge_identity": changed_judge}).model_dump_json())
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
