# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty CLI receipts retain the distinction between scores and execution blockers."""

import json
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx
import pytest
from finestore.reader import ReadView
from harbor.llms.vllm_tokenization import tokenize_url, completion_budget_from_count
from harbor.llms.openai_llm import OpenAILLM
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
from experiments.post_training.task_curation.difficulty_sampling import request_parameters


@pytest.mark.parametrize("graded_reward,upstream_failure,with_judge", [
    *[(reward, failure, judge) for reward in (0.0, 1.0) for failure in (None, "json") for judge in (False, True)],
    (1.0, "html", False), (1.0, "input_tokenize", False), (1.0, "native", False),
])
def test_mixed_scored_and_blocked_source_keeps_its_real_pending_reason(tmp_path, monkeypatch, graded_reward, upstream_failure, with_judge):
    infrastructure_first = upstream_failure in {"json", "html", "input_tokenize"}
    unserved = upstream_failure == "input_tokenize"
    native = upstream_failure in {"native", "input_tokenize"}
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
                         workload="native" if native else "task_spec", backend="local", reason=None, judge_model="glm-5.3" if with_judge else None),
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
        environment_options={}, judge_endpoint="https://judge.fixture/v1",
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
        "chat_template_sha256": config.identity.chat_template_sha256, "sampling": request_parameters(config.settings),
        "tools_thinking_probe": {"completion": {"system_fingerprint": "fixture-pin"}},
        "judge": {**config.judge_identity.model_dump(exclude={"serving_receipt"}),
                  "api_model": config.judge_api_model, "effective_temperature": 0.7},
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
            return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 20000}})
        candidate_requests.append(body)
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

    async def physical_trial(configuration, authentication):
        # Replace the machine/model boundary; the CLI still dispatches, archives and counts real receipts.
        if configuration.agent.name == "terminus-2":
            llm = OpenAILLM(model_name="openai/fixture", api_key="fixture", api_base=configuration.agent.kwargs["api_base"], model_info=configuration.agent.kwargs["model_info"])
            assert completion_budget_from_count(prompt_tokens=40000, context_window=llm.get_model_context_limit(), max_output_tokens=llm.get_model_output_limit()) == 16384
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
            with urlopen(request, timeout=10) as response:
                assert response.status == 200
        if failed:
            endpoint = configuration.agent.kwargs["api_base"]
            request = Request(tokenize_url(endpoint) if unserved else endpoint + "/chat/completions", method="POST",
                              data=json.dumps({"messages": [{"role": "user", "content": "Original task"}]}).encode(),
                              headers={"Content-Type": "application/json"})
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=10)
            assert error.value.code == (400 if unserved else 502)
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

    monkeypatch.setattr(difficulty_trials, "execute_trial", physical_trial)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(output), "--local-root", str(tmp_path / "local")])
    difficulty_trials.main()

    rows = list(ReadView(str(output / "catalog")).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == (
        "pending", 0 if unserved else 1, 0 if unserved else int(graded_reward), 2 if unserved else 1)
    assert blocker in measurement.reason
    assert measurement.infrastructure_attempts == int(infrastructure_first and not unserved)
    attempts = json.loads(next(output.glob("*/attempts.json")).read_text())
    pending = {"blocked": blocker}
    if unserved:
        pending["scored"] = difficulty_trials.INPUT_BUDGET_REASON
    assert attempts["pending_reasons"] == pending
    assert len(list(output.glob("*/attempts/*/*/trial-artifacts.tar.gz"))) == 1 + int(infrastructure_first and not unserved)
    assert len(candidate_requests) == int(infrastructure_first and not unserved)
    if unserved:
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
        assert {request["max_tokens"] for request in judge_requests} == {32768}
        assert continued.input_tokens == continued.output_tokens == 0
        ledger = json.loads(next((second / "trials").glob("*/attempts.json")).read_text())
        assert (ledger["judge_input_tokens"], ledger["judge_output_tokens"]) == (1, 20000)
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
