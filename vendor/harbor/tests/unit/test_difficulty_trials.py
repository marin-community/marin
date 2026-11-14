# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty CLI receipts retain the distinction between scores and execution blockers."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.error import HTTPError
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


@pytest.mark.parametrize("upstream_failure", [None, "json", "html"])
def test_mixed_scored_and_blocked_source_keeps_its_real_pending_reason(tmp_path, monkeypatch, upstream_failure):
    infrastructure_first = upstream_failure is not None
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
                         workload="task_spec", backend="local", reason=None, judge_model=None),
            PreparedTask(task_id="blocked", runtime_sha256=runtime_sha256, path=None,
                         workload="task_spec", backend="local", reason=blocker, judge_model=None),
        ],
    )
    directory = tmp_path / "source-task"
    directory.mkdir()
    (directory / "source-runtime.json").write_text(TaskSpecRuntimeBinding(
        task_sha256="d" * 64, runtime_sha256=runtime_sha256, runtime=runtime,
    ).model_dump_json())
    config = difficulty_trials.CandidateRun(
        solver=Solver.HOSTED,
        identity=SolverIdentity(model=SOLVER_MODELS[Solver.HOSTED], revision="fixture-pin",
                                tokenizer=SOLVER_MODELS[Solver.HOSTED], tokenizer_revision="a" * 40,
                                chat_template_sha256="f" * 64,
                                serving_receipt={"uri": "fixture:serving", "sha256": "0" * 64}),
        settings=SamplingSettings(context_tokens=65536, input_tokens=49152, output_tokens=16384,
                                  temperature=0.7, top_p=0.95, top_k=20, min_p=0, repetition_penalty=1,
                                  presence_penalty=0, frequency_penalty=0, thinking="low", deviations=[]),
        endpoint="https://model.fixture/v1", api_model="glm-5.3", api_key_secret=None,
        request_parameters={}, environment_options={}, judge_endpoint="https://judge.fixture/v1",
        judge_api_key_secret="unused", workers=2, gpu_count=0, attempts_per_task=2, task_limit=None,
    )
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps([source.model_dump(mode="json")]))
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    output = tmp_path / "output"

    candidate_requests = []
    real_client = httpx.Client

    def model_response(request):
        candidate_requests.append(json.loads(request.content))
        if upstream_failure == "json":
            return httpx.Response(503, json={"error": {"message": "Transient serving failure"}})
        return httpx.Response(502, text="<html>Temporary gateway failure</html>")

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(model_response)))

    physical_attempts = []

    async def physical_trial(configuration, authentication):
        # Replace the machine/model boundary; the CLI still dispatches, archives and counts real receipts.
        failed = infrastructure_first and not physical_attempts
        physical_attempts.append(configuration.trial_name)
        directory = configuration.trials_dir / configuration.trial_name
        agent = directory / "agent"
        verifier = directory / "verifier"
        agent.mkdir(parents=True)
        verifier.mkdir()
        if failed:
            endpoint = configuration.agent.kwargs["api_base"]
            request = Request(endpoint + "/chat/completions", method="POST",
                              data=json.dumps({"messages": [{"role": "user", "content": "Original task"}]}).encode(),
                              headers={"Content-Type": "application/json"})
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=10)
            assert error.value.code == 502
            assert isinstance(json.loads(error.value.read())["error"], str)
        if not failed:
            (verifier / "verdict.json").write_text(json.dumps({"status": "scored", "reward": 1.0}))
        (directory / "network-policy.json").write_text(json.dumps({
            "source_runtime_sha256": runtime_sha256, "actor_declared": "allow", "verifier_declared": "allow",
            "actor_required": True, "actor_effective": "allow", "verifier_effective": None if failed else "allow",
        }))
        payload = {"exception_info": {"exception_type": "RolloutInterrupted"} if failed else None,
                   "verifier_result": None if failed else {"rewards": {"reward": 1.0}}}
        (directory / "result.json").write_text(json.dumps(payload))
        result = SimpleNamespace(finished_at=datetime.now(UTC), completion=None if failed else "completed",
                                 exception_info=SimpleNamespace(exception_type="RolloutInterrupted") if failed else None,
                                 verifier_result=None if failed else SimpleNamespace(rewards={"reward": 1.0}),
                                 model_dump=lambda **kwargs: payload)
        trial = SimpleNamespace(paths=SimpleNamespace(trial_dir=directory, agent_dir=agent, verifier_dir=verifier))
        return trial, result

    monkeypatch.setattr(difficulty_trials, "execute_trial", physical_trial)
    monkeypatch.setattr("sys.argv", ["difficulty_trials", "--sources", str(source_path), "--config", str(config_path),
                                    "--output-root", str(output), "--local-root", str(tmp_path / "local")])
    difficulty_trials.main()

    rows = list(ReadView(str(output / "catalog")).iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields)))
    (measurement,) = [DifficultyMeasurement.model_validate(row) for row in rows]
    assert (measurement.status, measurement.n, measurement.solved, measurement.pending_tasks) == ("pending", 1, 1, 1)
    assert measurement.reason == blocker
    assert measurement.infrastructure_attempts == int(infrastructure_first)
    attempts = json.loads(next(output.glob("*/attempts.json")).read_text())
    assert attempts["pending_reasons"] == {"blocked": blocker}
    assert len(list(output.glob("*/attempts/*/*/trial-artifacts.tar.gz"))) == 1 + int(infrastructure_first)
    assert len(candidate_requests) == int(infrastructure_first)
