# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive native model requests into retained v2 scores and fail-closed ledger rows."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from harbor.agents import taskcompendium as agents
from harbor.agents.terminus_2.terminus_2 import EpisodeLoggingPaths, LLMRequestContext, Terminus2
from harbor.host_capabilities import HostCapabilities
from harbor.llms import openai_llm
from harbor.llms.chat import Chat
from openai import AsyncOpenAI
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import RolloutLimits
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome as GradeOutcome
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from taskcompendium.request_budget import RequestBudget

from experiments.post_training.task_curation import difficulty_relay, difficulty_trials
from experiments.post_training.task_curation.difficulty.accounting import GpuAccounting
from experiments.post_training.task_curation.difficulty.ledger import read_rows
from experiments.post_training.task_curation.difficulty.ledger_worker import record_attempt
from experiments.post_training.task_curation.difficulty.state import GapReason, Outcome
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.difficulty_provider import ProviderLimits
from experiments.post_training.task_curation.difficulty_sampling import request_parameters
from experiments.post_training.task_curation.tests.test_difficulty_runtime_protocols import candidate

HTTP_CLIENT = httpx.Client


@contextmanager
def native_route(monkeypatch, root, config, case):
    upstream = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return list(range(int(messages[-1]["content"])))

    def respond(request):
        body = json.loads(request.content)
        upstream.append({"body": body, "timeout": request.extensions["timeout"]})
        prompt = int(body["messages"][-1]["content"])
        output = body["max_tokens"] if case == "length" else 4
        if case == "short_length":
            output = 4
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": config.api_model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if case in {"length", "short_length"} else "stop",
                        "message": {
                            "role": "assistant",
                            "content": "12" if case == "answer" else None,
                            "reasoning_content": "Still working.",
                        },
                    }
                ],
                "usage": {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output},
            },
        )

    def client(**kwargs):
        if kwargs.get("headers", {}).get("Authorization") == "Bearer fixture-key":
            return HTTP_CLIENT(transport=httpx.MockTransport(respond), **kwargs)
        return HTTP_CLIENT(**kwargs)

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(difficulty_relay.httpx, "Client", client)
    monkeypatch.setattr(openai_llm, "_VLLM_CONTEXT_SAFETY_BUFFER", 0)
    with difficulty_relay.budget_relay(
        endpoint="https://model.invalid/v1",
        model=config.api_model,
        api_key="fixture-key",
        tokenizer_model=config.identity.tokenizer,
        tokenizer_revision=config.identity.tokenizer_revision,
        parameters=request_parameters(config.settings),
        budget=RequestBudget(
            config.settings.context_tokens, config.settings.input_tokens, config.settings.output_tokens
        ),
        provider_limits=config.provider_limits,
        receipts=root / "model-requests",
        dynamic_output=config.settings.dynamic_output,
    ) as url:
        yield url, upstream


@pytest.mark.parametrize(
    "case,expected_outcome,expected_reward",
    [
        ("stop", Outcome.SCORED, 0.0),
        ("length", Outcome.SCORED, 0.0),
        ("answer", Outcome.SCORED, 1.0),
        ("floor", Outcome.UNSCORED, None),
        ("oversize", Outcome.UNSCORED, None),
        ("short_length", Outcome.UNSCORED, None),
        ("grader", Outcome.UNSCORED, None),
        ("cleanup", Outcome.UNSCORED, None),
        ("secondary", Outcome.UNSCORED, None),
    ],
)
def test_native_terminus_budget_and_termination_reach_canonical_attempts(
    monkeypatch, tmp_path, case, expected_outcome, expected_reward
):
    monkeypatch.setenv("HARBOR_OPENAI_CONNECT_TIMEOUT_SEC", "20")
    config = candidate().model_copy(update={"provider_limits": ProviderLimits(1, 1, 7200)})
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "task.toml").write_text('[environment]\nworkdir="/app"\n')
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(packet),
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    source = PreparedSource.model_validate(
        {
            "sample": {
                "source_id": "fixture",
                "release": {"repo": "fixture", "revision": "pin", "url": "https://example.org"},
                "converter_version": "v1",
                "converter_revision": "b" * 40,
                "population_count": 1,
                "task_specs_sha256": "c" * 64,
                "drop_manifest": {"uri": "memory://drops", "sha256": "d" * 64},
                "seed": 0,
                "pre_oracle_filter": False,
                "resample_after_oracle_filter": False,
                "tasks": [{"task_id": "task", "task_sha256": "e" * 64}],
            },
            "sample_receipt": {"uri": "memory://sample", "sha256": "f" * 64},
            "tasks": [task],
        }
    )
    root = tmp_path / "local"
    root.mkdir()
    output = StoragePath(str(tmp_path / "run/trials"))
    ledger = str(tmp_path / "ledger")
    meter = GpuAccounting(0, lambda: 0.0)
    meter.start("task")
    metadata = []
    captured_errors = []

    async def execute(configuration, authentication, guest_environment, host_capabilities):
        directory = configuration.trials_dir / "candidate"
        directory.mkdir(parents=True)
        verifier = directory / "verifier"
        verifier.mkdir()
        agent = Terminus2(
            logs_dir=directory / "agent",
            model_name="openai/frozen",
            model_alias="openai/frozen",
            **configuration.agent.kwargs,
        )

        async def observe(request):
            if request.url.path.endswith("/chat/completions"):
                metadata.append(request.extensions["timeout"])

        agent._llm._openai_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=configuration.agent.kwargs["api_base"],
            max_retries=0,
            http_client=httpx.AsyncClient(event_hooks={"request": [observe]}),
        )
        chat = Chat(agent._llm)
        prompt = "49089" if case == "floor" else "70000" if case == "oversize" else "100"
        error = None
        try:
            await agent._check_proactive_summarization(chat, "original task", None, pending_prompt=prompt)
            await Terminus2._query_llm(
                agent,
                chat,
                prompt,
                LLMRequestContext(
                    EpisodeLoggingPaths(None, None, None), original_instruction="original task", session=None
                ),
            )
        except Exception as caught:
            error = type(caught).__name__
            captured_errors.append(error)
        finally:
            await agent._llm._openai_client.close()
        now = datetime.now(UTC)
        timing = SimpleNamespace(started_at=now, finished_at=now)
        rewards = None
        verifier_timing = None
        if case in {"answer", "grader"}:
            verifier_timing = timing
            verdict = (
                {"status": "scored", "reward": 1.0} if case == "answer" else {"status": "infra_error", "reward": 0.0}
            )
            (verifier / "verdict.json").write_text(json.dumps(verdict))
            if case == "answer":
                rewards = SimpleNamespace(rewards={"reward": 1.0})
        if case == "cleanup":
            error = "EnvironmentStopError"
        result = SimpleNamespace(
            finished_at=now,
            completion="closed",
            environment_setup=timing,
            agent_execution=timing,
            verifier=verifier_timing,
            verifier_result=rewards,
            secondary_exception_info=(
                [SimpleNamespace(exception_type="ArtifactPersistenceError")] if case == "secondary" else []
            ),
            exception_info=None if error is None else SimpleNamespace(exception_type=error),
            model_dump=lambda **kwargs: {"finished_at": now.isoformat(), "exception_type": error},
        )
        return SimpleNamespace(paths=SimpleNamespace(trial_dir=directory, verifier_dir=verifier)), result

    monkeypatch.setattr(difficulty_trials, "execute_trial", execute)
    with native_route(monkeypatch, root, config, case) as (url, upstream):
        difficulty_trials.run_task(
            task,
            config,
            url,
            root / "source",
            str(output / "source/attempts"),
            None,
            0,
            host_capabilities=HostCapabilities(),
            record_attempt=partial(
                record_attempt, source=source, config=config, output=output, ledger=ledger, meter=meter, task_key="task"
            ),
        )
    (row,) = read_rows(ledger)
    assert (row.outcome, row.reward, row.solver_protocol_id) == (
        expected_outcome,
        expected_reward,
        config.solver_protocol_id,
    )
    if case in {"floor", "oversize"}:
        assert not upstream
        assert row.reason == GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED
        assert captured_errors == ["ContextBudgetExceededError"]
    else:
        assert len(upstream) == len(metadata) >= 1
        if case != "short_length":
            assert len(upstream) == 1
        assert all(x["read"] == 7200 and x["connect"] == 20 for x in metadata)
        assert all(x["timeout"]["read"] == 7200 and x["timeout"]["connect"] == 600 for x in upstream)
        assert upstream[0]["body"]["max_tokens"] == 65372
        assert upstream[0]["body"]["messages"][-1]["content"] == "100"
    attempt = json.loads(Path(row.evidence_uri).read_text())
    if case in {"stop", "length"}:
        assert attempt["model_output_failure"]["max_tokens"] == 65372
        assert attempt["model_output_failure"]["finish_reason"] == ("stop" if case == "stop" else "length")
    else:
        assert "model_output_failure" not in attempt


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["stop", "length"])
async def test_task_spec_client_uses_v2_deadline_and_preserves_native_missing_answer_zero(monkeypatch, tmp_path, case):
    config = candidate().model_copy(update={"provider_limits": ProviderLimits(1, 1, 7200)})
    task = exact_answer_task(
        RawRow("fixture", Source(dataset="fixture", revision="pin", row="0", importer_revision="v1"), {}),
        prompt="100",
        answers=("12",),
        ignore_case=False,
    )
    deadlines = []
    connection_deadlines = []

    async def observe(request):
        deadlines.append(request.extensions["timeout"]["read"])
        connection_deadlines.append(request.extensions["timeout"]["connect"])

    original_client = agents.AsyncOpenAI

    def client(**kwargs):
        return original_client(**kwargs, http_client=httpx.AsyncClient(event_hooks={"request": [observe]}))

    monkeypatch.setattr(agents, "AsyncOpenAI", client)
    with native_route(monkeypatch, tmp_path, config, case) as (url, upstream):
        agent = agents.TaskSpecChatAgent(
            logs_dir=tmp_path / "agent",
            model_name="openai/frozen",
            model_alias="openai/frozen",
            api_base=url + "/task/v1",
            max_turns=50,
            max_tokens=config.settings.output_tokens,
            request_parameters=request_parameters(config.settings),
            request_read_timeout=config.provider_limits.request_timeout,
        )
        limits = RolloutLimits(
            max_turns=50,
            max_total_tokens=None,
            model_turn_timeout=None,
            tool_turn_timeout=None,
            total_turn_timeout=None,
            attempt_timeout=None,
            verifier_timeout=5,
            cleanup_timeout=5,
        )
        attempt = await agent.run_task(task, None, limits)
    grade = grade_answer(task, attempt)
    assert grade.status in {GradeOutcome.GRADED, GradeOutcome.SUBMISSION_FAILURE}
    assert grade.reward == 0.0
    assert attempt.stop_reason == ("stop" if case == "stop" else "length")
    assert len(upstream) == 1 and upstream[0]["body"]["max_tokens"] == 65372
    assert deadlines == [7200]
    assert connection_deadlines == [5.0]
