# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary roles: the same preamble bytes in the system turn of every request, an output budget per attempt
that ends the rollout as ``length`` and grades what was left, and evidence per role."""

import json
from dataclasses import dataclass, field, replace
from typing import Any

from rolloutengine.contracts import ModelRequest, ModelTurn
from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import ConversationInput, TextMessage
from taskcompendium.submission import PlainText

from taskforge.ledger.jsonl import read_entries
from taskforge.llm.client import GlmUnavailable
from taskforge.llm.recording import CallLedger
from taskforge.sandbox.factories import SHELLSIM
from taskforge.validate.adversary import SENTINEL_REPLIES, AdversaryRole, role_preamble, run_adversaries
from taskforge.validate.attempts import trial_files
from taskforge.validate.calibration import adversary_signals
from taskforge.validate.outcome import Graded, TrialKind
from taskforge.validate.solver import run_solver
from taskforge.validate.trials import EngineSettings

PLAIN = PlainText(id="plain")
ROLE_IDS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
TURN_TOKENS = 300


def settings(factory) -> EngineSettings:
    return EngineSettings(
        factories={EnvironmentKind.SHELLSIM: factory},
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
        max_turns=6,
        command_timeout=10,
        cleanup_timeout=10,
        conventions=(PLAIN,),
    )


def render(messages) -> tuple[int, ...]:
    """Render like a chat template: a role marker, then the turn's bytes."""
    ids: list[int] = []
    for message in messages:
        ids.append(ROLE_IDS[message["role"]])
        ids.extend(json.dumps({key: message[key] for key in ("content", "tool_calls") if key in message}).encode())
    return tuple(ids)


@dataclass
class TemplateModel:
    """Serves each scripted turn with the ids a chat template would render, as GLM serves a sampled turn."""

    turns: list[dict[str, Any]]
    requests: list[ModelRequest] = field(default_factory=list)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = sum(message["role"] == "assistant" for message in request.messages)
        message = self.turns[index]
        prompt = (*render(request.messages), ROLE_IDS["assistant"])
        response = render([message])[1:]
        return ModelTurn(message, prompt, response, None, "tool_calls" if "tool_calls" in message else "stop")


@dataclass
class EndlessShell:
    """Answers every request with the same shell call, spending ``TURN_TOKENS`` response ids per turn.

    With ``fail_on``, the request with that index (0-based, counted over every request served) raises
    ``GlmUnavailable`` instead, as a drained router would.
    """

    message: dict[str, Any]
    fail_on: int | None = None
    requests: list[ModelRequest] = field(default_factory=list)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if len(self.requests) - 1 == self.fail_on:
            raise GlmUnavailable("router drained", ())
        prompt = (*request.prefix_token_ids, 90) if request.prefix_token_ids else (10, 11)
        return ModelTurn(self.message, prompt, (20,) * TURN_TOKENS, None, "tool_calls")


async def test_every_role_sees_its_preamble_as_the_one_system_turn_on_every_request(tmp_path, file_task, rounds, fakes):
    inner = TemplateModel([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    policy = rounds.policy(adversary_k=1, adversary_output_tokens=4096)

    outcomes = await run_adversaries(
        rounds.draft(file_task, (), PLAIN),
        policy,
        rounds.site(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        lambda _: inner,
    )

    assert set(outcomes) == set(AdversaryRole)
    assert all(isinstance(o, Graded) and o.reward == 1.0 for role in outcomes.values() for o in role)
    assert len(inner.requests) == 2 * len(AdversaryRole)
    system_turns: dict[AdversaryRole, set[str]] = {}
    for request in inner.requests:
        systems = [m for m in request.messages if m["role"] == "system"]
        assert len(systems) == 1 and request.messages[0] is systems[0]
        role = next(r for r in AdversaryRole if SENTINEL_REPLIES[r] in systems[0]["content"])
        system_turns.setdefault(role, set()).add(systems[0]["content"])
    assert system_turns == {role: {role_preamble(role, 4096)} for role in AdversaryRole}
    for role, (preamble,) in system_turns.items():
        assert "4096 output tokens" in preamble and preamble.rstrip().endswith(
            f"{SENTINEL_REPLIES[role]}; you may explain above it."
        )


def test_the_budget_is_in_the_policy_digest(rounds):
    assert rounds.policy(adversary_output_tokens=1024).digest != rounds.policy(adversary_output_tokens=2048).digest


async def test_a_task_system_prompt_follows_the_preamble_in_the_same_turn(tmp_path, file_task, rounds, fakes):
    events = (TextMessage(role="system", content="You are careful."), *file_task.context.events)
    task = file_task.model_copy(update={"context": ConversationInput(events=events)})
    inner = TemplateModel([fakes.text("Done.")])
    policy = replace(rounds.policy(adversary_k=1), roles=(AdversaryRole.LEAK,))

    await run_adversaries(
        rounds.draft(task, (), PLAIN),
        policy,
        rounds.site(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        lambda _: inner,
    )

    first = inner.requests[0].messages[0]
    assert first["role"] == "system"
    assert first["content"] == f"{role_preamble(AdversaryRole.LEAK, policy.adversary_output_tokens)}\n\nYou are careful."


async def test_a_rollout_past_its_output_budget_ends_as_length_and_is_graded(
    tmp_path, file_task, file_facts, rounds, fakes
):
    model = EndlessShell(fakes.shell("ls -la /workspace"))
    policy = replace(rounds.policy(k=1, adversary_k=1, adversary_output_tokens=500), roles=(AdversaryRole.SHORTCUT,))
    draft = rounds.draft(file_task, (), PLAIN)
    site = rounds.site(tmp_path)

    adversaries = await run_adversaries(
        draft, policy, site, settings(fakes.flaky_factory(0, RuntimeError)), lambda _: model
    )
    solver = await run_solver(draft, policy, site, settings(fakes.flaky_factory(0, RuntimeError)), lambda _: model)

    (outcome,) = adversaries[AdversaryRole.SHORTCUT]
    assert isinstance(outcome, Graded) and outcome.reward == 0.0
    assert outcome.rollout.stop_reason == "length" and len(outcome.rollout.steps) == 2
    assert trial_files(site.evidence_dir, TrialKind.ADVERSARY)["shortcut/0"].settled
    signals = adversary_signals(AdversaryRole.SHORTCUT, outcome, file_facts, ())
    assert signals.budget_exhausted and signals.output_tokens == 2 * TURN_TOKENS
    (solved,) = solver
    assert solved.rollout is not None
    assert solved.rollout.stop_reason == "max_turns" and len(solved.rollout.steps) == 6


async def test_the_budget_resets_per_attempt(tmp_path, file_task, rounds, fakes):
    model = EndlessShell(fakes.shell("ls -la /workspace"), fail_on=1)
    policy = replace(
        rounds.policy(adversary_k=1, max_retries=1, adversary_output_tokens=500), roles=(AdversaryRole.SHORTCUT,)
    )
    site = rounds.site(tmp_path)

    (outcome,) = (
        await run_adversaries(
            rounds.draft(file_task, (), PLAIN),
            policy,
            site,
            settings(fakes.flaky_factory(0, RuntimeError)),
            lambda _: model,
        )
    )[AdversaryRole.SHORTCUT]

    # The first attempt spent 300 of 500 before the router drained; the retry gets the whole budget back.
    assert trial_files(site.evidence_dir, TrialKind.ADVERSARY)["shortcut/0"].attempts == 2
    assert isinstance(outcome, Graded) and outcome.rollout.stop_reason == "length"
    assert len(outcome.rollout.steps) == 2


async def test_the_first_request_is_never_refused(tmp_path, file_task, rounds, fakes):
    inner = TemplateModel([fakes.text("x " * 50)])
    policy = replace(rounds.policy(adversary_k=1, adversary_output_tokens=1), roles=(AdversaryRole.AMBIGUITY,))

    (outcome,) = (
        await run_adversaries(
            rounds.draft(file_task, (), PLAIN),
            policy,
            rounds.site(tmp_path),
            settings(fakes.flaky_factory(0, RuntimeError)),
            lambda _: inner,
        )
    )[AdversaryRole.AMBIGUITY]

    assert isinstance(outcome, Graded) and len(outcome.rollout.steps) == 1
    assert outcome.rollout.loss_mask.count(1) > 1


async def test_adversary_evidence_lands_per_role_and_index(tmp_path, file_task, rounds, fakes):
    inner = TemplateModel([fakes.text("NO_SHORTCUT_FOUND")])
    site = rounds.site(tmp_path)
    records: list[CallLedger] = []

    def models(record: CallLedger) -> TemplateModel:
        records.append(record)
        return inner

    outcomes = await run_adversaries(
        rounds.draft(file_task, (), PLAIN),
        rounds.policy(adversary_k=2),
        site,
        settings(fakes.flaky_factory(0, RuntimeError)),
        models,
    )

    assert {role: len(trials) for role, trials in outcomes.items()} == {role: 2 for role in AdversaryRole}
    files = sorted(p.relative_to(site.evidence_dir).as_posix() for p in site.evidence_dir.rglob("attempt-*.json"))
    assert files == sorted(f"adversary/{role}/{i}/attempt-0.json" for role in AdversaryRole for i in range(2))
    steps = {e.step for e in read_entries(tmp_path / "ledger" / "item.jsonl")}
    assert steps == {f"adversary/{role}/{i}/0" for role in AdversaryRole for i in range(2)}
    # Each trial's model records its calls under its own step, the prefix of its attempts' steps.
    assert sorted((r.item_id, r.round, r.step) for r in records) == sorted(
        (site.item_id, site.round, f"adversary/{role}/{i}") for role in AdversaryRole for i in range(2)
    )
