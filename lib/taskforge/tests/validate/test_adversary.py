# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary roles: the same preamble bytes in the system turn of every request, evidence per role."""

import json
from dataclasses import dataclass, field, replace
from typing import Any

from rolloutengine.contracts import ModelRequest, ModelTurn
from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import ConversationInput, TextMessage
from taskcompendium.submission import PlainText

from taskforge.ledger.jsonl import read_entries
from taskforge.llm.recording import CallLedger
from taskforge.sandbox.factories import SHELLSIM
from taskforge.validate.adversary import ROLE_PREAMBLES, AdversaryRole, run_adversaries
from taskforge.validate.outcome import Graded
from taskforge.validate.trials import EngineSettings

PLAIN = PlainText(id="plain")
ROLE_IDS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}


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


async def test_every_role_sees_its_preamble_as_the_one_system_turn_on_every_request(tmp_path, file_task, rounds, fakes):
    inner = TemplateModel([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    draft = rounds.draft(file_task, (), PLAIN)

    outcomes = await run_adversaries(
        draft,
        rounds.policy(adversary_k=1),
        rounds.site(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        lambda _: inner,
    )

    assert set(outcomes) == set(AdversaryRole)
    assert all(isinstance(o, Graded) and o.reward == 1.0 for role in outcomes.values() for o in role)
    assert len(inner.requests) == 2 * len(AdversaryRole)
    for request in inner.requests:
        systems = [m for m in request.messages if m["role"] == "system"]
        assert len(systems) == 1 and request.messages[0] is systems[0]
    system_turns: dict[AdversaryRole, set[str]] = {}
    for request in inner.requests:
        role = next(r for r in AdversaryRole if request.messages[0]["content"].startswith(ROLE_PREAMBLES[r]))
        system_turns.setdefault(role, set()).add(json.dumps(request.messages[0], sort_keys=True))
    assert set(system_turns) == set(AdversaryRole)
    assert all(len(turns) == 1 for turns in system_turns.values())


async def test_a_task_system_prompt_follows_the_preamble_in_the_same_turn(tmp_path, file_task, rounds, fakes):
    events = (TextMessage(role="system", content="You are careful."), *file_task.context.events)
    task = file_task.model_copy(update={"context": ConversationInput(events=events)})
    inner = TemplateModel([fakes.text("Done.")])
    policy = rounds.policy(adversary_k=1)

    await run_adversaries(
        rounds.draft(task, (), PLAIN),
        replace(policy, roles=(AdversaryRole.LEAK,)),
        rounds.site(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        lambda _: inner,
    )

    first = inner.requests[0].messages[0]
    assert first["role"] == "system"
    assert first["content"].startswith(ROLE_PREAMBLES[AdversaryRole.LEAK])
    assert first["content"].endswith("You are careful.")


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
