# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control replay through the real engine with a deterministic stand-in for the server's chat template."""

import json
from dataclasses import dataclass, replace

import pytest
from rigging.timing import ExponentialBackoff
from shellbox.machine import Backend
from taskcompendium.grading_result import Outcome
from taskcompendium.submission import PlainText

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.controls import ControlCategory, ControlKind, Expectation
from taskforge.validate.controls import ControlPlan, ControlVerdict, ServerTokenizer, replay
from taskforge.validate.outcome import Cause, Graded, Ungraded
from taskforge.validate.trials import Deadlines, EngineSettings

ROLE_IDS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}


def render(messages) -> tuple[int, ...]:
    """Render like a chat template: a role marker, then the turn's bytes."""
    ids: list[int] = []
    for message in messages:
        ids.append(ROLE_IDS[message["role"]])
        ids.extend(json.dumps({key: message[key] for key in ("content", "tool_calls") if key in message}).encode())
    return tuple(ids)


class TemplateTokenizer:
    async def prompt_ids(self, messages, options):
        return (*render(messages), ROLE_IDS["assistant"])

    async def rendered_ids(self, messages, options):
        return render(messages)


class ReorderingTokenizer(TemplateTokenizer):
    """A template whose rendering of a finished conversation does not extend its prompt rendering."""

    async def rendered_ids(self, messages, options):
        return render(messages)[::-1]


def settings(factory, max_turns: int = 6) -> EngineSettings:
    return EngineSettings(
        factories={Backend.SHELLSIM.value: factory},
        capabilities={Backend.SHELLSIM.value: SHELLSIM},
        max_turns=max_turns,
        command_timeout=10,
        tool_turn_timeout=20,
        model_turn_timeout=30,
        cleanup_timeout=10,
        conventions=(PlainText(id="plain"),),
    )


def plan(tmp_path, first_attempts: dict[str, int] | None = None) -> ControlPlan:
    return ControlPlan(
        item_id="item",
        round=0,
        deadlines=Deadlines(total_turn_timeout=30, attempt_timeout=60),
        max_retries=1,
        retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
        evidence_dir=tmp_path,
        ledger=JsonlLedger(tmp_path / "ledger"),
        first_attempts=first_attempts or {},
    )


async def test_transcript_and_workspace_controls_meet_their_expectations(tmp_path, file_task, file_controls, fakes):
    outcomes = await replay(
        file_task,
        file_controls,
        plan(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        TemplateTokenizer(),
    )

    assert [(o.control.id, o.verdict) for o in outcomes] == [(c.id, ControlVerdict.MET) for c in file_controls]
    rewards = {o.control.id: o.outcome.reward for o in outcomes if isinstance(o.outcome, Graded)}
    assert (rewards["correct"], rewards["workspace-correct"], rewards["plant-grader"]) == (1.0, 1.0, 0.0)
    assert sorted(path.parent.name for path in (tmp_path / "control").glob("*/attempt-0.json")) == sorted(
        c.id for c in file_controls
    )


async def test_workspace_files_land_after_environment_setup(tmp_path, file_task_with, file_controls, fakes):
    # A setup step that clears the output path must not erase a workspace control's file.
    task = file_task_with(setup=("rm -f /workspace/sum.txt",))

    outcomes = await replay(
        task,
        file_controls,
        plan(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        TemplateTokenizer(),
    )

    workspace = next(o for o in outcomes if o.control.id == "workspace-correct")
    assert workspace.verdict is ControlVerdict.MET
    assert isinstance(workspace.outcome, Graded) and workspace.outcome.reward == 1.0


async def test_math_controls(tmp_path, math_task, math_controls, fakes):
    outcomes = await replay(
        math_task,
        math_controls,
        plan(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        TemplateTokenizer(),
    )

    assert [o.verdict for o in outcomes] == [ControlVerdict.MET] * len(math_controls)


async def test_wrong_expectation_is_violated(tmp_path, math_task, math_controls, fakes):
    mislabeled = replace(
        math_controls[1],
        id="wrong-labeled-correct",
        kind=ControlKind.POSITIVE,
        category=ControlCategory.KNOWN_CORRECT,
        expect=Expectation(status=Outcome.GRADED, reward_min=1.0),
    )
    controls = (*math_controls, mislabeled)

    outcomes = await replay(
        math_task,
        controls,
        plan(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        TemplateTokenizer(),
    )

    assert [o.verdict for o in outcomes] == [ControlVerdict.MET] * len(math_controls) + [ControlVerdict.VIOLATED]


async def test_a_retried_control_replays_from_its_first_turn(tmp_path, file_task, file_controls, fakes):
    factory = fakes.flaky_factory(failures=1, error=lambda: RuntimeError("broker refused"))

    outcomes = await replay(file_task, file_controls, plan(tmp_path), settings(factory), TemplateTokenizer())

    assert [o.verdict for o in outcomes] == [ControlVerdict.MET] * len(file_controls)
    assert factory.creates == len(file_controls) + 1


async def test_a_re_entered_control_numbers_attempts_after_the_ones_on_disk(tmp_path, file_task, file_controls, fakes):
    earlier = tmp_path / "control" / "correct"
    earlier.mkdir(parents=True)
    for index in range(2):
        (earlier / f"attempt-{index}.json").write_bytes(b"earlier attempt")

    outcomes = await replay(
        file_task,
        file_controls,
        plan(tmp_path, first_attempts={"correct": 2}),
        settings(fakes.flaky_factory(0, RuntimeError)),
        TemplateTokenizer(),
    )

    assert [o.verdict for o in outcomes] == [ControlVerdict.MET] * len(file_controls)
    assert [(earlier / f"attempt-{index}.json").read_bytes() for index in range(2)] == [b"earlier attempt"] * 2
    written = sorted((path.parent.name, path.name) for path in (tmp_path / "control").glob("*/attempt-*.json"))
    expected = [(c.id, "attempt-0.json") for c in file_controls if c.id != "correct"]
    expected += [("correct", f"attempt-{index}.json") for index in range(3)]
    assert written == sorted(expected)


async def test_a_control_longer_than_max_turns_is_refused(tmp_path, file_task, file_controls, fakes):
    with pytest.raises(ValueError, match="max_turns"):
        await replay(
            file_task,
            file_controls,
            plan(tmp_path),
            settings(fakes.flaky_factory(0, RuntimeError), max_turns=1),
            TemplateTokenizer(),
        )


async def test_a_rendering_that_does_not_extend_the_prompt_is_ungraded_and_not_retried(
    tmp_path, math_task, math_controls, fakes
):
    outcomes = await replay(
        math_task,
        math_controls,
        plan(tmp_path),
        settings(fakes.flaky_factory(0, RuntimeError)),
        ReorderingTokenizer(),
    )

    assert all(o.verdict is ControlVerdict.UNGRADED for o in outcomes)
    assert all(isinstance(o.outcome, Ungraded) and o.outcome.cause is Cause.TOKEN_CONTRACT for o in outcomes)
    # Scripted turns render the same way every time, so a control has no token contract retries.
    assert sorted(path.name for path in (tmp_path / "control").glob("*/attempt-*.json")) == ["attempt-0.json"] * len(
        math_controls
    )


@dataclass
class PromptIdStream:
    """A one-token reply whose first chunk carries ``prompt_token_ids``; the fake server sends it as is."""

    events: list[dict]
    send_done: bool = True
    stall: None = None


def prompt_id_stream(prompt: list[int]) -> PromptIdStream:
    return PromptIdStream(
        [
            {"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}], "prompt_token_ids": prompt},
            {"choices": [{"index": 0, "delta": {"content": "x"}, "token_ids": [7], "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1}},
        ]
    )


async def test_server_tokenizer_renders_with_and_without_the_generation_prompt(fake_glm):
    fake_glm.responses.append(prompt_id_stream([1, 2, 3]))
    fake_glm.responses.append(prompt_id_stream([1, 2, 3, 4]))
    endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test", pool=Pool.HIGH)
    tools = [{"type": "function", "function": {"name": "shell"}}]
    async with GlmClient(endpoint, max_attempts=1) as client:
        tokenize = ServerTokenizer(client, LLMPolicy())
        messages = [{"role": "user", "content": "q"}]
        prompt = await tokenize.prompt_ids(messages, {"tools": tools})
        rendered = await tokenize.rendered_ids([*messages, {"role": "assistant", "content": "a"}], {"tools": tools})

    assert (prompt, rendered) == ((1, 2, 3), (1, 2, 3, 4))
    first, second = fake_glm.requests
    assert (first["max_tokens"], first["return_token_ids"], first["tools"]) == (1, True, tools)
    assert "add_generation_prompt" not in first and second["add_generation_prompt"] is False
