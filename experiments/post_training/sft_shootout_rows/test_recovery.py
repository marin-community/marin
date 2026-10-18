# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import random
from pathlib import Path

import pytest

from experiments.post_training.sft_shootout_rows import recovery
from experiments.post_training.sft_shootout_rows.arms import Attempt
from experiments.post_training.sft_shootout_rows.traces import INJECTED, render_trace
from experiments.post_training.sweep_data.prepare_agentic import STUDENT_MODEL, STUDENT_REVISION

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION)
    except OSError as error:
        pytest.skip(f"v0 tokenizer unavailable: {error}")


def terminus_turn(step: int) -> dict:
    content = json.dumps(
        {
            "analysis": f"Step {step}: the directory listing shows what to inspect next.",
            "plan": f"Run command {step} and read its output.",
            "commands": [{"keystrokes": f"echo step {step}\n", "duration": 0.1}],
        },
        indent=2,
    )
    return {"role": "assistant", "author": "glm", "content": content, "reasoning_content": f"Thinking {step}."}


def tmax_trace(turns: int = 5) -> dict:
    messages = [{"role": "user", "content": "Task: print the steps."}]
    for step in range(turns):
        messages.append(terminus_turn(step))
        messages.append({"role": "user", "content": f"New Terminal Output:\n\nstep {step}\n"})
    messages.pop()
    return {"trial_id": "tmax-trial", "family": "tmax", "messages": messages, "tools": None}


def swe_trace(turns: int = 5) -> dict:
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Fix it."}]
    for step in range(turns):
        call = {
            "id": f"c{step}",
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": f"ls {step}"})},
        }
        messages.append(
            {
                "role": "assistant",
                "author": "glm",
                "content": f"Look at {step}.",
                "reasoning_content": f"Hmm {step}.",
                "tool_calls": [call],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{step}", "content": f"listing {step}"})
    return {"trial_id": "swe-trial", "family": "swesmith", "messages": messages, "tools": TOOLS}


def trained(rendered) -> list[int]:
    return [token for token, weight in zip(rendered.ids, rendered.mask, strict=True) if weight]


def test_vendored_parser_is_the_pinned_harbor_file():
    lines = Path(recovery.__file__).with_name("terminus_json_plain_parser.py").read_text().splitlines(keepends=True)
    body = "".join(lines[lines.index("# fmt: off\n") + 1 :])
    assert (
        hashlib.sha256(body.encode()).hexdigest() == "b20ad9b8931d4f92c31aa588126a342d9cc6a59396f3ef21243bd077895d9022"
    )


@pytest.mark.parametrize(
    ("style", "error"),
    [
        ("no_json", "ERROR: No valid JSON found in response"),
        ("invalid_json", "ERROR: Invalid JSON: Expecting property name enclosed in double quotes"),
        ("missing_fields", "ERROR: Missing required fields: analysis, plan, commands"),
        ("commands_not_array", "ERROR: Field 'commands' must be an array"),
    ],
)
def test_tmax_styles_get_terminus_feedback(style, error):
    failed = recovery.tmax_failure(json.loads(terminus_turn(1)["content"]), style)
    feedback = recovery.terminus_feedback(failed)
    assert feedback.startswith("Previous response had parsing errors:\n" + error)
    assert feedback.endswith("\n\nPlease fix these issues and provide a proper JSON response.")


@pytest.mark.parametrize(
    ("make", "modes"), [(tmax_trace, ["harness_faithful"]), (swe_trace, ["harness_faithful", "with_failed_reply"])]
)
def test_injected_row_trains_exactly_its_plain_tokens_and_masks_the_injection(tokenizer, make, modes):
    trace = make()
    plain = render_trace(tokenizer, trace["messages"], trace["tools"])
    injections = [
        recovery.Injection(
            1, ("invalid_json", "no_json") if trace["family"] == "tmax" else ("prose_no_call", "text_tool_call")
        ),
        recovery.Injection(3, ("missing_fields",) if trace["family"] == "tmax" else ("prose_no_call",)),
    ]
    for mode in modes:
        messages, records = recovery.inject(trace["family"], trace["messages"], injections, mode)
        injected = render_trace(tokenizer, messages, trace["tools"])
        assert injected.supervised == plain.supervised
        assert trained(injected) == trained(plain)
        # Every added token is context: the masked count grows by exactly the length difference.
        added = len(injected.ids) - len(plain.ids)
        assert added > 0 and injected.mask.count(0) - plain.mask.count(0) == added
        assert [record["turn"] for record in records] == [1, 1, 3]
        failed = [m for m in messages if m.get("author") == INJECTED]
        errors = [
            m
            for m in messages
            if m["role"] == "user" and m["content"].startswith(("Previous response", "Tool call error"))
        ]
        assert len(errors) == 3
        if trace["family"] == "swesmith" and mode == "harness_faithful":
            # mini-swe-agent drops unparsable replies: consecutive error messages, no assistant turn between them.
            assert not failed
            index = messages.index(errors[0])
            assert [m["role"] for m in messages[index : index + 3]] == ["user", "user", "assistant"]
            assert errors[0]["content"] == recovery.SWE_NO_TOOL_CALL
        else:
            assert len(failed) == 3 and all(not m.get("reasoning_content") for m in failed if trace["family"] == "tmax")


def test_first_and_last_teacher_turns_are_never_injection_points():
    for trace in (tmax_trace(), swe_trace()):
        assert recovery.eligible_turns(trace["family"], trace["messages"]) == [1, 2, 3]
        for seed in range(50):
            for injection in recovery.plan_injections(random.Random(seed), trace["family"], trace["messages"]):
                assert 1 <= injection.turn <= 3 and 1 <= len(injection.styles) <= 3


def test_selection_is_an_exact_seeded_fraction_per_family():
    trials = {"tmax": [f"t{i}" for i in range(101)], "swesmith": [f"s{i}" for i in range(50)]}
    chosen = recovery.selected_rows(trials, "seed")
    assert sum(t.startswith("t") for t in chosen) == 20 and sum(t.startswith("s") for t in chosen) == 10
    assert chosen == recovery.selected_rows(trials, "seed") and chosen != recovery.selected_rows(trials, "other")


def test_a_row_over_the_context_keeps_its_plain_rendering(tokenizer, monkeypatch):
    rows = {}
    chosen = {}
    for index in range(5):
        trace = tmax_trace() | {"trial_id": f"trial-{index}"}
        rendered = render_trace(tokenizer, trace["messages"], trace["tools"])
        attempt = Attempt(
            task_id=f"task-{index}",
            family="tmax",
            trial_id=trace["trial_id"],
            sample=0,
            retry=0,
            supervised=rendered.supervised,
            context=len(rendered.ids),
            authors=rendered.authors,
        )
        rows[trace["trial_id"]] = (attempt, rendered, trace)
        chosen[attempt.task_id] = attempt
    # One selected row (round(0.2 * 5)); a context one token over the plain row cannot hold any injection.
    monkeypatch.setattr(recovery, "CONTEXT", len(next(iter(rows.values()))[1].ids) + 1)
    recovered = {}
    stats = recovery.build_recovery(chosen, rows, tokenizer, recovery.HARNESS_FAITHFUL, "seed", recovered)
    family = stats["families"]["tmax"]
    assert family["selected"] == 1 and family["skipped_over_context"] == 1 and family["injected"] == 0 and not recovered
    monkeypatch.setattr(recovery, "CONTEXT", 32768)
    stats = recovery.build_recovery(chosen, rows, tokenizer, recovery.HARNESS_FAITHFUL, "seed", recovered)
    assert stats["families"]["tmax"]["injected"] == 1 and len(recovered) == 1


def test_handoff_traces_take_conversations_after_the_first_with_history_masked():
    base = tmax_trace() | {"relay": False, "n_conversations": 3, "passed": True}
    summary = base | {
        "conversation": 1,
        "problems": ["record at message 3 adds an assistant message with no logged reply", "a real problem"],
        "messages": [
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": "earlier turn"},
            {"role": "user", "content": "You are about to hand off your work to another AI agent."},
            {"role": "assistant", "author": "glm", "content": "Summary of the work."},
        ],
    }
    empty = summary | {
        "conversation": 2,
        "messages": [*summary["messages"][:-1], summary["messages"][-1] | {"content": ""}],
    }
    out = recovery.handoff_traces([base | {"conversation": 0}, summary, empty])
    assert [trace["trial_id"] for trace in out] == ["tmax-trial#conversation-1"]
    assert [m.get("author") for m in out[0]["messages"] if m["role"] == "assistant"] == ["history", "glm"]
    assert out[0]["problems"] == ["a real problem"]
    assert recovery.handoff_kind(out[0]["messages"]) == "summary"


def test_handoff_cap_and_budget_rule():
    def attempt(index: int, task: str, supervised: int) -> Attempt:
        return Attempt(
            task_id=task,
            family="tmax",
            trial_id=f"t{index}",
            sample=0,
            retry=0,
            supervised=supervised,
            context=supervised,
            authors=("glm",),
        )

    candidates = [attempt(i, f"task-{i % 30}", 100) for i in range(40)]
    chosen = recovery.plan_handoff(candidates, {f"task-{i}" for i in range(20)}, family_rows=100, seed="s")
    assert len(chosen) == 13 and all(a.task_id in {f"task-{i}" for i in range(20)} for a in chosen)
    clean = [attempt(100 + i, f"clean-{i}", 250) for i in range(20)]
    dropped = recovery.hold_budget(clean, added=1300, seed="s")
    assert sum(a.supervised for a in dropped) >= 1300 > sum(a.supervised for a in dropped[:-1])
