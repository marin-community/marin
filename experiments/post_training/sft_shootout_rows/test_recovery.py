# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import itertools
import json
import random
from pathlib import Path

import pytest

from experiments.post_training.sft_shootout_rows import build, recovery
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


def failures(*styles: str) -> tuple:
    return tuple(recovery.Failure(style, seed) for seed, style in enumerate(styles))


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
            1,
            (
                failures("invalid_json", "no_json")
                if trace["family"] == "tmax"
                else failures("prose_no_call", "text_tool_call")
            ),
        ),
        recovery.Injection(3, failures("missing_fields") if trace["family"] == "tmax" else failures("prose_no_call")),
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
                assert 1 <= injection.turn <= 3 and 1 <= len(injection.failures) <= 3


REALISTIC_TURN = {
    "analysis": "The directory has signs.txt and sum.txt; we need to inspect both before writing players.txt.",
    "plan": "List the files, then view the first lines of each.",
    "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}, {"keystrokes": "head -5 signs.txt\n", "duration": 0.1}],
    "task_complete": False,
}


@pytest.mark.parametrize("style", sorted(recovery.TMAX_VARIANTS))
def test_every_tmax_variant_gets_its_styles_error(style):
    for variant in recovery.TMAX_VARIANTS[style]:
        texts = set()
        for seed in range(30):
            text = recovery.tmax_variant(REALISTIC_TURN, style, variant, random.Random(seed))
            error = recovery.terminus_feedback(text).split("ERROR: ", 1)[1]
            assert error.startswith(recovery.TMAX_ERRORS[style]), (variant, error)
            texts.add(text)
        assert len(texts) > 1, f"{style}/{variant} renders one fixed text"


def test_consecutive_tmax_failures_differ_and_vary_between_seeds():
    for seed in range(30):
        rng, previous = random.Random(seed), None
        for style in ("invalid_json", "invalid_json", "no_json", "no_json", "missing_fields"):
            text, variant = recovery.render_tmax_failure(REALISTIC_TURN, style, rng, previous)
            assert text != previous and variant in recovery.TMAX_VARIANTS[style]
            previous = text
    renders = {recovery.render_tmax_failure(REALISTIC_TURN, "no_json", random.Random(s), None)[0] for s in range(30)}
    assert len(renders) > 3


def test_tmax_repeats_follow_v0s_measured_rates():
    pairs = same = repeats = 0
    rng = random.Random(0)
    for _ in range(20000):
        planned = recovery.plan_failures(rng, "tmax", 3)
        for before, after in itertools.pairwise(planned):
            pairs += 1
            same += before.style == after.style
            repeats += after.repeat
    assert abs(same / pairs - recovery.TMAX_REPEAT["same_style"]) < 0.01
    assert repeats / pairs == recovery.TMAX_REPEAT["identical_text"] == 0


def test_a_tmax_failure_is_labeled_with_the_parsers_error():
    # Prose with braces in it is not "No valid JSON": the parser extracts "{3}" and calls it Invalid JSON.
    trace = tmax_trace()
    data = json.loads(trace["messages"][5]["content"])
    data["analysis"] = "Match the code with the regex `PROD-[A-Z]{3}-[0-9]{4}`."
    data["plan"] = "Write each match as `[TIMESTAMP] {json}`."
    trace["messages"][5]["content"] = json.dumps(data, indent=2)
    _, records = recovery.inject("tmax", trace["messages"], [recovery.Injection(2, failures("no_json"))])
    assert [(r["style"], r["error"]) for r in records] == [("invalid_json", "Invalid JSON")]


def test_a_repeat_failure_is_the_previous_text_word_for_word():
    trace = tmax_trace()
    first = recovery.Failure("invalid_json", 7)
    plan = [recovery.Injection(2, (first, recovery.Failure("invalid_json", 7, repeat=True)))]
    messages, records = recovery.inject("tmax", trace["messages"], plan)
    failed = [m["content"] for m in messages if m.get("author") == INJECTED]
    assert len(failed) == 2 and failed[0] == failed[1] and [r["repeat"] for r in records] == [False, True]
    assert recovery.failed_texts(messages) == [failed]


def test_selection_is_an_exact_seeded_fraction_per_family():
    trials = {"tmax": [f"t{i}" for i in range(101)], "swesmith": [f"s{i}" for i in range(50)]}
    chosen = recovery.selected_rows(trials, "seed")
    assert sum(t.startswith("t") for t in chosen) == 20 and sum(t.startswith("s") for t in chosen) == 10
    assert chosen == recovery.selected_rows(trials, "seed") and chosen != recovery.selected_rows(trials, "other")


def tmax_rows(tokenizer, count: int = 5) -> tuple[dict, dict]:
    """`count` plain TMax rows as build_recovery takes them: ({trial: (Attempt, rendering, trace)}, {task: Attempt})."""
    rows, chosen = {}, {}
    for index in range(count):
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
    return rows, chosen


def test_a_row_over_the_context_keeps_its_plain_rendering(tokenizer, monkeypatch):
    rows, chosen = tmax_rows(tokenizer)
    # One selected row (round(0.2 * 5)); a context one token over the plain row cannot hold any injection.
    monkeypatch.setattr(recovery, "CONTEXT", len(next(iter(rows.values()))[1].ids) + 1)
    recovered = {}
    stats = recovery.build_recovery(chosen, rows, tokenizer, recovery.HARNESS_FAITHFUL, "seed", recovered)
    family = stats["families"]["tmax"]
    assert family["selected"] == 1 and family["skipped_over_context"] == 1 and family["injected"] == 0 and not recovered
    monkeypatch.setattr(recovery, "CONTEXT", 32768)
    stats = recovery.build_recovery(chosen, rows, tokenizer, recovery.HARNESS_FAITHFUL, "seed", recovered)
    assert stats["families"]["tmax"]["injected"] == 1 and len(recovered) == 1


def test_smaller_plans_shrink_one_attempt_at_a_time_then_drop_the_last_point():
    plan = [recovery.Injection(1, failures("a", "b")), recovery.Injection(3, failures("c", "d", "e"))]
    shapes = [[(i.turn, len(i.failures)) for i in step] for step in recovery.smaller_plans(plan)]
    assert shapes == [[(1, 2), (3, 3)], [(1, 2), (3, 2)], [(1, 2), (3, 1)], [(1, 1), (3, 1)], [(1, 1)]]


def test_a_row_whose_full_injection_is_over_the_context_takes_a_smaller_one(tokenizer, monkeypatch):
    rows, chosen = tmax_rows(tokenizer)
    (trial,) = recovery.selected_rows({"tmax": sorted(rows)}, "seed")
    trace = rows[trial][2]
    plan = recovery.plan_injections(random.Random(f"seed:{trial}"), "tmax", trace["messages"])
    lengths = [
        len(render_trace(tokenizer, recovery.inject("tmax", trace["messages"], step)[0], None).ids)
        for step in recovery.smaller_plans(plan)
    ]
    if len(lengths) < 2:
        pytest.skip("the seeded plan has a single attempt")
    # Fits the second plan, not the first.
    monkeypatch.setattr(recovery, "CONTEXT", lengths[1])
    recovered = {}
    family = recovery.build_recovery(chosen, rows, tokenizer, recovery.HARNESS_FAITHFUL, "seed", recovered)["families"][
        "tmax"
    ]
    assert family["injected"] == 1 and family["reduced_to_fit"] == 1 and family["skipped_over_context"] == 0
    assert len(recovered[trial].ids) == lengths[1]
    assert (
        sum(family["attempts_per_point"].values()) == len(plan)
        and sum(family["styles"].values()) == sum(len(i.failures) for i in plan) - 1
    )


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
    chosen = recovery.plan_handoff(candidates, family_rows=100, seed="s")
    assert len(chosen) == 13 and chosen == recovery.plan_handoff(list(reversed(candidates)), family_rows=100, seed="s")
    clean = [attempt(100 + i, f"clean-{i}", 250) for i in range(20)]
    dropped = recovery.hold_budget(clean, added=1300, seed="s")
    assert sum(a.supervised for a in dropped) >= 1300 > sum(a.supervised for a in dropped[:-1])


def test_handoff_rows_hold_b_by_dropping_clean_tmax_rows(tokenizer):
    rows, chosen = tmax_rows(tokenizer, count=10)
    summarized = {
        "trial_id": "summarized",
        "task_id": "handoff-task",
        "family": "tmax",
        "relay": False,
        "passed": True,
        "problems": [],
        "sample": 0,
        "retry": 0,
        "tools": None,
        "n_conversations": 2,
        "conversation": 1,
        "messages": [
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": "earlier turn"},
            {"role": "user", "content": "You are picking up work from a previous AI agent."},
            {"role": "assistant", "author": "glm", "content": "What did the previous agent leave unfinished?"},
        ],
    }
    pool = {"handoff-task": {"heldout": False, "family": "tmax", "import_names": []}}
    plain = {"task_slots": sorted(chosen), "B_supervised_tokens": sum(a.supervised for a in chosen.values())}
    kept = dict(rows)
    handoff, removed, stats = build.handoff_rows([summarized], pool, tokenizer, kept, chosen, {}, plain, "seed")
    assert [a.trial_id for a in handoff] == ["summarized#conversation-1"] and handoff[0].trial_id in kept
    dropped = [a for a in chosen.values() if a.trial_id in removed]
    assert stats["supervised_tokens_dropped"] == sum(a.supervised for a in dropped) >= handoff[0].supervised
    assert stats["B_minus_plain_B"] == handoff[0].supervised - stats["supervised_tokens_dropped"] <= 0
    assert stats["kinds"] == {"questions": 1} and stats["tasks_outside_plain_slots"] == 1
