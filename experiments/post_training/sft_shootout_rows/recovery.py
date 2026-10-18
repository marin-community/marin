# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recovery rows (task #225): plain's trajectories with masked synthetic failed attempts before some teacher turns.

A reconstruction of Marianna's recovery SFT (marianna13/grug-sft-mix-v4-40M-think) for the shootout. A fixed, seeded
20% of each family's plain rows get 1 or 2 injection points. An injection point is a teacher turn other than the
first and the last. Before it go 1 to 3 failed attempts, each followed by the harness's real reply to that failure.
Injected turns and replies are context (zero loss). Every teacher turn trains exactly as in plain, so a row's
supervised tokens equal its plain row's; only context grows.

Each injected row must match what the deployed harness would send:
- Terminus-2 (TMax) keeps a failed reply in history and answers with its parser's feedback. The failed attempt is
  the teacher turn re-rendered in one of v0's measured error styles. The reply is the exact prompt Terminus-2
  builds from the vendored parser's error and warning (terminus_2.py:2029-2035, 2273-2278).
- mini-swe-agent 2.1.0 (SWE-smith) drops a reply it cannot parse, raising FormatError before the reply enters
  history (default.py:113-116), and sends only its error message. So by default (HARNESS_FAITHFUL) an SWE
  injection is 1 to 3 consecutive copies of that message, with no failed reply. The other mode (with_failed_reply)
  also inserts the failed reply, which is the plan's original wording.

Sampling distributions:
- Error styles: v0's own format errors in the relay traces (422 in swesmith-relay and tmax-relay shard-00
  round 1; see the recovery README).
- Points per row and attempts per point: Marianna's mix (2 points in 355 of 400 rows; 1, 2 or 3 attempts at
  231, 258 and 264 of 753 points).
"""

import collections
import json
import random
import re
from dataclasses import dataclass

from experiments.post_training.sft_shootout_rows.terminus_json_plain_parser import TerminusJSONPlainParser
from experiments.post_training.sft_shootout_rows.traces import HISTORY, INJECTED, TEACHER, render_trace
from experiments.post_training.sweep_data.select import CONTEXT

FRACTION = 0.2
POINTS = {1: 45, 2: 355}
ATTEMPTS = {1: 231, 2: 258, 3: 264}
# v0's TMax format errors by the parser's error kind (329), and SWE failures by reply shape (93).
TMAX_STYLES = {"invalid_json": 144, "no_json": 136, "missing_fields": 45, "commands_not_array": 4}
SWE_STYLES = {"prose_no_call": 61, "text_tool_call": 32}
# mini-swe-agent 2.1.0's reply to a response without a parsable tool call, byte for byte as in the traces (all 93
# of v0's SWE failures in swesmith-relay shard-00 round 1 got exactly this text).
SWE_NO_TOOL_CALL = (
    "Tool call error:\n\n<error>\nNo tool calls found in the response. Every response MUST include at least one tool "
    "call.\n</error>\n\nHere is general guidance on how to submit correct toolcalls:\n\nEvery response needs to use "
    "the 'bash' tool at least once to execute commands.\n\nCall the bash tool with your command as the argument:\n- "
    'Tool: bash\n- Arguments: {"command": "your_command_here"}\n\nIf you want to end the task, please issue the '
    "following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command."
)
HARNESS_FAITHFUL = "harness_faithful"
WITH_FAILED_REPLY = "with_failed_reply"
PARSER = TerminusJSONPlainParser()


def weighted(rng: random.Random, weights: dict):
    keys = sorted(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys])[0]


def terminus_feedback(text: str) -> str | None:
    """Terminus-2's reply to `text` when the vendored parser finds an error, or None when the reply would parse."""
    result = PARSER.parse_response(text)
    if not result.error:
        return None
    feedback = f"ERROR: {result.error}"
    if result.warning:
        feedback += f"\nWARNINGS: {result.warning}"
    return (
        f"Previous response had parsing errors:\n{feedback}\n\n"
        "Please fix these issues and provide a proper JSON response."
    )


def terminus_fields(content: str) -> dict | None:
    """The teacher turn's JSON object when it is a clean Terminus-2 response (no parser error or warning)."""
    result = PARSER.parse_response(content)
    if result.error or result.warning:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("commands"), list) or not data["commands"]:
        return None
    return data


def tmax_failure(data: dict, style: str) -> str:
    """The teacher turn re-rendered in a v0 error style. v0 writes its reasoning as prose before any JSON."""
    prose = f"{data.get('analysis', '')}\n\n{data.get('plan', '')}".strip()
    if style == "no_json":
        return prose
    if style == "invalid_json":
        # An unquoted key: "Invalid JSON: Expecting property name enclosed in double quotes", as in v0's failures.
        return prose + "\n\n" + json.dumps(data, indent=2).replace('"plan":', "plan:", 1)
    if style == "missing_fields":
        # v0 often emits a bare command object, so the parser finds none of analysis, plan, commands.
        return prose + "\n\n" + json.dumps(data["commands"][0], indent=2)
    if style == "commands_not_array":
        return prose + "\n\n" + json.dumps(data | {"commands": data["commands"][0]}, indent=2)
    raise ValueError(f"Unknown TMax style {style!r}")


def swe_failure(message: dict, style: str) -> dict:
    """A v0-style SWE reply with no parsable tool call: prose only, or the call written as text inside the content."""
    failed = {"role": "assistant", "author": INJECTED, "content": message.get("content") or ""}
    if message.get("reasoning_content"):
        failed["reasoning_content"] = message["reasoning_content"]
    if style == "text_tool_call":
        call = message["tool_calls"][0]["function"]
        arguments = call["arguments"] if isinstance(call["arguments"], str) else json.dumps(call["arguments"])
        # Unclosed, so a Hermes-style parser cannot extract it (v0's text calls: closed 16, unclosed 10, mixed 6).
        failed["content"] += f'<tool_call>\n{{"name": "{call["name"]}", "arguments": {arguments}'
    elif style != "prose_no_call":
        raise ValueError(f"Unknown SWE style {style!r}")
    return failed


@dataclass(frozen=True)
class Injection:
    turn: int  # index of the teacher turn among the trace's assistant turns
    styles: tuple[str, ...]  # one per failed attempt


def eligible_turns(family: str, messages: list[dict]) -> list[int]:
    """Teacher turns that may be an injection point: not the first or the last, and re-renderable for TMax."""
    assistant = [m for m in messages if m["role"] == "assistant"]
    turns = []
    for index, message in enumerate(assistant[1:-1], start=1):
        if message.get("author") != TEACHER:
            continue
        if family == "tmax" and terminus_fields(message.get("content") or "") is None:
            continue
        if family == "swesmith" and not message.get("tool_calls"):
            continue
        turns.append(index)
    return turns


def plan_injections(rng: random.Random, family: str, messages: list[dict]) -> list[Injection]:
    turns = eligible_turns(family, messages)
    if not turns:
        return []
    count = min(weighted(rng, POINTS), len(turns))
    styles = TMAX_STYLES if family == "tmax" else SWE_STYLES
    return [
        Injection(turn, tuple(weighted(rng, styles) for _ in range(weighted(rng, ATTEMPTS))))
        for turn in sorted(rng.sample(turns, count))
    ]


def inject(
    family: str, messages: list[dict], injections: list[Injection], swe_mode: str = HARNESS_FAITHFUL
) -> tuple[list[dict], list[dict]]:
    """Insert each injection's failed attempts and harness replies before its teacher turn.

    Returns the messages and one record per failed attempt (turn, style used, harness error). A TMax style whose
    rendering the parser accepts falls back to invalid_json, which always fails (an unquoted key).
    """
    if swe_mode not in (HARNESS_FAITHFUL, WITH_FAILED_REPLY):
        raise ValueError(f"Unknown SWE mode {swe_mode!r}")
    by_turn = {injection.turn: injection for injection in injections}
    out, records, turn = [], [], -1
    for message in messages:
        if message["role"] == "assistant":
            turn += 1
            for style in by_turn[turn].styles if turn in by_turn else ():
                if family == "tmax":
                    data = terminus_fields(message["content"])
                    failed = tmax_failure(data, style)
                    feedback = terminus_feedback(failed)
                    if feedback is None:
                        style = "invalid_json"
                        failed = tmax_failure(data, style)
                        feedback = terminus_feedback(failed)
                    out.append({"role": "assistant", "author": INJECTED, "content": failed})
                    out.append({"role": "user", "content": feedback})
                    error = feedback.split("ERROR: ", 1)[1].split("\n", 1)[0].split(":", 1)[0]
                else:
                    if swe_mode == WITH_FAILED_REPLY:
                        out.append(swe_failure(message, style))
                    else:
                        style = "error_message_only"
                    out.append({"role": "user", "content": SWE_NO_TOOL_CALL})
                    error = "No tool calls found in the response"
                records.append({"turn": turn, "style": style, "error": error})
        out.append(message)
    return out, records


def selected_rows(trial_ids_by_family: dict[str, list[str]], seed: str) -> set[str]:
    """Exactly round(FRACTION * n) trials of each family, by a seeded shuffle of the sorted trial ids."""
    chosen = set()
    for family, trials in sorted(trial_ids_by_family.items()):
        order = sorted(trials)
        random.Random(f"{seed}:select:{family}").shuffle(order)
        chosen.update(order[: round(FRACTION * len(order))])
    return chosen


def build_recovery(chosen: dict, kept: dict, tokenizer, swe_mode: str, seed: str, recovered: dict) -> dict:
    """Inject into a seeded 20% of each family's chosen trajectories; fill `recovered` and return manifest stats.

    `chosen` maps task -> Attempt (plain's selection); `kept` maps trial id -> (Attempt, rendering, trace). An injected
    row must train exactly the plain row's tokens (checked); one that no longer fits the context keeps its plain row.
    """
    by_family = collections.defaultdict(list)
    for attempt in chosen.values():
        by_family[attempt.family].append(attempt.trial_id)
    selected = selected_rows(by_family, seed)
    stats = {
        family: {
            "rows": len(trials),
            "selected": 0,
            "injected": 0,
            "skipped_no_eligible_turn": 0,
            "skipped_over_context": 0,
            "points": collections.Counter(),
            "attempts_per_point": collections.Counter(),
            "styles": collections.Counter(),
            "errors": collections.Counter(),
            "context_tokens_added": 0,
        }
        for family, trials in sorted(by_family.items())
    }
    for trial in sorted(selected):
        attempt, rendered, trace = kept[trial]
        family = stats[attempt.family]
        family["selected"] += 1
        injections = plan_injections(random.Random(f"{seed}:{trial}"), attempt.family, trace["messages"])
        if not injections:
            family["skipped_no_eligible_turn"] += 1
            continue
        messages, records = inject(attempt.family, trace["messages"], injections, swe_mode)
        injected = render_trace(tokenizer, messages, trace.get("tools"))
        trained = [token for token, weight in zip(injected.ids, injected.mask, strict=True) if weight]
        if trained != [token for token, weight in zip(rendered.ids, rendered.mask, strict=True) if weight]:
            raise ValueError(f"Injected row {trial} trains different tokens than its plain row")
        if len(injected.ids) > CONTEXT:
            family["skipped_over_context"] += 1
            continue
        recovered[trial] = injected
        family["injected"] += 1
        family["points"][len(injections)] += 1
        for injection in injections:
            family["attempts_per_point"][len(injection.styles)] += 1
        for record in records:
            family["styles"][record["style"]] += 1
            family["errors"][record["error"]] += 1
        family["context_tokens_added"] += len(injected.ids) - len(rendered.ids)
    return {
        "seed": seed,
        "fraction": FRACTION,
        "swe_mode": swe_mode,
        "families": {
            name: (
                {key: dict(value) if isinstance(value, collections.Counter) else value for key, value in family.items()}
                | {"injected_share": family["injected"] / family["rows"] if family["rows"] else None}
            )
            for name, family in stats.items()
        },
        "sampling": {"points": POINTS, "attempts": ATTEMPTS, "tmax_styles": TMAX_STYLES, "swe_styles": SWE_STYLES},
        "rule": (
            "injection points are teacher turns other than the first and last (TMax: clean Terminus-2 JSON turns; "
            "SWE: turns with a tool call); an injected row trains exactly its plain row's tokens; a row over the "
            "context keeps its plain rendering"
        ),
    }


# Context-handoff rows (off by default; rule set by the queen on 2026-10-18 so the switch needs no design round).
HANDOFF_SHARE = 0.13
ECHOED_HISTORY = re.compile(r"record at message \d+ adds an assistant message with no logged reply")
HANDOFF_KINDS = (
    ("You are about to hand off your work", "summary"),
    ("Briefly summarize how another agent should continue", "short_summary"),
    ("You are picking up work from a previous AI agent", "questions"),
)


def handoff_traces(traces: list[dict]) -> list[dict]:
    """Conversations 1+ of summarized TMax plain trials, as candidate rows of their own.

    Echoed earlier turns (no author in the trace) become HISTORY, which is context; each conversation gets its own
    trial id ('<trial>#conversation-<n>') so it is a separate candidate.
    """
    out = []
    for trace in traces:
        if trace["family"] != "tmax" or trace["relay"] or trace.get("n_conversations", 1) < 2:
            continue
        if trace.get("conversation", 0) < 1:
            continue
        messages = [
            (
                message | {"author": HISTORY}
                if message["role"] == "assistant" and message.get("author") is None
                else message
            )
            for message in trace["messages"]
        ]
        if not any(m.get("author") == TEACHER and (m.get("content") or "").strip() for m in messages):
            continue
        # The rebuild flags every echoed history turn ("adds an assistant message with no logged reply"); those are
        # exactly the HISTORY turns. Any other problem still drops the conversation.
        problems = [p for p in trace["problems"] if not ECHOED_HISTORY.fullmatch(str(p))]
        trial = f"{trace['trial_id']}#conversation-{trace['conversation']}"
        out.append(trace | {"messages": messages, "problems": problems, "trial_id": trial})
    return out


def handoff_kind(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user":
            for prefix, kind in HANDOFF_KINDS:
                if prefix in (message.get("content") or ""):
                    return kind
    return "continuation"


def seeded(items: list, seed: str) -> list:
    order = sorted(items, key=lambda attempt: attempt.trial_id)
    random.Random(seed).shuffle(order)
    return order


def plan_handoff(candidates: list, family_rows: int, seed: str) -> list:
    """Handoff rows in seeded order, up to HANDOFF_SHARE of the family's rows.

    Not limited to plain's task slots: a task with a passing summarized trial has, in the dry build, no other passing
    trajectory that fits 32k, so it is never one of plain's slots, and such a limit would leave no handoff row.
    """
    return seeded(candidates, f"{seed}:handoff")[: round(HANDOFF_SHARE * family_rows)]


def hold_budget(clean: list, added: int, seed: str) -> list:
    """Clean rows of the same family to drop, in seeded order, until they cover the handoff rows' supervised tokens."""
    dropped, removed = [], 0
    for attempt in seeded(clean, f"{seed}:handoff-drop"):
        if removed >= added:
            break
        dropped.append(attempt)
        removed += attempt.supervised
    return dropped
