# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recovery rows: plain's trajectories with masked synthetic failed attempts before some teacher turns.

A reconstruction of Marianna's recovery SFT (marianna13/grug-sft-mix-v4-40M-think) for the shootout. A fixed, seeded
20% of each family's plain rows get 1 or 2 injection points. An injection point is a teacher turn other than the
first and the last. Before it go 1 to 3 failed attempts, each followed by the harness's real reply to that failure.
Injected turns and replies are context (zero loss). Every teacher turn trains exactly as in plain, so a row's
supervised tokens equal its plain row's; only context grows.

Each injected row must match what the deployed harness would send:
- Terminus-2 (TMax) keeps a failed reply in history and answers with its parser's feedback. The failed attempt is
  the teacher turn re-rendered in one of v0's measured error styles, in a seeded variant of that style (the prose
  form, the JSON layout and how the JSON breaks vary), so failures are not one fixed text. The reply is the exact
  prompt Terminus-2 builds from the vendored parser's error and warning (terminus_2.py:2029-2035, 2273-2278).
- mini-swe-agent 2.1.0 (SWE-smith) drops a reply it cannot parse, raising FormatError before the reply enters
  history (default.py:113-116), and sends only its error message. So by default (HARNESS_FAITHFUL) an SWE
  injection is 1 to 3 consecutive copies of that message, with no failed reply. The other mode (with_failed_reply)
  also inserts the failed reply, which is the plan's original wording.

Sampling distributions:
- Error styles: v0's own format errors in the relay traces (422 in swesmith-relay and tmax-relay shard-00
  round 1; see the recovery README).
- TMax variants within a style: v0's parser errors in its own TBLite-100 run (653; see TMAX_VARIANTS).
- TMax repeats within a point: v0's consecutive rejected turns in that run (221 pairs: the next failure keeps the
  style in 125, and is never word for word the previous one). SWE failed replies cannot be measured (mini-swe-agent
  logs drop them), so SWE styles are drawn independently; the default SWE mode has no failed reply anyway.
- Points per row and attempts per point: Marianna's mix (2 points in 355 of 400 rows; 1, 2 or 3 attempts at
  231, 258 and 264 of 753 points).
"""

import collections
import itertools
import json
import random
import re
from dataclasses import dataclass, replace

from experiments.post_training.sft_shootout_rows.terminus_json_plain_parser import TerminusJSONPlainParser
from experiments.post_training.sft_shootout_rows.traces import HISTORY, INJECTED, TEACHER, render_trace
from experiments.post_training.sweep_data.select import CONTEXT

FRACTION = 0.2
POINTS = {1: 45, 2: 355}
ATTEMPTS = {1: 231, 2: 258, 3: 264}
# v0's TMax format errors by the parser's error kind (329), and SWE failures by reply shape (93).
TMAX_STYLES = {"invalid_json": 144, "no_json": 136, "missing_fields": 45, "commands_not_array": 4}
SWE_STYLES = {"prose_no_call": 61, "text_tool_call": 32}
# How v0's TMax failures break, per style, from the parser errors in v0's TBLite-100 reference run
# (task208-v0-reference, 2026-10-18; 653 rejected turns). Invalid JSON: unquoted key ("Expecting property name") 202,
# elided value ("Expecting value") 52, missing comma 14, invalid escape 10, raw control character 4 (the 3 "Expecting
# ':' delimiter" are left out). Missing fields: a bare command object 122, no plan 19, commands only 6, analysis only
# 4, no analysis 1. commands_not_array has no breakdown (31), so its variants are equal.
TMAX_VARIANTS = {
    "no_json": {"prose": 1, "prose_bash_block": 1, "prose_next_command": 1},
    "invalid_json": {
        "unquoted_key": 202,
        "elided_value": 52,
        "missing_comma": 14,
        "invalid_escape": 10,
        "raw_newline": 4,
    },
    "missing_fields": {"bare_command": 122, "no_plan": 19, "commands_only": 6, "analysis_only": 4, "no_analysis": 1},
    "commands_not_array": {"single_object": 1, "null": 1, "string": 1},
}
# The parser error each TMax style must produce.
TMAX_ERRORS = {
    "no_json": "No valid JSON found",
    "invalid_json": "Invalid JSON",
    "missing_fields": "Missing required fields",
    "commands_not_array": "Field 'commands' must be an array",
}
# Consecutive TMax failures at one point, measured on the same v0 run: of 221 pairs of consecutive rejected turns,
# 125 keep the style and 0 repeat the previous text (none even after whitespace normalization).
TMAX_REPEAT = {
    "same_style": 125 / 221,
    "identical_text": 0 / 221,
    "pairs": 221,
    "source": "v0 TBLite-100 reference run (task208-v0-reference-openthoughts-tblite-100-6263), Terminus-2, no takeover",
}
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
# The agent harness a trace ran under. v7 generation traces carry no harness field: SWE-smith ran mini-swe-agent 2.1.0
# and TMax ran Terminus-2. Contributor traces (contributor.py) name theirs: Kimi SWE-smith ran Terminus-2, and Qwen
# TMax ran mini-swe-agent with a `bash` tool call per turn, which keeps a rejected reply and answers "Format error".
TERMINUS_2 = "terminus-2"
MINI_SWE_AGENT = "mini-swe-agent-2.1"
MINI_SWE_FORMAT_ERROR = "mini-swe-agent-bash-tool"
V7_HARNESS = {"swesmith": MINI_SWE_AGENT, "tmax": TERMINUS_2}


def harness(trace: dict) -> str:
    return trace.get("harness") or V7_HARNESS[trace["family"]]
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
    """The style's base rendering, the fallback when no variant gives the style's error. v0 writes prose first."""
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


ELIDED = "@@elided@@"


def tmax_prose(data: dict, rng: random.Random) -> str:
    """The turn's analysis and plan as v0 writes them before (or instead of) its JSON, in one of four forms."""
    analysis, plan = str(data.get("analysis") or "").strip(), str(data.get("plan") or "").strip()
    form = rng.randrange(4)
    if form == 1 and analysis and plan:
        return f"Analysis: {analysis}\nPlan: {plan}"
    if form == 2 and analysis:
        return analysis
    if form == 3 and plan:
        return plan
    return f"{analysis}\n\n{plan}".strip()


def tmax_json(data: dict, rng: random.Random, variant: str) -> str | None:
    """The turn's JSON broken as `variant` says, in an indented or a compact layout; None if the turn cannot take it."""
    indent = rng.choice((2, None))
    keys = list(data)
    if variant == "unquoted_key":
        key = rng.choice(keys)
        return re.sub(rf'"{re.escape(key)}":', f"{key}:", json.dumps(data, indent=indent), count=1)
    if variant == "elided_value":
        key = rng.choice(keys)
        return json.dumps(data | {key: ELIDED}, indent=indent).replace(f'"{ELIDED}"', "...")
    if variant == "missing_comma":
        if len(keys) < 2:
            return None
        key = rng.choice(keys[1:])
        return re.sub(rf',\s*"{re.escape(key)}":', f'\n"{key}":', json.dumps(data, indent=indent), count=1)
    if variant == "invalid_escape":
        # An unescaped backslash, as in a regex or a Windows path written straight into a string.
        text = json.dumps(data, indent=indent)
        return text.replace('"keystrokes": "', '"keystrokes": "grep -E \'\\d+\' ; ', 1)
    if variant == "raw_newline":
        analysis = str(data.get("analysis") or "")
        if " " not in analysis:
            return None
        cut = rng.choice([i for i, ch in enumerate(analysis) if ch == " "])
        halves = [json.dumps(part)[1:-1] for part in (analysis[:cut], analysis[cut + 1 :])]
        return json.dumps(data | {"analysis": ELIDED}, indent=indent).replace(ELIDED, "\n".join(halves))
    if variant == "bare_command":
        return json.dumps(rng.choice(data["commands"]), indent=indent)
    if variant in ("no_plan", "no_analysis"):
        return json.dumps({k: v for k, v in data.items() if k != variant[3:]}, indent=indent)
    if variant == "commands_only":
        return json.dumps({k: v for k, v in data.items() if k not in ("analysis", "plan")}, indent=indent)
    if variant == "analysis_only":
        return json.dumps({k: v for k, v in data.items() if k not in ("plan", "commands")}, indent=indent)
    if variant == "single_object":
        return json.dumps(data | {"commands": rng.choice(data["commands"])}, indent=indent)
    if variant == "null":
        return json.dumps(data | {"commands": None}, indent=indent)
    if variant == "string":
        keystrokes = "".join(str(c.get("keystrokes", "")) for c in data["commands"] if isinstance(c, dict))
        return json.dumps(data | {"commands": keystrokes}, indent=indent)
    raise ValueError(f"Unknown TMax variant {variant!r}")


def tmax_variant(data: dict, style: str, variant: str, rng: random.Random) -> str | None:
    """One rendering of the teacher turn in `style`'s `variant`: v0's prose, then (except no_json) the broken JSON."""
    prose = tmax_prose(data, rng)
    if style == "no_json":
        commands = [str(c.get("keystrokes", "")).strip() for c in data["commands"] if isinstance(c, dict)]
        commands = [c for c in commands if c]
        if variant == "prose_bash_block" and commands:
            return prose + "\n\n```bash\n" + "\n".join(commands) + "\n```"
        if variant == "prose_next_command" and commands:
            return prose + "\n\nNext: " + commands[0].splitlines()[0]
        return prose
    text = tmax_json(data, rng, variant)
    return None if text is None else prose + "\n\n" + text


def render_tmax_failure(data: dict, style: str, rng: random.Random, previous: str | None) -> tuple[str, str]:
    """(text, variant): a seeded variant of `style` that the parser rejects with the style's error and that differs
    from the previous failure at this point. Variants are tried in a seeded, weighted order, a few renders each; when
    none qualifies, the style's base rendering is used (variant 'base'), then the always-failing unquoted plan key."""
    weights = dict(TMAX_VARIANTS[style])
    while weights:
        variant = weighted(rng, weights)
        del weights[variant]
        for _ in range(3):
            text = tmax_variant(data, style, variant, rng)
            if text is None:
                break
            feedback = terminus_feedback(text)
            if feedback and feedback.split("ERROR: ", 1)[1].startswith(TMAX_ERRORS[style]) and text != previous:
                return text, variant
    text = tmax_failure(data, style)
    if terminus_feedback(text) and text != previous:
        return text, "base"
    return tmax_failure(data, "invalid_json"), "fallback_invalid_json"


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
class Failure:
    style: str
    seed: int  # seeds the variant of the style, so a row's text does not depend on the other attempts kept
    repeat: bool = False  # word for word the previous failure at this point


@dataclass(frozen=True)
class Injection:
    turn: int  # index of the teacher turn among the trace's assistant turns
    failures: tuple[Failure, ...]  # one per failed attempt


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


def plan_failures(rng: random.Random, family: str, attempts: int) -> tuple[Failure, ...]:
    """The failed attempts at one point. TMax follows v0's measured repeats: each later failure repeats the previous
    text with p = identical_text, else keeps its style so that p(same style) = same_style overall, else takes another
    style by weight. SWE styles are drawn independently (v0's SWE failed replies are not logged)."""
    if family != "tmax":
        return tuple(Failure(weighted(rng, SWE_STYLES), rng.getrandbits(32)) for _ in range(attempts))
    identical, same = TMAX_REPEAT["identical_text"], TMAX_REPEAT["same_style"]
    keep = (same - identical) / (1 - identical)
    failures = [Failure(weighted(rng, TMAX_STYLES), rng.getrandbits(32))]
    for _ in range(attempts - 1):
        previous = failures[-1]
        if rng.random() < identical:
            failures.append(Failure(previous.style, previous.seed, repeat=True))
            continue
        if rng.random() < keep:
            style = previous.style
        else:
            style = weighted(rng, {k: v for k, v in TMAX_STYLES.items() if k != previous.style})
        failures.append(Failure(style, rng.getrandbits(32)))
    return tuple(failures)


def plan_injections(rng: random.Random, family: str, messages: list[dict]) -> list[Injection]:
    turns = eligible_turns(family, messages)
    if not turns:
        return []
    count = min(weighted(rng, POINTS), len(turns))
    return [
        Injection(turn, plan_failures(rng, family, weighted(rng, ATTEMPTS))) for turn in sorted(rng.sample(turns, count))
    ]


def smaller_plans(injections: list[Injection]):
    """The planned injections, then ever smaller ones, for a row whose full injection does not fit the context.

    Each step removes one failed attempt from the point with the most (the later point on a tie); once every point has
    one attempt, it drops the last point. The last plan yielded is one attempt at the first point.
    """
    plan = list(injections)
    while plan:
        yield plan
        most = max(range(len(plan)), key=lambda i: (len(plan[i].failures), i))
        if len(plan[most].failures) > 1:
            plan = [*plan[:most], replace(plan[most], failures=plan[most].failures[:-1]), *plan[most + 1 :]]
        else:
            plan = plan[:-1]


def inject(
    family: str, messages: list[dict], injections: list[Injection], swe_mode: str = HARNESS_FAITHFUL
) -> tuple[list[dict], list[dict]]:
    """Insert each injection's failed attempts and harness replies before its teacher turn.

    Returns the messages and one record per failed attempt (turn, style used, variant, repeat, harness error). A TMax
    failure is a seeded variant of its style (render_tmax_failure); a repeat is the previous failure word for word.
    """
    if swe_mode not in (HARNESS_FAITHFUL, WITH_FAILED_REPLY):
        raise ValueError(f"Unknown SWE mode {swe_mode!r}")
    by_turn = {injection.turn: injection for injection in injections}
    out, records, turn = [], [], -1
    for message in messages:
        if message["role"] == "assistant":
            turn += 1
            previous = None
            for failure in by_turn[turn].failures if turn in by_turn else ():
                style = failure.style
                if family == "tmax":
                    if failure.repeat and previous is not None:
                        failed, variant = previous
                    else:
                        data = terminus_fields(message["content"])
                        failed, variant = render_tmax_failure(
                            data, style, random.Random(failure.seed), previous[0] if previous else None
                        )
                    previous = (failed, variant)
                    feedback = terminus_feedback(failed)
                    out.append({"role": "assistant", "author": INJECTED, "content": failed})
                    out.append({"role": "user", "content": feedback})
                    error = feedback.split("ERROR: ", 1)[1].split("\n", 1)[0].split(":", 1)[0]
                    # The style is the parser's error: a fallback render, or prose with braces in it ("{3}"), can
                    # draw another error than the planned style's.
                    style = next((s for s, e in TMAX_ERRORS.items() if error.startswith(e)), style)
                else:
                    variant = None
                    if swe_mode == WITH_FAILED_REPLY:
                        out.append(swe_failure(message, style))
                    else:
                        style = "error_message_only"
                    out.append({"role": "user", "content": SWE_NO_TOOL_CALL})
                    error = "No tool calls found in the response"
                records.append(
                    {"turn": turn, "style": style, "variant": variant, "repeat": failure.repeat, "error": error}
                )
        out.append(message)
    return out, records


def failed_texts(messages: list[dict]) -> list[list[str]]:
    """The failed replies at each injection point, in order (consecutive injected turns form one point)."""
    points, run = [], []
    for message in messages:
        if message.get("author") == INJECTED:
            run.append(message["content"])
        elif message["role"] == "assistant":
            if run:
                points.append(run)
            run = []
    return points


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
    row must train exactly the plain row's tokens (checked). A row whose planned injection does not fit the context
    takes the largest of `smaller_plans` that fits; one that cannot fit even one failed attempt keeps its plain row.
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
            "reduced_to_fit": 0,
            "points": collections.Counter(),
            "attempts_per_point": collections.Counter(),
            "styles": collections.Counter(),
            "variants": collections.Counter(),
            "errors": collections.Counter(),
            "consecutive_pairs": 0,
            "pairs_same_style": 0,
            "pairs_identical_text": 0,
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
        plain_trained = [token for token, weight in zip(rendered.ids, rendered.mask, strict=True) if weight]
        for plan in smaller_plans(injections):
            messages, records = inject(attempt.family, trace["messages"], plan, swe_mode)
            injected = render_trace(tokenizer, messages, trace.get("tools"))
            trained = [token for token, weight in zip(injected.ids, injected.mask, strict=True) if weight]
            if trained != plain_trained:
                raise ValueError(f"Injected row {trial} trains different tokens than its plain row")
            if len(injected.ids) <= CONTEXT:
                break
        else:
            family["skipped_over_context"] += 1
            continue
        family["reduced_to_fit"] += plan != injections
        injections = plan
        recovered[trial] = injected
        family["injected"] += 1
        family["points"][len(injections)] += 1
        for injection in injections:
            family["attempts_per_point"][len(injection.failures)] += 1
        for record in records:
            family["styles"][record["style"]] += 1
            family["variants"][f"{record['style']}/{record['variant']}"] += 1
            family["errors"][record["error"]] += 1
        texts = failed_texts(messages)
        for before, after in itertools.pairwise(records):
            if before["turn"] == after["turn"]:
                family["consecutive_pairs"] += 1
                family["pairs_same_style"] += before["style"] == after["style"]
        for point in texts:
            family["pairs_identical_text"] += sum(a == b for a, b in itertools.pairwise(point))
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
        "sampling": {
            "points": POINTS,
            "attempts": ATTEMPTS,
            "tmax_styles": TMAX_STYLES,
            "tmax_variants": TMAX_VARIANTS,
            "tmax_repeat": TMAX_REPEAT,
            "swe_styles": SWE_STYLES,
            "swe_repeat": "independent draws: mini-swe-agent logs drop v0's failed replies, so no rate can be measured",
        },
        "rule": (
            "injection points are teacher turns other than the first and last (TMax: clean Terminus-2 JSON turns; "
            "SWE: turns with a tool call); an injected row trains exactly its plain row's tokens; a row over the "
            "context drops one attempt at a time (from the point with the most, the later on a tie), then the last "
            "point, until it fits; a row that cannot fit one attempt keeps its plain rendering"
        ),
    }


# Context-handoff rows, off by default. The rule is fixed here so that turning the switch on changes no other choice.
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
