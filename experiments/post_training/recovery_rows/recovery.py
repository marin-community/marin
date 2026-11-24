# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recovery SFT rows: agent trajectories with synthetic failed attempts injected before some assistant turns.

An injection point is an assistant turn other than the first and the last. Before it go one or more failed attempts,
each followed by the harness's real reply to that failure. The model then sees the original turn as the recovery.
Every injected message carries ``INJECTED_KEY``; the caller masks those messages from the loss, so a row trains the
same tokens as the original trajectory and only the context grows.

Injected rows match what each deployed harness would send:

- Terminus-2 keeps a failed reply in its history and answers with its parser's feedback. A failed attempt is the
  original JSON turn re-rendered in a seeded variant of a measured error style. The reply is the prompt Terminus-2
  builds from the parser's error, using the parser vendored at ``vendor/harbor``.
- mini-swe-agent drops a reply it cannot parse before it enters history and sends only its format error. A failed
  attempt is therefore that error message alone, with no assistant turn.

The default weights come from measured failures: error styles and variants from one Terminus-2 student's own parser
errors on TBLite-100 (653 rejected turns), and points per row and attempts per point from the mix in
``marianna13/grug-sft-mix-v4-40M-think``.
"""

import importlib.util
import json
import random
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

INJECTED_KEY = "injected"

POINTS = {1: 45, 2: 355}
ATTEMPTS = {1: 231, 2: 258, 3: 264}
TERMINUS_STYLES = {"invalid_json": 144, "no_json": 136, "missing_fields": 45, "commands_not_array": 4}
TERMINUS_VARIANTS = {
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
# Expected parser error prefixes for planned Terminus-2 styles.
TERMINUS_ERRORS = {
    "no_json": "No valid JSON found",
    "invalid_json": "Invalid JSON",
    "missing_fields": "Missing required fields",
    "commands_not_array": "Field 'commands' must be an array",
}
# Probability that a later failure at one point keeps the previous failure's style (125 of 221 measured pairs).
TERMINUS_SAME_STYLE = 125 / 221
# mini-swe-agent 2.1.0's reply to a response without a parsable tool call, byte for byte.
MINI_SWE_ERROR = "No tool calls found in the response"
MINI_SWE_NO_TOOL_CALL = (
    f"Tool call error:\n\n<error>\n{MINI_SWE_ERROR}. Every response MUST include at least one tool "
    "call.\n</error>\n\nHere is general guidance on how to submit correct toolcalls:\n\nEvery response needs to use "
    "the 'bash' tool at least once to execute commands.\n\nCall the bash tool with your command as the argument:\n- "
    'Tool: bash\n- Arguments: {"command": "your_command_here"}\n\nIf you want to end the task, please issue the '
    "following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command."
)
MINI_SWE_STYLE = "no_tool_call"
ELIDED = "@@elided@@"
HARBOR_PARSER_PATH = (
    Path(__file__).resolve().parents[3] / "vendor/harbor/src/harbor/agents/terminus_2/terminus_json_plain_parser.py"
)


class Harness(StrEnum):
    TERMINUS_2 = "terminus-2"
    MINI_SWE_AGENT = "mini-swe-agent"


@dataclass(frozen=True)
class Failure:
    style: str
    seed: int  # seeds the rendering, so a failure's text does not depend on the other attempts kept


@dataclass(frozen=True)
class Injection:
    turn: int  # index of the original turn among the trajectory's assistant turns
    failures: tuple[Failure, ...]


@dataclass(frozen=True)
class InjectedFailure:
    turn: int
    style: str  # for Terminus-2, the style the parser's error names, which can differ from the planned style
    variant: str | None
    error: str


def _load_terminus_parser():
    # Harbor is not installed in the Marin workspace, and importing its package pulls in the whole Terminus-2 agent.
    # The parser file has no dependencies outside the standard library, so load it straight from the vendored tree.
    spec = importlib.util.spec_from_file_location("harbor_terminus_json_plain_parser", HARBOR_PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TerminusJSONPlainParser()


PARSER = _load_terminus_parser()


def weighted(rng: random.Random, weights: dict):
    keys = sorted(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys])[0]


def terminus_feedback(result) -> str:
    """Terminus-2's reply to a response whose parse `result` has an error."""
    feedback = f"ERROR: {result.error}"
    if result.warning:
        feedback += f"\nWARNINGS: {result.warning}"
    return (
        f"Previous response had parsing errors:\n{feedback}\n\n"
        "Please fix these issues and provide a proper JSON response."
    )


def terminus_fields(content: str) -> dict | None:
    """The turn's JSON object when it is a clean Terminus-2 response with at least one command."""
    result = PARSER.parse_response(content)
    if result.error or result.warning:
        return None
    data = json.loads(content)
    return data if data["commands"] else None


def terminus_prose(data: dict, rng: random.Random) -> str:
    """The turn's analysis and plan as a student writes them before, or instead of, its JSON."""
    analysis, plan = str(data.get("analysis") or "").strip(), str(data.get("plan") or "").strip()
    form = rng.randrange(4)
    if form == 1 and analysis and plan:
        return f"Analysis: {analysis}\nPlan: {plan}"
    if form == 2 and analysis:
        return analysis
    if form == 3 and plan:
        return plan
    return f"{analysis}\n\n{plan}".strip()


def terminus_broken_json(data: dict, rng: random.Random, variant: str) -> str | None:
    """Render an invalid response as `variant`, or return None if it does not apply."""
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
        # An unescaped backslash, as in a regex written straight into a string.
        return json.dumps(data, indent=indent).replace('"keystrokes": "', '"keystrokes": "grep -E \'\\d+\' ; ', 1)
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
        return json.dumps({k: v for k, v in data.items() if k != variant.removeprefix("no_")}, indent=indent)
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
    raise ValueError(f"Unknown Terminus-2 variant {variant!r}")


def terminus_variant(data: dict, style: str, variant: str, rng: random.Random) -> str | None:
    """One rendering of the turn in `style`'s `variant`: prose, then (except no_json) the broken JSON."""
    prose = terminus_prose(data, rng)
    if style == "no_json":
        commands = [str(c.get("keystrokes", "")).strip() for c in data["commands"] if isinstance(c, dict)]
        commands = [c for c in commands if c]
        if variant == "prose_bash_block" and commands:
            return prose + "\n\n```bash\n" + "\n".join(commands) + "\n```"
        if variant == "prose_next_command" and commands:
            return prose + "\n\nNext: " + commands[0].splitlines()[0]
        return prose
    text = terminus_broken_json(data, rng, variant)
    return None if text is None else prose + "\n\n" + text


def terminus_unquoted_plan(data: dict) -> str:
    """The turn's JSON with the `plan` key unquoted, which the parser always rejects as invalid JSON."""
    return json.dumps(data, indent=2).replace('"plan":', "plan:", 1)


def terminus_base_failure(data: dict, style: str) -> str:
    """The style's fixed rendering, used when no seeded variant produces the style's error."""
    prose = f"{data.get('analysis', '')}\n\n{data.get('plan', '')}".strip()
    if style == "no_json":
        return prose
    if style == "invalid_json":
        return prose + "\n\n" + terminus_unquoted_plan(data)
    if style == "missing_fields":
        return prose + "\n\n" + json.dumps(data["commands"][0], indent=2)
    if style == "commands_not_array":
        return prose + "\n\n" + json.dumps(data | {"commands": data["commands"][0]}, indent=2)
    raise ValueError(f"Unknown Terminus-2 style {style!r}")


def render_terminus_failure(data: dict, style: str, rng: random.Random, previous: str | None) -> tuple[str, str]:
    """(text, variant) for a failure the parser rejects with `style`'s error and that differs from `previous`.

    Variants are tried in a seeded, weighted order, a few renders each. When none qualifies, the style's base rendering
    is used (variant "base"), then the broken JSON alone (variant "fallback_invalid_json"), which may equal `previous`.
    """
    weights = dict(TERMINUS_VARIANTS[style])
    while weights:
        variant = weighted(rng, weights)
        del weights[variant]
        for _ in range(3):
            text = terminus_variant(data, style, variant, rng)
            if text is None:
                break
            if PARSER.parse_response(text).error.startswith(TERMINUS_ERRORS[style]) and text != previous:
                return text, variant
    text = terminus_base_failure(data, style)
    if PARSER.parse_response(text).error and text != previous:
        return text, "base"
    # No prose here: the parser takes the first balanced object, so an analysis that quotes a valid response makes
    # every rendering with prose parse clean.
    text = terminus_unquoted_plan(data)
    assert PARSER.parse_response(text).error, text
    return text, "fallback_invalid_json"


def eligible_turns(harness: Harness, messages: list[dict]) -> list[int]:
    """Assistant turns that may be an injection point: not the first or the last, and re-renderable as a failure."""
    assistant = [m for m in messages if m["role"] == "assistant"]
    turns = []
    for index, message in enumerate(assistant[1:-1], start=1):
        if harness == Harness.TERMINUS_2 and terminus_fields(message.get("content") or "") is None:
            continue
        if harness == Harness.MINI_SWE_AGENT and not message.get("tool_calls"):
            continue
        turns.append(index)
    return turns


def plan_failures(rng: random.Random, harness: Harness, attempts: int) -> tuple[Failure, ...]:
    """The failed attempts at one point. A later Terminus-2 failure keeps the previous style with
    TERMINUS_SAME_STYLE, else takes another style by weight."""
    if harness == Harness.MINI_SWE_AGENT:
        return tuple(Failure(MINI_SWE_STYLE, rng.getrandbits(32)) for _ in range(attempts))
    failures = [Failure(weighted(rng, TERMINUS_STYLES), rng.getrandbits(32))]
    for _ in range(attempts - 1):
        previous = failures[-1].style
        if rng.random() < TERMINUS_SAME_STYLE:
            style = previous
        else:
            style = weighted(rng, {k: v for k, v in TERMINUS_STYLES.items() if k != previous})
        failures.append(Failure(style, rng.getrandbits(32)))
    return tuple(failures)


def plan_injections(rng: random.Random, harness: Harness, messages: list[dict]) -> list[Injection]:
    turns = eligible_turns(harness, messages)
    if not turns:
        return []
    count = min(weighted(rng, POINTS), len(turns))
    return [
        Injection(turn, plan_failures(rng, harness, weighted(rng, ATTEMPTS)))
        for turn in sorted(rng.sample(turns, count))
    ]


def smaller_plans(injections: list[Injection]) -> Iterator[list[Injection]]:
    """The planned injections, then ever smaller ones, for a row that does not fit the context.

    Each step removes one failed attempt from the point with the most (the later point on a tie). Once every point
    has one attempt, it drops the last point. The last plan yielded is one attempt at the first point.
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
    harness: Harness, messages: list[dict], injections: list[Injection]
) -> tuple[list[dict], list[InjectedFailure]]:
    """Insert each injection's failed attempts and harness replies before its turn; return messages and records."""
    by_turn = {injection.turn: injection for injection in injections}
    out, records, turn = [], [], -1
    for message in messages:
        if message["role"] == "assistant":
            turn += 1
            previous = None
            for failure in by_turn[turn].failures if turn in by_turn else ():
                if harness == Harness.MINI_SWE_AGENT:
                    out.append({"role": "user", "content": MINI_SWE_NO_TOOL_CALL, INJECTED_KEY: True})
                    records.append(InjectedFailure(turn, MINI_SWE_STYLE, None, MINI_SWE_ERROR))
                    continue
                failed, variant = render_terminus_failure(
                    terminus_fields(message["content"]), failure.style, random.Random(failure.seed), previous
                )
                previous = failed
                result = PARSER.parse_response(failed)
                out.append({"role": "assistant", "content": failed, INJECTED_KEY: True})
                out.append({"role": "user", "content": terminus_feedback(result), INJECTED_KEY: True})
                error = result.error.split(":", 1)[0]
                # Label by the parser's actual error: a fallback render, or prose with braces in it ("{3}"), can
                # draw another error than the planned style's.
                style = next((s for s, e in TERMINUS_ERRORS.items() if error.startswith(e)), failure.style)
                records.append(InjectedFailure(turn, style, variant, error))
        out.append(message)
    return out, records
