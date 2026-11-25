# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit fixed evaluation cohorts: agent action loops, token tail loops, and trial metrics.

The token tail detector is Dr Doom's exact ``tail_loop`` (#12240), reused unchanged. The
agent-loop rules (#12264 §1) run over each retained trial's parsed conversation: the
assistant messages are the action batches and the messages between them are the
observations. The audit reads the ``dumped_evals`` JSONL rows of one saved session per
cohort, matched against the frozen cohort manifest.
"""

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.config.reward_shaping import LoopCreditConfig
from skyrl_train.trajectory_runners.trajectory_reward_shaping import tail_loop
from skyrlrecipe.cohort_identity import cohort_hash, prompt_hash
from verifyit.tool_call_markup import THINK_CLOSE, THINK_OPEN

from rolloutengine.models import ATTEMPT_SECONDS_METRIC, REPEAT_STALL_STOP_REASON
from rolloutengine.repetition import (
    action_identity,
    duplicate_tool_call_fraction,
    normalize_text,
    observation_identity,
    parse_commands,
    wait_only,
)

MINIMUM_TARGET_LOOPS = 10
"""The low-incidence rule: a cohort under this many confirmed looping trials is non-target."""
TAIL_MINIMUM_OCCURRENCES = 8
"""The frozen tail-loop detector's occurrence threshold: ``LoopCreditConfig(4096, 32768, 8)``."""

STALL_BATCHES = 3
"""A stall is three consecutive identical action and observation batches."""
RUN_LENGTH = 8
"""A run loop is at least this many consecutive identical actions."""
FREQUENT_COUNT = 10
"""A frequent action repeats at least this many times in one trial."""
FREQUENT_SHARE = 0.3
"""A frequent action also covers at least this share of the trial's assistant turns."""
MESSAGE_RUN_LENGTH = 8
"""A message loop is at least this many consecutive identical normalized assistant messages."""
MESSAGE_OBSERVATION_SHARE = 0.8
"""The share of a message run that one repeated observation must cover."""
INTERRUPT_LENGTH = 3
"""A Terminal-Bench interrupt loop is at least this many consecutive interrupt turns."""

STALL = "stall"
RUN8 = "run8"
FREQUENT = "frequent"
TERMINUS8 = "terminus8"
INTERRUPT = "interrupt"
TAIL = "tail"
AGENT_RULES = (STALL, RUN8, FREQUENT, TERMINUS8)
"""An agent loop is any of the four action-loop rules; ``interrupt`` and ``tail`` report separately."""
GRADED_AGENT_EXCEPTIONS = frozenset({"ContextLengthExceededError", "AgentTimeoutError", "AgentTimeout"})


@dataclass(frozen=True)
class TrialTurn:
    """One assistant turn of a retained trial, with the observation batch that followed it."""

    action: str
    """The canonical action-batch identity of the assistant message."""
    observation: str
    """The canonical observation-batch identity of the messages that follow the assistant message."""
    message: str
    """The normalized text of the assistant message, for the message-run rule."""
    keystrokes: str
    """The Terminus keystrokes of the message, or ``""`` when it carries none."""
    wait: bool
    """Whether the action batch only waits."""

    @classmethod
    def of(cls, message: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> "TrialTurn":
        content = message.get("content")
        commands = parse_commands(content)
        if commands is not None:
            keystrokes = "\x1f".join(commands.keystrokes)
        else:
            keystrokes = ""
        text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)
        return cls(
            action_identity(message),
            observation_identity(observations),
            normalize_text(text or ""),
            keystrokes,
            wait_only(message),
        )


@dataclass(frozen=True)
class LoopRun:
    """One maximal run that fired a rule: the rule, the turn span, and the action."""

    rule: str
    start: int
    length: int
    action: str


def trial_turns(messages: Sequence[Mapping[str, Any]]) -> list[TrialTurn]:
    """Split a retained conversation into assistant turns with their following observations."""
    turns = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        end = next(
            (stop for stop in range(index + 1, len(messages)) if messages[stop].get("role") == "assistant"),
            len(messages),
        )
        turns.append(TrialTurn.of(message, messages[index + 1 : end]))
    return turns


def _identical_runs(turns: Sequence[TrialTurn], *, observations: bool) -> list[tuple[int, int]]:
    """The maximal spans of equal consecutive turns, keyed on actions or on action plus observation."""
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(turns) + 1):
        same = (
            index < len(turns)
            and turns[index].action == turns[start].action
            and (not observations or turns[index].observation == turns[start].observation)
        )
        if not same:
            if index - start > 1:
                runs.append((start, index - start))
            start = index
    return runs


def _message_runs(turns: Sequence[TrialTurn]) -> list[tuple[int, int]]:
    """The maximal spans of equal consecutive normalized assistant messages."""
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(turns) + 1):
        same = index < len(turns) and turns[index].message == turns[start].message
        if not same:
            if index - start > 1:
                runs.append((start, index - start))
            start = index
    return runs


def _observation_covers_run(turns: Sequence[TrialTurn], start: int, length: int) -> bool:
    """Whether one observation batch precedes at least the required share of a run's turns."""
    counts: dict[str, int] = {}
    for index in range(start + 1, start + length):
        counts[turns[index].observation] = counts.get(turns[index].observation, 0) + 1
    if not counts:
        return False
    return max(counts.values()) >= MESSAGE_OBSERVATION_SHARE * length


def _interrupt_turn(turn: TrialTurn) -> bool:
    """Whether one turn's Terminus keystrokes interrupt the running command.

    ``C-c`` alone, or ``C-c`` plus a re-sent command, is an interrupt attempt; the run rule
    also requires the re-sent command, when present, to be identical across the run.
    """
    if not turn.keystrokes:
        return False
    prefix, separator, resent = turn.keystrokes.partition("\x1f")
    return prefix == "C-c" and (not separator or not resent or bool(resent))


def _interrupt_runs(turns: Sequence[TrialTurn]) -> list[tuple[int, int]]:
    """The maximal spans of interrupt turns with an unchanged observation."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, turn in enumerate(turns):
        continuing = (
            start is not None
            and _interrupt_turn(turn)
            and turn.observation == turns[start].observation
            and turn.keystrokes == turns[start].keystrokes
        )
        if _interrupt_turn(turn) and start is None:
            start = index
        if not continuing and start is not None:
            if index - start >= INTERRUPT_LENGTH:
                runs.append((start, index - start))
            start = index if _interrupt_turn(turn) else None
    if start is not None and len(turns) - start >= INTERRUPT_LENGTH:
        runs.append((start, len(turns) - start))
    return runs


def find_loop_runs(turns: Sequence[TrialTurn]) -> list[LoopRun]:
    """Fire the four agent-loop rules and the Terminal-Bench interrupt rule over one trial's turns."""
    runs: list[LoopRun] = []
    for start, length in _identical_runs(turns, observations=True):
        # The stall rule exempts wait-only actions; the other rules measure them as-is.
        if turns[start].wait:
            continue
        if length >= STALL_BATCHES:
            runs.append(LoopRun(STALL, start, length, turns[start].action))
    counts: dict[str, int] = {}
    for turn in turns:
        if not turn.wait:
            counts[turn.action] = counts.get(turn.action, 0) + 1
    frequent = {
        action
        for action, count in counts.items()
        if count >= FREQUENT_COUNT and count >= FREQUENT_SHARE * len(turns)
    }
    for start, length in _identical_runs(turns, observations=False):
        if length >= RUN_LENGTH:
            runs.append(LoopRun(RUN8, start, length, turns[start].action))
        if length >= STALL_BATCHES and turns[start].action in frequent:
            runs.append(LoopRun(FREQUENT, start, length, turns[start].action))
    for start, length in _message_runs(turns):
        if length >= MESSAGE_RUN_LENGTH and _observation_covers_run(turns, start, length):
            runs.append(LoopRun(TERMINUS8, start, length, turns[start].message))
    for start, length in _interrupt_runs(turns):
        runs.append(LoopRun(INTERRUPT, start, length, turns[start].keystrokes))
    return runs


@dataclass(frozen=True)
class TrialAudit:
    """Per-trial loop evidence: which rules fired, repeat turns, and the duplicate-call fraction."""

    rules: frozenset[str]
    """The agent-loop rules that fired; ``interrupt`` joins when the cohort measures it."""
    repeat_turns: int
    """Turns inside agent-loop runs beyond each run's first copy."""
    interrupt_repeat_turns: int
    """Turns inside interrupt runs beyond each run's first copy."""
    assistant_turns: int
    duplicate_fraction: float | None
    """The exact-duplicate tool-call fraction of the trial; ``None`` with no tool calls."""


def audit_trial(row: Mapping[str, Any]) -> TrialAudit:
    turns = trial_turns(row.get("messages") or ())
    runs = find_loop_runs(turns)
    rules = {run.rule for run in runs}
    repeat_turns = {
        index
        for run in runs
        if run.rule in AGENT_RULES
        for index in range(run.start + 1, run.start + run.length)
    }
    if row.get("stop_reason") == REPEAT_STALL_STOP_REASON:
        rules.add(STALL)
        repeat_turns.update(range(max(0, len(turns) - 2), len(turns)))
    interrupt_repeat = sum(run.length - 1 for run in runs if run.rule == INTERRUPT)
    messages = row.get("messages") or ()
    calls = duplicate_tool_call_fraction(messages)
    return TrialAudit(frozenset(rules), len(repeat_turns), interrupt_repeat, len(turns), calls)


@dataclass(frozen=True)
class Incidence:
    """Exact loop counts, including UB, the confirmed-plus-unassessed upper bound."""

    N: int
    confirmed: int
    unassessed: int
    UB: int
    passes: int
    unavailable: int
    errors: int
    length_stop: int
    open_think: int
    length_p50: float | None
    length_p90: float | None
    length_p99: float | None


@dataclass(frozen=True)
class TrialCost:
    """Per-trial cost medians over the assessed rows: turns, generated tokens, and wall seconds."""

    turns_p50: float | None
    tokens_p50: float | None
    wall_seconds_p50: float | None


@dataclass(frozen=True)
class AgentIncidence:
    """The agent-loop share metrics of one cohort, over its assessed rows."""

    loop_trials: int
    """Assessed trials where at least one agent-loop rule fired."""
    trial_share: float
    trial_share_ub: float
    repeat_turn_share: float
    dup_tool_call_frac_median: float | None
    dup_tool_call_frac_p90: float | None
    interrupt_trial_share: float | None
    interrupt_repeat_share: float | None
    rule_fires: dict[str, int] = field(default_factory=dict)
    """Trials where each rule fired, ``tail`` included, with the tail overlap of each agent rule."""
    rule_tail_overlap: dict[str, int] = field(default_factory=dict)
    mean_reward: float | None = None
    stop_reasons: dict[str, int] = field(default_factory=dict)
    cost: TrialCost | None = None


def _outcome_reward(score: Any) -> float | None:
    """The row's outcome reward: a scalar score, or the last element of a per-token list."""
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, list) and score and all(isinstance(value, (int, float)) for value in score):
        return float(score[-1])
    return None


def _wall_seconds(row: Mapping[str, Any]) -> float | None:
    metrics = row.get("env_metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(ATTEMPT_SECONDS_METRIC)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def audit_cohort(cohort: Mapping[str, Any], rows: Iterable[dict[str, Any]]) -> tuple[Incidence, AgentIncidence]:
    """Count missing or damaged generations in the loop upper bound, never as passes.

    Damaged trials also stay out of the agent-loop confirmed count and of every share
    denominator except the upper bound, which keeps them as unassessed.
    """
    tasks = {row["task_id"]: row.get("prompt_hash") for row in cohort["rows"]}
    if len(tasks) != len(cohort["rows"]):
        raise ValueError("cohort contains duplicate task IDs")
    trials = cohort.get("trials_per_task", 1)
    expected = {(task_id, trial): digest for task_id, digest in tasks.items() for trial in range(trials)}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id not in tasks:
            continue
        trial = counts.get(task_id, 0)
        if trial >= trials:
            raise ValueError(f"multiple generations for cohort task {task_id}; select one evaluation session")
        indexed[task_id, trial] = row
        counts[task_id] = trial + 1
    config = LoopCreditConfig(
        max_period_tokens=cohort["max_period_tokens"],
        tail_tokens=cohort["budget"],
        minimum_occurrences=cohort.get("minimum_occurrences", TAIL_MINIMUM_OCCURRENCES),
    )
    measures_interrupt = bool(cohort.get("interrupt"))
    confirmed = unassessed = passes = unavailable = errors = length_stop = open_think = 0
    agent_confirmed = 0
    rule_fires: dict[str, int] = {name: 0 for name in (*AGENT_RULES, INTERRUPT, TAIL)}
    rule_tail_overlap: dict[str, int] = {name: 0 for name in AGENT_RULES}
    repeat_turns = interrupt_repeat_turns = assistant_turns = 0
    duplicate_fractions: list[float] = []
    interrupt_trials = 0
    rewards: list[float] = []
    lengths: list[int] = []
    turn_counts: list[int] = []
    wall_times: list[float] = []
    stop_reasons: dict[str, int] = {}
    for task_id, expected_hash in expected.items():
        row = indexed.get(task_id)
        if row is None:
            unassessed += 1
            unavailable += 1
            continue
        tokens = row.get("response_token_ids")
        prompt = row.get("prompt_token_ids")
        valid_tokens = isinstance(tokens, list) and bool(tokens) and all(type(token) is int for token in tokens)
        valid_prompt = expected_hash is None or (
            isinstance(prompt, list) and bool(prompt) and prompt_hash(prompt) == expected_hash
        )
        verification = row.get("verification") or {}
        status = verification.get("status")
        graded_agent_error = row.get("exception_type") in GRADED_AGENT_EXCEPTIONS and status == "verified"
        infrastructure_error = (
            bool(row.get("server_error"))
            or (bool(row.get("exception_type")) and not graded_agent_error)
            or (row.get("error_treatment") not in (None, "none") and not graded_agent_error)
        )
        damaged = (
            not valid_tokens
            or not valid_prompt
            or infrastructure_error
            or row.get("is_last_step", True) is not True
        )
        trial = audit_trial(row)
        if damaged:
            unassessed += 1
        else:
            reason = row.get("stop_reason") or "missing"
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
            rules = set(trial.rules)
            repeat_turns_trial = trial.repeat_turns
            turn_tokens = row.get("assistant_turn_token_ids")
            tail_sequences = turn_tokens if turn_tokens is not None else [tokens] if valid_tokens else []
            tail_hit = any(
                tail_loop(sequence, [1] * len(sequence), config) is not None for sequence in tail_sequences if sequence
            )
            # ``confirmed`` keeps Doom's tail semantics; agent loops count separately below.
            if tail_hit:
                confirmed += 1
                rule_fires[TAIL] += 1
            if rules:
                agent_confirmed += 1
            for name in AGENT_RULES:
                if name in rules:
                    rule_fires[name] += 1
                    if tail_hit:
                        rule_tail_overlap[name] += 1
            if INTERRUPT in rules:
                interrupt_trials += 1
            repeat_turns += repeat_turns_trial
            interrupt_repeat_turns += trial.interrupt_repeat_turns
            assistant_turns += trial.assistant_turns
            turn_counts.append(trial.assistant_turns)
            if trial.duplicate_fraction is not None:
                duplicate_fractions.append(trial.duplicate_fraction)
            score = _outcome_reward(verification.get("score", row.get("score")))
            if score is not None:
                rewards.append(score)
            if valid_tokens:
                lengths.append(sum(len(sequence) for sequence in turn_tokens) if turn_tokens is not None else len(tokens))
            wall = _wall_seconds(row)
            if wall is not None:
                wall_times.append(wall)
        row_error = status == "error" or infrastructure_error
        score = _outcome_reward(verification.get("score"))
        passed = score == 1.0 if score is not None else verification.get("passed")
        errors += int(row_error)
        unavailable += int(not row_error and (status != "verified" or type(passed) is not bool))
        passes += int((not damaged or graded_agent_error) and not row_error and status == "verified" and passed is True)
        at_cap = row.get("stop_reason") in ("length", "max_tokens", "max_length") or (
            valid_tokens and len(tokens) >= cohort["budget"]
        )
        length_stop += int(at_cap)
        response = (row.get("input_prompt") or "") + (row.get("output_response") or "")
        open_think += int(at_cap and response.rfind(THINK_OPEN) > response.rfind(THINK_CLOSE))
    rule_fires[INTERRUPT] = interrupt_trials
    quantiles = np.percentile(lengths, [50, 90, 99]).tolist() if lengths else [None] * 3
    duplicates = np.percentile(duplicate_fractions, [50, 90]).tolist() if duplicate_fractions else [None] * 2
    denominator = len(expected) - unassessed
    incidence = Incidence(
        len(expected), confirmed, unassessed, confirmed + unassessed, passes, unavailable, errors,
        length_stop, open_think, *quantiles,
    )
    agent = AgentIncidence(
        loop_trials=agent_confirmed,
        trial_share=agent_confirmed / denominator if denominator else 0.0,
        trial_share_ub=(agent_confirmed + unassessed) / len(expected) if expected else 0.0,
        repeat_turn_share=repeat_turns / assistant_turns if assistant_turns else 0.0,
        dup_tool_call_frac_median=duplicates[0],
        dup_tool_call_frac_p90=duplicates[1],
        interrupt_trial_share=(interrupt_trials / denominator if denominator else 0.0) if measures_interrupt else None,
        interrupt_repeat_share=(
            interrupt_repeat_turns / assistant_turns if assistant_turns else 0.0
        ) if measures_interrupt else None,
        rule_fires=rule_fires,
        rule_tail_overlap=rule_tail_overlap,
        mean_reward=float(np.mean(rewards)) if rewards else None,
        stop_reasons=stop_reasons,
        cost=TrialCost(
            float(np.median(turn_counts)) if turn_counts else None,
            float(np.median(lengths)) if lengths else None,
            float(np.median(wall_times)) if wall_times else None,
        ),
    )
    return incidence, agent


@dataclass(frozen=True)
class DomainAudit:
    incidence: Incidence
    agent: AgentIncidence
    cohort_hash: str
    target_status: str


def _cohort_digest(cohort: Mapping[str, Any]) -> str:
    """Hash the cohort's frozen task set: exact prompt digests when present, else task IDs."""
    return cohort_hash(
        row["prompt_hash"] if row.get("prompt_hash") is not None else prompt_hash([row["task_id"]])
        for row in cohort["rows"]
        for _ in range(cohort.get("trials_per_task", 1))
    )


def audit_domain(name: str, cohort: Mapping[str, Any], rows: Iterable[dict[str, Any]]) -> DomainAudit:
    """Audit a baseline cohort and apply the low-incidence rule to its target status."""
    incidence, agent = audit_cohort(cohort, rows)
    status = "target" if agent.loop_trials >= MINIMUM_TARGET_LOOPS else "non-target"
    return DomainAudit(incidence, agent, _cohort_digest(cohort), status)


def audit_candidate(
    cohort: Mapping[str, Any], rows: Iterable[dict[str, Any]], baseline: DomainAudit
) -> DomainAudit:
    """Keep the sealed baseline's task set and target decision for every candidate."""
    digest = _cohort_digest(cohort)
    if digest != baseline.cohort_hash:
        raise ValueError("candidate task IDs differ from the sealed baseline")
    incidence, agent = audit_cohort(cohort, rows)
    return DomainAudit(incidence, agent, digest, baseline.target_status)


def read_rows(pattern: str) -> Iterable[dict[str, Any]]:
    for path in sorted(StoragePath(pattern).glob(), key=str):
        if path.name == "aggregated_results.jsonl":
            continue
        with path.open("r") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def _agent_json(agent: AgentIncidence) -> dict[str, Any]:
    """Serialize the agent metrics for JSON output and baseline reload."""
    payload = asdict(agent)
    payload["cost"] = dict(asdict(agent.cost)) if agent.cost else None
    return payload


def _agent_from_json(payload: Mapping[str, Any]) -> AgentIncidence:
    cost = payload.get("cost")
    return AgentIncidence(
        **{**payload, "cost": TrialCost(**cost) if cost else None},
    )


def _markdown(document: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> str:
    columns = ("N", "confirmed", "unassessed", "UB", "passes", "unavailable", "errors", "length_stop", "open_think")
    agent_columns = (
        "trial_share", "trial_share_ub", "repeat_turn_share", "dup_frac_median", "dup_frac_p90",
        "interrupt_trial_share", "interrupt_repeat_share", "mean_reward",
    )
    lines = [
        "| Cohort | " + " | ".join(columns) + " | " + " | ".join(agent_columns) + " |",
        "| --- | " + " | ".join(["---:"] * (len(columns) + len(agent_columns))) + " |",
    ]
    for name, row in results.items():
        incidence, agent = row["incidence"], row["agent"]
        values = [str(incidence[key]) for key in columns]
        values += [
            _share(agent["trial_share"]), _share(agent["trial_share_ub"]), _share(agent["repeat_turn_share"]),
            _fraction(agent["dup_tool_call_frac_median"]), _fraction(agent["dup_tool_call_frac_p90"]),
            _share(agent["interrupt_trial_share"]), _share(agent["interrupt_repeat_share"]),
            _fraction(agent["mean_reward"]),
        ]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    fires = " ".join(
        f"{cohort}:{name}={count}"
        for cohort, row in results.items()
        for name, count in row["agent"]["rule_fires"].items()
        if count
    )
    return (
        "UB is the loop upper bound: confirmed + unassessed. An agent loop is any of "
        + ", ".join(AGENT_RULES)
        + ".\n\n"
        + "\n".join(lines)
        + "\n\nRule fires: "
        + (fires or "none")
        + "\n\nTarget status: "
        + ", ".join(f"{name}={status}" for name, status in document["target_status"].items())
        + ".\n"
    )


def _share(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _fraction(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", action="append", required=True, help="COHORT=JSONL_GLOB; one saved session per cohort")
    parser.add_argument("--baseline", help="Sealed baseline audit JSON; fixes each candidate's cohort and target status")
    parser.add_argument("--output", required=True, help="JSON output URI; Markdown is written beside it")
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest_payload = StoragePath(args.manifest).read_bytes()
    manifest = json.loads(manifest_payload)
    baseline = json.loads(StoragePath(args.baseline).read_text()) if args.baseline else None
    results: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    statuses: dict[str, str] = {}
    input_sources: dict[str, str] = {}
    for item in args.input:
        name, pattern = item.split("=", 1)
        if name in results:
            raise ValueError(f"duplicate input for {name}")
        input_sources[name] = pattern
        if baseline is None:
            audited = audit_domain(name, manifest["cohorts"][name], read_rows(pattern))
        else:
            reference = DomainAudit(
                Incidence(**baseline["domains"][name]),
                _agent_from_json(baseline["agents"][name]),
                baseline["cohort_hashes"][name],
                baseline["target_status"][name],
            )
            audited = audit_candidate(manifest["cohorts"][name], read_rows(pattern), reference)
        results[name] = {"incidence": asdict(audited.incidence), "agent": _agent_json(audited.agent)}
        hashes[name] = audited.cohort_hash
        statuses[name] = audited.target_status
    document = {
        "manifest": args.manifest,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "baseline": args.baseline,
        "inputs": input_sources,
        "cohort_hashes": hashes,
        "domains": {name: row["incidence"] for name, row in results.items()},
        "agents": {name: row["agent"] for name, row in results.items()},
        "target_status": statuses,
    }
    StoragePath(args.output).write_text(json.dumps(document, indent=2) + "\n")
    markdown = _markdown(document, results)
    StoragePath(args.output.removesuffix(".json") + ".md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
