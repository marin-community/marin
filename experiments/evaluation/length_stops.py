# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the scored generation attempts of an evaluation result root that stopped at their output token limit."""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from finestore.reader import ReadView
from marin.rollouts.evaluation import RUN_OBJECT
from marin.rollouts.evaluation_results import SCORED_OUTCOMES, terminal_attempts
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rolloutengine.models import LENGTH_STOP_REASON
from taskcompendium.evaluation.spec import EvaluationProtocol, OperationKind

# Evidence blobs repeat the prompt, so an attempt with a long-context prompt takes megabytes.
BLOB_BATCH = 64
GENERATION = (OperationKind.CHAT_GENERATION, OperationKind.TEXT_GENERATION)

type Population = tuple[str, tuple[str, str] | None]
"""Benchmark label, and a summary group name and value or ``None`` for the whole benchmark."""


@dataclass(frozen=True)
class ScoredAttempt:
    reward: float
    stop_reason: str | None
    response_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class LengthStops:
    """Length stops among the scored attempts of a benchmark, or of one of its summary group labels.

    Length-stop and response-token fields use the names of SkyRL's per-data-source evaluation metrics. Every mean and
    fraction divides by ``scored``.
    """

    benchmark: str
    group: tuple[str, str] | None
    attempts: int
    """Latest finished attempts, one per source row and trial."""
    scored: int
    """Attempts graded or failed at submission whose grade has a reward."""
    mean_reward: float | None
    length_stop_count: int
    length_stop_fraction: float | None
    """``None`` when no attempt is scored or a scored attempt has no stop reason; so is the score contribution."""
    length_stop_score_contribution: float | None
    response_tokens_mean: float | None
    response_tokens_max: int | None
    max_tokens: tuple[int, int] | None
    """Smallest and largest output token limit of the scored attempts."""


def _scored_attempt(outcome: str | None, raw: dict) -> ScoredAttempt | None:
    reward = raw["grade"]["reward"]
    if outcome not in SCORED_OUTCOMES or reward is None:
        return None
    turns = [step["turn"] for conversation in raw["conversations"] for step in conversation["steps"]]
    return ScoredAttempt(
        reward=reward,
        stop_reason=raw["stop_reason"] or None,
        response_tokens=sum(
            turn["usage"]["completion_tokens"] if turn["usage"] is not None else len(turn["response_token_ids"])
            for turn in turns
        ),
        max_tokens=raw["options"]["max_tokens"],
    )


def _summary(population: Population, attempts: int, scored: Sequence[ScoredAttempt]) -> LengthStops:
    count = len(scored)
    stops = [attempt for attempt in scored if attempt.stop_reason == LENGTH_STOP_REASON]
    known = count > 0 and all(attempt.stop_reason is not None for attempt in scored)
    limits = [attempt.max_tokens for attempt in scored]
    return LengthStops(
        benchmark=population[0],
        group=population[1],
        attempts=attempts,
        scored=count,
        mean_reward=sum(attempt.reward for attempt in scored) / count if count else None,
        length_stop_count=len(stops),
        length_stop_fraction=len(stops) / count if known else None,
        length_stop_score_contribution=sum(attempt.reward for attempt in stops) / count if known else None,
        response_tokens_mean=sum(attempt.response_tokens for attempt in scored) / count if count else None,
        response_tokens_max=max((attempt.response_tokens for attempt in scored), default=None),
        max_tokens=(min(limits), max(limits)) if limits else None,
    )


def _labels(protocols: dict[str, EvaluationProtocol]) -> dict[str, str]:
    """Name each corpus by its benchmark, adding the version, or else the dataset digest, when a benchmark repeats."""
    benchmarks = Counter(protocol.benchmark for protocol in protocols.values())
    versions = Counter((protocol.benchmark, protocol.version) for protocol in protocols.values())
    labels = {}
    for dataset, protocol in protocols.items():
        if benchmarks[protocol.benchmark] == 1:
            labels[dataset] = protocol.benchmark
        elif versions[(protocol.benchmark, protocol.version)] == 1:
            labels[dataset] = f"{protocol.benchmark}@{protocol.version}"
        else:
            labels[dataset] = f"{protocol.benchmark} ({dataset[:12]})"
    return labels


def length_stops(root: str) -> list[LengthStops]:
    """Summarize each generation corpus's terminal attempts, then those of each summary group label."""
    reader = ReadView(root)
    saved = reader.read_blob(RUN_OBJECT)
    if saved is None:
        raise ValueError(f"No evaluation run is stored in {root}")
    protocols = {
        entry["dataset"]: protocol
        for entry in json.loads(saved)["datasets"]
        if (protocol := EvaluationProtocol.model_validate(entry["protocol"])).operation in GENERATION
    }
    labels = _labels(protocols)
    attempts: Counter[Population] = Counter({(label, None): 0 for label in labels.values()})
    scored: defaultdict[Population, list[ScoredAttempt]] = defaultdict(list)
    terminal = [attempt for attempt in terminal_attempts(reader).values() if attempt["dataset"] in labels]
    for batch in itertools.batched(terminal, BLOB_BATCH):
        blobs = [attempt["result_blob"] for attempt in batch]
        evidence = reader.read_blobs(blobs)
        for attempt, blob in zip(batch, blobs, strict=True):
            raw = json.loads(evidence[blob])
            label = labels[attempt["dataset"]]
            result = _scored_attempt(attempt["outcome"], raw)
            for population in ((label, None), *((label, group) for group in raw["item"]["groups"].items())):
                attempts[population] += 1
                if result is not None:
                    scored[population].append(result)
    order = sorted(attempts, key=lambda population: (population[0], population[1] is not None, population[1] or ()))
    return [_summary(population, attempts[population], scored[population]) for population in order]


def summary_table(summaries: Sequence[LengthStops]) -> str:
    def number(value: float | None, digits: int) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    rows = [
        (
            "benchmark",
            "group",
            "attempts",
            "scored",
            "mean_reward",
            "length_stop_count",
            "length_stop_fraction",
            "length_stop_score_contribution",
            "response_tokens_mean",
            "response_tokens_max",
            "max_tokens",
        )
    ]
    for summary in summaries:
        limits = summary.max_tokens
        rows.append(
            (
                summary.benchmark,
                "-" if summary.group is None else "=".join(summary.group),
                str(summary.attempts),
                str(summary.scored),
                number(summary.mean_reward, 4),
                str(summary.length_stop_count),
                number(summary.length_stop_fraction, 4),
                number(summary.length_stop_score_contribution, 4),
                number(summary.response_tokens_mean, 1),
                "-" if summary.response_tokens_max is None else str(summary.response_tokens_max),
                "-" if limits is None else str(limits[0]) if limits[0] == limits[1] else f"{limits[0]}-{limits[1]}",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip() for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Result root written by experiments.evaluation.task_driver.")
    arguments = parser.parse_args()
    configure_coreweave_s3()
    print(summary_table(length_stops(arguments.root)))


if __name__ == "__main__":
    main()
