# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize taskforge JSONL ledgers into per-kind and per-step wall time and token totals.

``busy`` sums span durations (concurrent spans add up); ``elapsed`` is first start to last end.

    uv run lib/taskforge/scripts/ledger_summary.py <ledger dir or .jsonl files>... [--format json]
"""

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from taskforge.ledger.jsonl import ledger_files, read_entries
from taskforge.ledger.records import LedgerEntry


@dataclass
class Totals:
    count: int = 0
    failed: int = 0
    busy: float = 0.0
    first_start: float = float("inf")
    last_end: float = float("-inf")
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0

    @property
    def elapsed(self) -> float:
        return self.last_end - self.first_start

    def add(self, entry: LedgerEntry) -> None:
        self.count += 1
        self.failed += entry.cause is not None
        self.busy += entry.wall
        self.first_start = min(self.first_start, entry.started)
        self.last_end = max(self.last_end, entry.ended)
        self.tokens_in += entry.tokens_in or 0
        self.tokens_out += entry.tokens_out or 0
        self.tokens_reasoning += entry.tokens_reasoning or 0


@dataclass(frozen=True)
class Summary:
    by_kind: dict[str, Totals]
    by_step: dict[str, Totals]


def summarize(entries: Iterable[LedgerEntry]) -> Summary:
    by_kind: dict[str, Totals] = {}
    by_step: dict[str, Totals] = {}
    for entry in entries:
        by_kind.setdefault(entry.kind.value, Totals()).add(entry)
        by_step.setdefault(f"{entry.kind.value}/{entry.step}", Totals()).add(entry)
    return Summary(by_kind=dict(sorted(by_kind.items())), by_step=dict(sorted(by_step.items())))


def ledger_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for p in inputs:
        paths.extend(ledger_files(p) if p.is_dir() else [p])
    return paths


def summary_json(summary: Summary) -> dict:
    def group(totals: dict[str, Totals]) -> dict:
        return {key: {**asdict(t), "elapsed": t.elapsed} for key, t in totals.items()}

    return {"by_kind": group(summary.by_kind), "by_step": group(summary.by_step)}


def summary_text(summary: Summary) -> str:
    header = (
        f"{'group':<40} {'count':>6} {'failed':>6} {'busy_s':>10} {'elapsed_s':>10} "
        f"{'tok_in':>12} {'tok_out':>12} {'tok_reason':>12}"
    )
    lines: list[str] = []
    for title, totals in (("per kind", summary.by_kind), ("per kind/step", summary.by_step)):
        lines += [title, header]
        for key, t in totals.items():
            lines.append(
                f"{key:<40} {t.count:>6} {t.failed:>6} {t.busy:>10.1f} {t.elapsed:>10.1f} "
                f"{t.tokens_in:>12} {t.tokens_out:>12} {t.tokens_reasoning:>12}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="ledger directories or .jsonl files")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    summary = summarize(entry for path in ledger_paths(args.inputs) for entry in read_entries(path))
    print(json.dumps(summary_json(summary), indent=2) if args.format == "json" else summary_text(summary))


if __name__ == "__main__":
    main(sys.argv[1:])
