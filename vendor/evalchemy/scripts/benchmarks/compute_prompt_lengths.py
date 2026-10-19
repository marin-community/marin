#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Measure every benchmark's longest prompt and store it in prompt_lengths.json.

Each benchmark's prompts are rendered by driving its own ``generate_responses``
with a prompt-capturing LM stand-in (``eval/contracts/prompt_corpus.py``), so the
measurement reflects the benchmark's real prompt shape, chat template, and
few-shot configuration. Measuring needs the benchmark's dataset and its extra
installed, which is why this runs as a repo tool rather than at eval time.

Usage:
  uv run python scripts/benchmarks/compute_prompt_lengths.py                 # every benchmark
  uv run python scripts/benchmarks/compute_prompt_lengths.py AIME24 MATH500  # refresh a subset
  uv run python scripts/benchmarks/compute_prompt_lengths.py --check         # report drift, write nothing

``--check`` exits nonzero when a stored value no longer matches what the
benchmark renders, and is how the nightly job catches stale metadata.
"""

import argparse
import contextlib
import json
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.contracts.prompt_corpus import load_reference_tokenizer, render_prompt_corpus  # noqa: E402
from eval.contracts.prompt_length import (  # noqa: E402
    PROMPT_LENGTHS_PATH,
    REFERENCE_TOKENIZER,
    BenchmarkPromptLength,
    PromptLengths,
    load_prompt_lengths,
)

CHAT_BENCHMARKS_DIR = REPO_ROOT / "eval" / "chat_benchmarks"

# A benchmark that downloads a multi-gigabyte dataset, or waits on something that
# never arrives, must not stall the whole sweep: it is recorded as unmeasured with
# the timeout as its reason.
DEFAULT_TIMEOUT = 900


def registered_benchmarks() -> list[str]:
    """Return every benchmark dir the driver would load."""
    return sorted(path.parent.name for path in CHAT_BENCHMARKS_DIR.glob("*/eval_instruct.py"))


@contextlib.contextmanager
def time_limit(seconds: int):
    """Interrupt the enclosed block once it has run for ``seconds``."""

    def expire(_signum, _frame):
        raise TimeoutError(f"rendering exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def measure(task_names: list[str], timeout: int) -> tuple[dict[str, BenchmarkPromptLength], dict[str, str]]:
    """Render each benchmark's prompt corpus, collecting failures as reasons."""
    tokenizer = load_reference_tokenizer()
    measured: dict[str, BenchmarkPromptLength] = {}
    unmeasured: dict[str, str] = {}
    for task_name in task_names:
        try:
            with time_limit(timeout):
                measured[task_name] = render_prompt_corpus(task_name, tokenizer)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}".replace(f"{REPO_ROOT}/", "")
            unmeasured[task_name] = reason
            print(f"{task_name}: unmeasured ({reason})", flush=True)
            continue
        corpus = measured[task_name]
        print(
            f"{task_name}: {corpus.distinct_prompt_count} prompts, "
            f"longest {corpus.longest_prompt_tokens} tokens / {corpus.longest_prompt_chars} chars",
            flush=True,
        )
    return measured, unmeasured


def merge(
    stored: PromptLengths,
    measured: dict[str, BenchmarkPromptLength],
    unmeasured: dict[str, str],
) -> PromptLengths:
    """Return the stored metadata with the refreshed benchmarks replaced.

    A benchmark that stopped rendering loses its stored length, which a transient
    download failure can also cause, so each demotion is called out before the
    file is written.
    """
    refreshed = set(measured) | set(unmeasured)
    benchmarks = {name: entry for name, entry in stored.benchmarks.items() if name not in refreshed}
    reasons = {name: reason for name, reason in stored.unmeasured.items() if name not in refreshed}
    for name, reason in unmeasured.items():
        if name in stored.benchmarks:
            print(f"{name}: dropping its stored prompt length ({reason})", file=sys.stderr)
    benchmarks.update(measured)
    reasons.update(unmeasured)
    return PromptLengths(REFERENCE_TOKENIZER, stored.prompt_margin_tokens, benchmarks, reasons)


def drift(stored: PromptLengths, measured: dict[str, BenchmarkPromptLength], unmeasured: dict[str, str]) -> list[str]:
    """Return one message per benchmark whose stored value no longer holds."""
    problems = []
    for name, rendered in measured.items():
        entry = stored.benchmarks.get(name)
        if entry is None:
            problems.append(f"{name}: newly measurable but stored as unmeasured")
        elif entry != rendered:
            problems.append(f"{name}: stored {entry.to_dict()} but rendered {rendered.to_dict()}")
    for name, reason in unmeasured.items():
        if name in stored.benchmarks:
            problems.append(f"{name}: stored a prompt length but no longer renders ({reason})")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", help="benchmark dirs to refresh (default: all)")
    parser.add_argument("--check", action="store_true", help="report drift and write nothing")
    parser.add_argument("--path", type=Path, default=PROMPT_LENGTHS_PATH, help="metadata file to read and write")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"seconds to spend rendering one benchmark before giving up (default {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    stored = load_prompt_lengths(args.path)
    task_names = args.tasks or registered_benchmarks()
    measured, unmeasured = measure(task_names, args.timeout)

    if args.check:
        problems = drift(stored, measured, unmeasured)
        for problem in problems:
            print(f"stale prompt length: {problem}", file=sys.stderr)
        if problems:
            print(f"\nRefresh with: uv run python {Path(__file__).relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        print(f"prompt lengths OK: {len(measured)} measured, {len(unmeasured)} unmeasured")
        return 0

    args.path.write_text(json.dumps(merge(stored, measured, unmeasured).to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
