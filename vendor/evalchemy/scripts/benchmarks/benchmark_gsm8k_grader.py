"""Time the self-contained GSM8K grader in eval/graders on the 100-problem benchmark.

Reports single-CPU wall time for ``tests/graders/data/gsm8k_grader_benchmark.jsonl``
under both of the task's filters, and checks every verdict against the pinned
lm-evaluation-harness ``v0.4.12`` grade recorded in that fixture.

The fixture holds 100 GSM8K test problems, each with a synthesized completion in
the format the 5-shot prompt elicits -- reasoning closing with a
``#### <number>`` line -- mixing verbatim, equivalently reformatted, wrong, and
malformed answers.

Unlike the MATH graders this one memoizes nothing, so there is no cold/warm
split to report.

Usage:
    uv run python scripts/benchmarks/benchmark_gsm8k_grader.py
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import gsm8k  # noqa: E402

BENCHMARK = REPO / "tests/graders/data/gsm8k_grader_benchmark.jsonl"

GRADERS = [
    ("strict-match", gsm8k.grade, "strict_match_reference_grade"),
    ("flexible-extract", gsm8k.grade_flexible_extract, "flexible_extract_reference_grade"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timed runs")
    args = parser.parse_args()

    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    print(f"{len(records)} problems from {BENCHMARK.name}\n")

    for name, grade, reference_field in GRADERS:
        grades = [grade(r["problem"], r["solution"], r["reference_answer"]) for r in records]

        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for record in records:
                grade(record["problem"], record["solution"], record["reference_answer"])
            runs.append(time.perf_counter() - start)
        elapsed = statistics.median(runs)

        expected = [r[reference_field] for r in records]
        disagreements = [i for i, (got, want) in enumerate(zip(grades, expected, strict=True)) if got != want]

        print(f"{name}")
        print(f"  score          {sum(grades):.0f}/{len(records)} (harness: {sum(expected):.0f})")
        print(f"  agreement      {len(records) - len(disagreements)}/{len(records)} with lm-eval-harness")
        print(f"  median         {elapsed * 1e3:7.3f} ms  ({elapsed / len(records) * 1e6:6.2f} us/problem)")
        for index in disagreements:
            record = records[index]
            print(f"  DISAGREE #{index} kind={record['kind']} reference_answer={record['reference_answer']!r}")
        print()


if __name__ == "__main__":
    main()
