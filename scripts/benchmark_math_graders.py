# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the self-contained MATH graders on the 100-problem benchmark.

Reports single-CPU wall time for ``tests/evaluation/data/math_grader_benchmark.jsonl``
and checks every verdict against the lm-evaluation-harness ``d5e3391`` grade
recorded in that fixture.

The fixture holds 100 MATH test problems spanning all seven subjects, each with
a synthesized completion in the format its task's prompt elicits: a chain of
thought closing with the "Final Answer: ..." template for ``minerva_math``, a
short answer for ``hendrycks_math``. Completions mix verbatim, equivalently
reformatted, wrong, and malformed answers.

Usage:
    uv run --package marin-core python scripts/benchmark_math_graders.py
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from marin.evaluation.graders import hendrycks_math, minerva_math

BENCHMARK = Path(__file__).resolve().parents[1] / "tests/evaluation/data/math_grader_benchmark.jsonl"

GRADERS = [
    ("hendrycks_math", hendrycks_math, "hendrycks_solution", "hendrycks_reference_grade"),
    ("minerva_math", minerva_math, "minerva_solution", "minerva_reference_grade"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timed runs after the cold run")
    args = parser.parse_args()

    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    print(f"{len(records)} problems from {BENCHMARK.name}\n")

    # Importing sympy and building its ANTLR parser tables is a one-time cost
    # the reference pays too, so charge it before timing any grading.
    start = time.perf_counter()
    minerva_math.is_equiv("\\sqrt{2}", "\\sqrt{2}")
    print(f"one-time sympy + ANTLR init: {(time.perf_counter() - start) * 1e3:.1f} ms\n")

    for name, grader, solution_field, reference_field in GRADERS:
        minerva_math._sympy_parses.cache_clear()

        start = time.perf_counter()
        grades = [grader.grade(r["problem"], r[solution_field], r["reference_answer"]) for r in records]
        cold = time.perf_counter() - start

        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for record in records:
                grader.grade(record["problem"], record[solution_field], record["reference_answer"])
            runs.append(time.perf_counter() - start)

        expected = [r[reference_field] for r in records]
        disagreements = [i for i, (got, want) in enumerate(zip(grades, expected, strict=True)) if got != want]

        print(f"{name}")
        print(f"  score          {sum(grades):.0f}/{len(records)} (harness: {sum(expected):.0f})")
        print(f"  agreement      {len(records) - len(disagreements)}/{len(records)} with lm-eval-harness")
        print(f"  cold cache     {cold * 1e3:7.2f} ms  ({cold / len(records) * 1e3:6.3f} ms/problem)")
        print(
            f"  warm (median)  {statistics.median(runs) * 1e3:7.2f} ms  "
            f"({statistics.median(runs) / len(records) * 1e3:6.3f} ms/problem)"
        )
        for index in disagreements:
            record = records[index]
            print(f"  DISAGREE #{index} kind={record['kind']} reference_answer={record['reference_answer']!r}")
        print()


if __name__ == "__main__":
    main()
