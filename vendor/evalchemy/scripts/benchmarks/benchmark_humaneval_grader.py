"""Time the self-contained HumanEval grader in eval/graders on the 100-problem benchmark.

Reports single-CPU wall time for ``tests/graders/data/humaneval_grader_benchmark.jsonl``
and checks every verdict against the pinned lm-evaluation-harness ``v0.4.12``
grade recorded in that fixture.

Timings are broken out by candidate kind because the aggregate is misleading:
a non-terminating candidate costs a full SIGALRM wait, identical in the
reference and here, so it dilutes any per-candidate gain. The terminating line
is the one that reflects the work this grader actually removes.

Usage:
    uv run python scripts/benchmarks/benchmark_humaneval_grader.py
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import humaneval  # noqa: E402

BENCHMARK = REPO / "tests/graders/data/humaneval_grader_benchmark.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="grade only the first N problems")
    args = parser.parse_args()

    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} problems from {BENCHMARK.name}\n")

    grades, by_kind = [], collections.defaultdict(list)
    start = time.perf_counter()
    for record in records:
        candidate_start = time.perf_counter()
        grades.append(humaneval.grade(record["problem"], record["solution"], record["reference_answer"]))
        by_kind[record["kind"]].append(time.perf_counter() - candidate_start)
    elapsed = time.perf_counter() - start

    expected = [r["reference_grade"] for r in records]
    disagreements = [i for i, (got, want) in enumerate(zip(grades, expected, strict=True)) if got != want]

    print(f"  score        {sum(grades):.0f}/{len(records)} (harness: {sum(expected):.0f})")
    print(f"  agreement    {len(records) - len(disagreements)}/{len(records)} with lm-eval-harness")
    print(f"  total        {elapsed:7.2f} s  ({elapsed / len(records) * 1e3:7.1f} ms/problem)\n")

    for kind, times in sorted(by_kind.items()):
        print(f"    {kind:13} {len(times):3d} x {sum(times) / len(times) * 1e3:8.1f} ms  = {sum(times):6.2f} s")
    terminating = [t for kind, times in by_kind.items() if kind != "timeout" for t in times]
    if terminating:
        mean = sum(terminating) / len(terminating)
        print(f"    {'terminating':13} {len(terminating):3d} x {mean * 1e3:8.1f} ms  = {sum(terminating):6.2f} s")

    for index in disagreements:
        record = records[index]
        print(f"  DISAGREE #{index} kind={record['kind']} task_id={record['task_id']}")


if __name__ == "__main__":
    main()
