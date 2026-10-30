"""Regenerate tests/graders/data/humaneval_grader_benchmark.jsonl.

The fixture records the verdict the pinned lm-evaluation-harness produced for
each candidate, because re-running the reference needs ``HF_ALLOW_CODE_EVAL=1``
and a Hub download of the ``code_eval`` metric module -- neither of which
belongs in the default test path. This script is how those verdicts are
reproduced: pinned dataset revision in, pinned reference out.

Both inputs are pinned -- the dataset revision and the code_eval Space revision --
so a rerun grades the same candidates against the same reference. The only
non-reproducible case would be a candidate whose runtime straddles the grader's
timeout; the non-terminating bodies below are far past it, not near it.

Usage:
    HF_ALLOW_CODE_EVAL=1 uv run python scripts/benchmarks/build_humaneval_benchmark.py

Guarded by __main__: multiprocessing may use the spawn start method, in which
case children re-import this file and any module-level work would re-run per
candidate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BENCHMARK = REPO / "tests/graders/data/humaneval_grader_benchmark.jsonl"

DATASET = "openai/openai_humaneval"
DATASET_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"

# code_eval is a Space on the Hub, so an unpinned load resolves to whatever is
# published at the time. Pin it: the recorded verdicts are only reproducible
# against the same reference implementation that produced them.
CODE_EVAL = "code_eval"
CODE_EVAL_REVISION = "262b7e74cf29a715d74f8b02ba1d6ef74e432333"

FIELDS = ["kind", "task_id", "problem", "solution", "reference_answer", "reference_grade"]

# Roughly the shape of real completions: mostly terminating, with a small tail of
# non-terminating ones. Timeouts are capped at 5% because each costs a full
# irreducible SIGALRM wait in the reference and the port alike.
WRONG_BODY = "    return None\n"
SYNTAX_ERROR_BODY = "    return (\n"
NON_TERMINATING_BODY = "    while True:\n        pass\n"


def build_records(rows, count):
    records = []
    for index, row in enumerate(rows[:count]):
        bucket = index % 20
        if bucket < 12:
            kind, solution = "correct", row["canonical_solution"]
        elif bucket < 17:
            kind, solution = "wrong", WRONG_BODY
        elif bucket < 19:
            kind, solution = "syntax_error", SYNTAX_ERROR_BODY
        else:
            kind, solution = "timeout", NON_TERMINATING_BODY
        records.append(
            {
                "kind": kind,
                "task_id": row["task_id"],
                "problem": row["prompt"],
                "solution": solution,
                # The task's doc_to_target: "{{test}}\ncheck({{entry_point}})".
                "reference_answer": f"{row['test']}\ncheck({row['entry_point']})",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="problems to record")
    args = parser.parse_args()

    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise SystemExit("code_eval executes model-generated code; set HF_ALLOW_CODE_EVAL=1 to proceed")

    # Imported here rather than at module level: under the spawn start method each
    # child re-imports this file, and these are expensive enough to dominate.
    import datasets
    import evaluate

    code_eval = evaluate.load(CODE_EVAL, revision=CODE_EVAL_REVISION)
    rows = list(datasets.load_dataset(DATASET, split="test", revision=DATASET_REVISION))
    records = build_records(rows, args.count)

    for record in records:
        candidate = record["problem"] + record["solution"]
        result = code_eval.compute(
            references=[record["reference_answer"]],
            predictions=[[candidate]],
            k=[1],
        )
        record["reference_grade"] = float(result[0]["pass@1"])

    with BENCHMARK.open("w") as f:
        for record in records:
            f.write(json.dumps({field: record[field] for field in FIELDS}) + "\n")

    passed = sum(r["reference_grade"] for r in records)
    print(f"wrote {len(records)} records to {BENCHMARK.relative_to(REPO)} (reference score {passed:.0f})")


if __name__ == "__main__":
    main()
