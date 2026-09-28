# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Grade MT-MBPP generations against the validated tests and write pass@1 per checkpoint and language.

Reads each plan's completed generation artifacts (``evaluate_table9_accuracy.completed_task``), cuts every completion
at its closing code fence, assembles it with the task's test (``assemble.program``) and runs it in the sandbox.
Only documents whose test is valid (the reference passes, the stub fails) are scored; pass@1 is the fraction of those
that pass. Graded samples go to ``OUTPUT/<plan>/<checkpoint>/<task>.jsonl.gz`` keyed by the digest of the tests used,
so a rerun grades only new tasks or tasks whose tests changed; ``OUTPUT/summary.json`` collects the scores.

usage: uv run --offline --no-sync python -m experiments.domain_phase_mix.mt_mbpp_exec.grade_generations \
    --plan PLAN [--plan PLAN ...] --signatures SIGNATURES.jsonl.gz --translations TRANSLATIONS.jsonl \
    --validation VALIDATION.jsonl --output DIR --log LOG
"""

import argparse
import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from marin.evaluation.olmo_base_eval.components import MT_MBPP_SUBTASKS

from experiments.domain_phase_mix import evaluate_table9_accuracy as inference
from experiments.domain_phase_mix.mt_mbpp_exec.assemble import passed
from experiments.domain_phase_mix.mt_mbpp_exec.validate_tests import latest_translations, run, test_key

WORKERS = 12


def valid_tests(translations: Path, validation: Path) -> dict[tuple[str, int], dict]:
    """The latest translation of each task whose own validation passed."""
    latest = latest_translations(translations)
    verdicts = {}
    for row in map(json.loads, validation.read_text().splitlines()):
        verdicts[(row["language"], row["task_id"], row["test_sha256"])] = row["valid"]
    return {key: t["test"] for key, t in latest.items() if verdicts.get((key[0], key[1], test_key(t["test"])), False)}


def grade_task(
    plan: dict, label: str, row: dict, task: str, tests: dict, functions: dict, out: Path, log: Path
) -> dict | None:
    marker = inference.completed_task(plan, row, task, 0)
    if marker is None:
        return None
    language = task.removeprefix("mt_mbpp_")
    task_tests = {tid: t for (lang, tid), t in tests.items() if lang == language}
    digest = hashlib.sha256(
        json.dumps({str(k): test_key(v) for k, v in sorted(task_tests.items())}).encode()
    ).hexdigest()
    path = out / label / row["name"] / f"{task}.jsonl.gz"
    summary_path = path.with_suffix("").with_suffix(".summary.json")
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved["tests_sha256"] == digest and saved["generation_artifact"] == marker["artifact"]:
            return saved
    root = inference.result_root(plan, row, task, 0)
    samples = json.loads(gzip.decompress(inference.existing.read_bytes(root + "/samples.json.gz")))
    scored = [s for s in samples if s["metadata"]["id"] in task_tests]

    def grade(sample: dict) -> dict:
        tid = sample["metadata"]["id"]
        code = sample["generation"].split("```")[0]
        result = run(language, code, task_tests[tid], functions[(language, tid)])
        return {
            "doc_id": sample["doc_id"],
            "task_id": tid,
            "passed": passed(language, result),
            "phase": result["phase"],
            "exit_code": result["exit_code"],
            "timeout": result["timeout"],
            "stderr_tail": result["stderr_tail"][-400:],
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        graded = list(pool.map(grade, scored))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for g in graded:
            handle.write(json.dumps(g, sort_keys=True) + "\n")
    summary = {
        "checkpoint": row["name"],
        "plan": label,
        "region": plan["region"],
        "task": task,
        "valid_documents": len(scored),
        "documents": len(samples),
        "passed": sum(g["passed"] for g in graded),
        "pass@1": sum(g["passed"] for g in graded) / max(len(scored), 1),
        "tests_sha256": digest,
        "generation_artifact": marker["artifact"],
        "generation_root": root,
    }
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    with log.open("a") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} graded {row['name']} {task}: {summary['passed']}/{len(scored)}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    functions = {
        (r["language"], r["task_id"]): r.get("function") or r["mbpp_function"]
        for r in map(json.loads, gzip.decompress(args.signatures.read_bytes()).decode().splitlines())
        if r["split"] == "test"
    }
    tests = valid_tests(args.translations, args.validation)
    summaries = []
    for plan_path in args.plan:
        plan = json.loads(plan_path.read_text())
        for row in plan["rows"]:
            for task in MT_MBPP_SUBTASKS:
                result = grade_task(plan, plan_path.stem, row, task, tests, functions, args.output, args.log)
                if result is not None:
                    summaries.append(result)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=1) + "\n")
    total = sum(len(json.loads(p.read_text())["rows"]) for p in args.plan) * len(MT_MBPP_SUBTASKS)
    print(f"{len(summaries)} of {total} tasks graded")


if __name__ == "__main__":
    main()
