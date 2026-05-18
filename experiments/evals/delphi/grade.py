# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade rollouts produced by ``run_rollouts``.

GSM8K grading: exact replication of Will's setup
(``origin/will/delphi-evals:experiments/exp1337_eval_suite.py``) — instantiate
the lm-eval-harness ``gsm8k`` task, apply its filter chain (``strict-match``
and ``flexible-extract``), and score with its ``exact_match`` metric.

MATH-500 grading: ``extract_boxed`` + ``safe_grade`` from marin's existing
math grading utilities (``marin.rl.environments.tinker_environments``).

Parallelism / resume: each rollout record is fanned out per completion via
``flat_map``, sharded across ``num_workers`` Zephyr workers, graded per item,
and written to per-shard JSONL files under ``{output_path}/shards/``.
Zephyr's ``write_jsonl(skip_existing=True)`` short-circuits already-written
shards on retry. The orchestrator then follows the same
init-dict / iterate-files / append / validate pattern as
``rollout.py:369-384``: reads metadata once from ``rollouts.jsonl.gz``,
groups per-completion graded items by ``problem_id``, validates counts, sorts
each group by ``completion_idx``, and writes per-problem records to
``{output_path}/graded.jsonl.gz`` — same path and per-record schema as
before, so downstream consumers don't change.
"""

import json
import logging
import os
from dataclasses import dataclass

import fsspec
from marin.execution.executor import InputName
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)


# Registered grader names — dispatched in ``grade_rollouts``.
GSM8K_LM_EVAL = "gsm8k_lm_eval"
MATH_BOXED = "math_boxed"


@dataclass(frozen=True)
class GradingConfig:
    rollout_path: str | InputName
    output_path: str
    grader: str  # one of GSM8K_LM_EVAL, MATH_BOXED
    num_workers: int = 32


# === Per-completion fan-out ===


def _flatten(rec):
    pid = rec["problem_id"]
    ground_truth = rec["ground_truth"]
    problem = rec["problem"]
    for i, completion in enumerate(rec["completions"]):
        yield {
            "problem_id": pid,
            "completion_idx": i,
            "completion": completion,
            "ground_truth": ground_truth,
            "problem": problem,
        }


# === Per-shard graders ===
#
# Each is a generator: builds any once-per-worker state at the top, then yields
# per-completion graded dicts. Imports happen inside so pickling these across
# Zephyr workers doesn't require deps in the coordinator process.


def _grade_gsm8k_shard(items, shard_info):
    import lm_eval.tasks
    from lm_eval.api.instance import Instance

    task = lm_eval.tasks.get_task_dict(["gsm8k"])["gsm8k"]
    filter_names = [f.name for f in task._filters]

    for item in items:
        doc = {"question": item["problem"], "answer": f"#### {item['ground_truth']}"}
        inst = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=("", {}),
            idx=item["completion_idx"],
            task_name="gsm8k",
        )
        inst.resps = [item["completion"]]
        task._instances = [inst]
        task.apply_filters()

        out: dict = {"problem_id": item["problem_id"], "completion_idx": item["completion_idx"]}
        for name in filter_names:
            key = name.replace("-", "_")
            filtered = inst.filtered_resps[name]
            out[f"extraction_{key}"] = filtered
            out[f"correct_{key}"] = bool(task.process_results(doc, [filtered])["exact_match"])
        yield out


def _grade_math_shard(items, shard_info):
    from marin.rl.environments.tinker_environments.math_env import safe_grade
    from marin.rl.environments.tinker_environments.math_grading import extract_boxed

    for item in items:
        try:
            ex = extract_boxed(item["completion"])
        except ValueError:
            ex = None
        if ex is None:
            correct = False
        else:
            correct = safe_grade(ex, item["ground_truth"], grader="sympy", timeout=1.0)
        yield {
            "problem_id": item["problem_id"],
            "completion_idx": item["completion_idx"],
            "extraction": ex,
            "correct": correct,
        }


# === Orchestration ===


def grade_rollouts(config: GradingConfig) -> None:
    in_path = os.path.join(config.rollout_path, "rollouts.jsonl.gz")

    # Pre-read metadata. Mirrors rollout.py:355-360 loading the dataset:
    # we need problem_ids for the dict-init and n_samples for the count check.
    # Completions are dropped (held only by Zephyr's streaming read below).
    metadata: list[dict] = []
    with fsspec.open(in_path, "rt", compression="gzip") as f:
        for line in f:
            rec = json.loads(line)
            metadata.append(
                {
                    "problem_id": rec["problem_id"],
                    "ground_truth": rec["ground_truth"],
                    "n_samples": len(rec["completions"]),
                }
            )

    if config.grader == GSM8K_LM_EVAL:
        shard_fn = _grade_gsm8k_shard
    elif config.grader == MATH_BOXED:
        shard_fn = _grade_math_shard
    else:
        raise ValueError(f"Unknown grader: {config.grader!r}")

    shards_pattern = os.path.join(config.output_path, "shards", "graded-shard-{shard:04d}.jsonl.gz")
    pipeline = (
        Dataset.from_files(in_path)
        .load_jsonl()
        .flat_map(_flatten)
        .reshard(config.num_workers)
        .map_shard(shard_fn)
        .write_jsonl(shards_pattern, skip_existing=True)
    )
    ZephyrContext(name=f"grade-{config.grader}", max_workers=config.num_workers).execute(pipeline)

    # Group per-completion graded items by problem_id, validate counts.
    # Same shape as rollout.py:369-384 (init-by-pid / iterate-files / append / validate).
    items_by_pid: dict = {m["problem_id"]: [] for m in metadata}
    for shard_idx in range(config.num_workers):
        shard_path = os.path.join(config.output_path, "shards", f"graded-shard-{shard_idx:04d}.jsonl.gz")
        with fsspec.open(shard_path, "rt", compression="gzip") as f:
            for line in f:
                rec = json.loads(line)
                items_by_pid[rec["problem_id"]].append(rec)

    for m in metadata:
        pid = m["problem_id"]
        if len(items_by_pid[pid]) != m["n_samples"]:
            raise RuntimeError(
                f"Problem {pid} aggregated {len(items_by_pid[pid])} graded items, expected {m['n_samples']}"
            )

    # Build per-problem records, sorted by completion_idx, in the existing schema.
    final_path = os.path.join(config.output_path, "graded.jsonl.gz")
    with fsspec.open(final_path, "wt", compression="gzip") as out:
        for m in metadata:
            pid = m["problem_id"]
            items = sorted(items_by_pid[pid], key=lambda x: x["completion_idx"])
            graded: dict = {"problem_id": pid, "ground_truth": m["ground_truth"]}
            if config.grader == GSM8K_LM_EVAL:
                for key in ("strict_match", "flexible_extract"):
                    graded[f"extractions_{key}"] = [it[f"extraction_{key}"] for it in items]
                    graded[f"correct_{key}"] = [it[f"correct_{key}"] for it in items]
            else:  # MATH_BOXED
                graded["extractions"] = [it["extraction"] for it in items]
                graded["correct"] = [it["correct"] for it in items]
            out.write(json.dumps(graded) + "\n")
    logger.info(f"Wrote {final_path} ({len(metadata)} graded records)")
