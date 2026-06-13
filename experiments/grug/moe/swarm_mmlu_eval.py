# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run MMLU (0-shot and 5-shot) on every finished swarm candidate's checkpoint.

For each of the ~485 finished ``swarm_fisher_dsp_d512_NNNNNN`` candidates,
locate its successful output directory on GCS (via the dashboard state cache)
and emit one ``ExecutorStep`` per (candidate, MMLU task). Submits via
``executor_main`` so already-evaluated cells stay cached on re-run.

Resources: v4-8 (preemptible) in ``us-central2-b`` to stay alongside the
training swarm without contending for the reserved pool.
"""

import json
import re
from pathlib import Path

import fsspec
from concurrent.futures import ThreadPoolExecutor
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.execution.types import this_output_path
from marin.execution.remote import remote

from experiments.exp1337_eval_suite import LOGPROB_TASKS
from experiments.grug.moe.eval_logprob import GrugLogprobEvalConfig, run_grug_logprob_eval
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.swarm_fisher_dsp import _HIDDEN_DIM, _BUDGET as _EVAL_BUDGET, _TARGET_STEPS

_OUTPUT_PREFIX = "gs://marin-us-central2/grug/"
_CANDIDATE_RE = re.compile(r"swarm_fisher_dsp_d512_(\d{6})-[a-f0-9]+")
_BUDGET = 840

# Logprob task surface — we keep growing this from the all_evals.py PR
# (marin-community/marin#2663) one eval at a time.
_EXTRA_LOGPROB_TASKS: tuple[EvalTaskConfig, ...] = (
    EvalTaskConfig("arc_easy", 0, task_alias="arc_easy_0shot"),
    EvalTaskConfig("arc_challenge", 0, task_alias="arc_challenge_0shot"),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    EvalTaskConfig("winogrande", 0, task_alias="winogrande_0shot"),
    EvalTaskConfig("piqa", 0, task_alias="piqa_0shot"),
    # social_iqa skipped: HF script-loader deprecated in datasets>=4.0;
    # bye-fork doesn't host a parquet copy.
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    EvalTaskConfig("openbookqa", 0, task_alias="openbookqa_0shot"),
    EvalTaskConfig("boolq", 0, task_alias="boolq_0shot"),
    EvalTaskConfig("commonsense_qa", 0, task_alias="csqa_0shot"),
    EvalTaskConfig("lambada_openai", 0, task_alias="lambada_0shot"),
    # wsc273 skipped: same HF script-loader issue (winograd_wsc.py).
    EvalTaskConfig("copa", 0, task_alias="copa_0shot"),
    EvalTaskConfig("truthfulqa_mc1", 0, task_alias="truthfulqa_mc1_0shot"),
    EvalTaskConfig("truthfulqa_mc2", 0, task_alias="truthfulqa_mc2_0shot"),
    # logiqa skipped: same HF script-loader issue (logiqa.py).
    # logiqa2 keeps: uses parquet-hosted datasets.
    # Further tasks added one-at-a-time by the orchestrator
    # (`experiments/grug/moe/_mega_evals_orchestrator.py`).
)

# MMLU 0/5-shot + GSM8K 5-shot + HumanEval 10-shot + arc_easy 0-shot.
_EVAL_TASKS: tuple[EvalTaskConfig, ...] = LOGPROB_TASKS + _EXTRA_LOGPROB_TASKS


def _find_finished_checkpoints() -> dict[int, str]:
    """Returns {idx: checkpoint_subpath} for every candidate whose
    .executor_status reads SUCCESS, picking the highest-step attempt per idx.
    The path is the GCS-prefix-relative checkpoints/ dir used by the
    Levanter checkpointer."""
    fs = fsspec.filesystem("gs")
    cand_dirs = [d for d in fs.ls(_OUTPUT_PREFIX, detail=False) if "swarm_fisher_dsp_d512_" in d]

    def probe(d):
        try:
            with fs.open(f"gs://{d}/.executor_status", "rt") as f:
                return d, f.read().strip() == "SUCCESS"
        except FileNotFoundError:
            return d, False

    with ThreadPoolExecutor(max_workers=64) as ex:
        results = list(ex.map(probe, cand_dirs))

    by_idx: dict[int, str] = {}
    for d, ok in results:
        if not ok:
            continue
        m = _CANDIDATE_RE.search(d)
        if not m:
            continue
        idx = int(m.group(1))
        existing = by_idx.get(idx)
        if existing is None or d > existing:
            by_idx[idx] = d
    return {idx: f"{d.removeprefix('marin-us-central2/')}/checkpoints" for idx, d in by_idx.items()}


_MODEL, _, _, _ = build_from_heuristic(
    budget=_EVAL_BUDGET,
    hidden_dim=_HIDDEN_DIM,
    target_steps=_TARGET_STEPS,
)


def _build_step(idx: int, ckpt_subpath: str, task: EvalTaskConfig) -> ExecutorStep:
    task_key = task.task_alias or f"{task.name}_{task.num_fewshot}shot"
    slug = f"swarm_fisher_dsp_d{_HIDDEN_DIM}_{idx:06d}"
    return ExecutorStep(
        name=f"evaluation/grug_logprob/{slug}/{task_key}",
        fn=remote(
            run_grug_logprob_eval,
            resources=ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=True),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=GrugLogprobEvalConfig(
            grug_model_config=_MODEL,
            checkpoint_path=InputName.hardcoded(ckpt_subpath),
            output_path=this_output_path(),
            wandb_run_name=f"{slug}_{task_key}",
            task=task,
            wandb_tags=("grug", "logprob_eval", "swarm_fisher_dsp", slug, task_key),
        ),
    )


_FINISHED = _find_finished_checkpoints()
print(f"swarm_mmlu_eval: {len(_FINISHED)} finished candidates × {len(_EVAL_TASKS)} tasks")

swarm_mmlu_steps: list[ExecutorStep] = [
    _build_step(idx, ckpt, task)
    for idx, ckpt in sorted(_FINISHED.items())
    for task in _EVAL_TASKS
]


if __name__ == "__main__":
    executor_main(
        steps=swarm_mmlu_steps,
        description=(
            f"Logprob eval suite (MMLU 0/5-shot, GSM8K 5-shot, HumanEval 10-shot) on "
            f"{len(_FINISHED)} finished swarm candidates (D512, ~100B-token grug-MoE). "
            "v4-8 preemptible, us-central2-b."
        ),
        max_concurrent=240,
    )
