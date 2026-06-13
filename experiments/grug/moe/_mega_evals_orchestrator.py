#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate adding mega-eval logprob tasks one-at-a-time to swarm_mmlu_eval.py.

For each task in ``_TASKS_TO_ADD``:
  1. Patch ``swarm_mmlu_eval.py`` so the task is appended to
     ``_EXTRA_LOGPROB_TASKS``.
  2. Submit the iris job.
  3. Poll until parent reaches a terminal state.
  4. Verify at least one candidate's ``results.json`` for the new task contains
     a real metric (any populated dict under ``results[<task_alias>]``).
  5. If parent succeeded OR counts climbed, move on. Otherwise, stop the loop
     so a human can investigate.

Designed to run as a single long-lived background bash invocation. Logs every
state change to stdout.
"""

import logging
import re
import subprocess
import time
from pathlib import Path

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mega_evals")

_SCRIPT_PATH = Path("experiments/grug/moe/swarm_mmlu_eval.py")
_EXTRA_BLOCK_START = "_EXTRA_LOGPROB_TASKS: tuple[EvalTaskConfig, ...] = ("
_EXTRA_BLOCK_END = ")\n"

# Task names + alias derived from PR mega-evals. Order = priority.
# arc_challenge is also the first task — already patched into the file by the
# previous (broken) orchestrator run, so the patcher's idempotency check
# (``if new_line in text:``) silently skips re-patching it.
_TASKS_TO_ADD: list[tuple[str, int, str]] = [
    ("arc_challenge", 0, "arc_challenge_0shot"),
    ("hellaswag", 0, "hellaswag_0shot"),
    ("winogrande", 0, "winogrande_0shot"),
    ("piqa", 0, "piqa_0shot"),
    # social_iqa dropped: HF script-loader deprecated, no parquet copy.
    ("sciq", 0, "sciq_0shot"),
    ("openbookqa", 0, "openbookqa_0shot"),
    ("boolq", 0, "boolq_0shot"),
    ("commonsense_qa", 0, "csqa_0shot"),
    ("lambada_openai", 0, "lambada_0shot"),
    # wsc273 dropped: same HF script-loader issue.
    ("copa", 0, "copa_0shot"),
    ("truthfulqa_mc1", 0, "truthfulqa_mc1_0shot"),
    ("truthfulqa_mc2", 0, "truthfulqa_mc2_0shot"),
    # logiqa + logiqa2 both dropped: same HF script-loader issue.
]

_PPL_PREFIX = "gs://marin-us-central2/evaluation/grug_logprob/"
_WANDB_KEY = "7c86993d6d6a1af7a92c1c22a44eb7aaccc50504"
_POLL_SECONDS = 300


def _add_task_to_script(task_name: str, num_fewshot: int, task_alias: str) -> None:
    """Append a new ``EvalTaskConfig`` line to ``_EXTRA_LOGPROB_TASKS``.

    Locate the tuple by its assignment, then walk a paren-depth counter from
    the opening ``(`` until depth returns to 0 — that's the tuple's closing
    paren, not the inner ``EvalTaskConfig(...)`` paren that a naive
    ``str.index(')')`` would find first.
    """
    text = _SCRIPT_PATH.read_text()
    new_line = f'    EvalTaskConfig("{task_name}", {num_fewshot}, task_alias="{task_alias}"),\n'
    if new_line in text:
        logger.info("task %s already in script, skipping patch", task_alias)
        return
    start = text.index(_EXTRA_BLOCK_START)
    paren_open = text.index("(", start)
    depth = 0
    paren_close = None
    for i in range(paren_open, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                paren_close = i
                break
    if paren_close is None:
        raise RuntimeError(f"could not find matching ')' for _EXTRA_LOGPROB_TASKS tuple at offset {paren_open}")
    patched = text[:paren_close] + new_line + text[paren_close:]
    _SCRIPT_PATH.write_text(patched)
    logger.info("patched script: appended %s", task_alias)


def _submit_iris_job() -> str:
    out = subprocess.run(
        [
            ".venv/bin/iris", "--cluster=marin", "job", "run", "--no-wait",
            "--cpu=1", "--memory=8G", "--enable-extra-resources", "--extra=cpu",
            "--region", "us-central2", "--priority", "production",
            "-e", "WANDB_API_KEY", _WANDB_KEY,
            "--", "python", "experiments/grug/moe/swarm_mmlu_eval.py",
        ],
        capture_output=True, text=True, check=True,
    )
    m = re.search(r"Job submitted: (/held/iris-run-\S+)", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"could not parse submitted job id; stdout={out.stdout!r} stderr={out.stderr!r}")
    return m.group(1)


def _poll_until_terminal(job_id: str) -> str:
    while True:
        result = subprocess.run(
            [".venv/bin/iris", "--cluster=marin", "job", "summary", job_id],
            capture_output=True, text=True,
        )
        text = result.stdout + result.stderr
        m = re.search(r"^State:\s+(\S+)", text, re.MULTILINE)
        state = m.group(1) if m else "unknown"
        logger.info("job %s state=%s", job_id, state)
        if state in ("succeeded", "failed", "killed", "cancelled"):
            return state
        time.sleep(_POLL_SECONDS)


def _count_real_results(task_alias: str) -> int:
    """Number of result.json files for this task that have a non-empty,
    error-free entry under ``results[<task_alias>]`` (or any task whose key
    contains the alias prefix — group tasks like mmlu_sl_verb publish under
    multiple sub-keys)."""
    fs = fsspec.filesystem("gs")
    paths = fs.glob(f"{_PPL_PREFIX}swarm_fisher_dsp_d512_*/{task_alias}*/results.json")
    n_real = 0
    for p in paths:
        try:
            import json
            with fs.open(f"gs://{p}", "rt") as f:
                blob = json.load(f)
        except Exception:  # pragma: no cover
            continue
        results = blob.get("results") or {}
        for k, v in results.items():
            if isinstance(v, dict) and any(
                isinstance(val, (int, float)) for key, val in v.items() if key not in ("alias",) and "stderr" not in key
            ):
                n_real += 1
                break
    return n_real


_INITIAL_JOB_TO_WAIT_ON = "/held/iris-run-swarm_mmlu_eval-20260612-162712"


def main() -> None:
    if _INITIAL_JOB_TO_WAIT_ON:
        logger.info("waiting on initial job %s before starting loop", _INITIAL_JOB_TO_WAIT_ON)
        state = _poll_until_terminal(_INITIAL_JOB_TO_WAIT_ON)
        logger.info("initial job %s reached state=%s", _INITIAL_JOB_TO_WAIT_ON, state)
        before = _count_real_results("arc_easy_0shot")
        logger.info("arc_easy_0shot real_results: %d", before)
    for task_name, num_fewshot, task_alias in _TASKS_TO_ADD:
        before = _count_real_results(task_alias)
        logger.info("=== adding %s (current real results: %d) ===", task_alias, before)
        _add_task_to_script(task_name, num_fewshot, task_alias)
        job_id = _submit_iris_job()
        logger.info("submitted %s", job_id)
        state = _poll_until_terminal(job_id)
        after = _count_real_results(task_alias)
        logger.info("after %s: state=%s real_results=%d (delta=+%d)", task_alias, state, after, after - before)
        if state != "succeeded" and after <= before:
            logger.error("STOPPING: %s neither succeeded nor produced new results", task_alias)
            return
    logger.info("all tasks added")


if __name__ == "__main__":
    main()
