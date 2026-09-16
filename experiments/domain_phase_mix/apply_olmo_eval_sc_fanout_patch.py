# /// script
# requires-python = ">=3.11"
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply SC-only OLMo-Eval patches needed for fanout-safe OLMoBaseEval arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

RUNNER_IMPORT_OLD = """import asyncio
import contextlib
import multiprocessing as mp
import random
import time
"""

RUNNER_IMPORT_NEW = """import asyncio
import contextlib
import multiprocessing as mp
import os
import random
import time
"""

RUNNER_CONSTANT_OLD = """_DEFAULT_SANDBOX_ENV = "__default__"
_DEFAULT_PROCESS_POOL = "cpu"
"""

RUNNER_CONSTANT_NEW = """_DEFAULT_SANDBOX_ENV = "__default__"
_DEFAULT_PROCESS_POOL = "cpu"
_TASK_PREP_WORKERS_ENV = "OLMO_EVAL_TASK_PREP_WORKERS"


def _task_prep_worker_count(task_count: int) -> int:
    if task_count < 1:
        return 1
    raw = os.environ.get(_TASK_PREP_WORKERS_ENV)
    if raw is None or raw == "":
        return min(32, task_count)
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_TASK_PREP_WORKERS_ENV} must be an integer, got {raw!r}") from exc
    if workers < 1:
        raise ValueError(f"{_TASK_PREP_WORKERS_ENV} must be >= 1, got {workers}")
    return min(workers, task_count)
"""

RUNNER_EXECUTOR_OLD = """        # Collect prepared tasks in parallel, but accumulate results for deterministic ordering
        prepared_results: dict[str, tuple[TaskTracker, list[QueueItem]]] = {}
        with ThreadPoolExecutor(max_workers=min(32, len(expanded_tasks))) as executor:
            futures = {executor.submit(prepare_one, spec): spec for spec in expanded_tasks}
            for future in as_completed(futures):
                spec, tracker, task_items = future.result()
                trackers[spec] = tracker
                prepared_results[spec] = (tracker, task_items)
"""

RUNNER_EXECUTOR_NEW = """        # Collect prepared tasks in parallel, but accumulate results for deterministic ordering.
        # SC data-mixing arrays set OLMO_EVAL_TASK_PREP_WORKERS=1 to avoid HF datasets
        # cache/progress-bar races when many Slurm workers prepare tasks concurrently.
        prepared_results: dict[str, tuple[TaskTracker, list[QueueItem]]] = {}
        task_prep_workers = _task_prep_worker_count(len(expanded_tasks))
        if task_prep_workers == 1:
            for spec in expanded_tasks:
                spec, tracker, task_items = prepare_one(spec)
                trackers[spec] = tracker
                prepared_results[spec] = (tracker, task_items)
        else:
            with ThreadPoolExecutor(max_workers=task_prep_workers) as executor:
                futures = {executor.submit(prepare_one, spec): spec for spec in expanded_tasks}
                for future in as_completed(futures):
                    spec, tracker, task_items = future.result()
                    trackers[spec] = tracker
                    prepared_results[spec] = (tracker, task_items)
"""

BASIC_IMPORT_OLD = """import random
import sys
from collections.abc import Iterator
from typing import Any
"""

BASIC_IMPORT_NEW = """import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
"""

BASIC_CONSTANT_OLD = """_HF_BASE = "hf://datasets/allenai/basic-skills"
"""

BASIC_CONSTANT_NEW = """_HF_BASE = "hf://datasets/allenai/basic-skills"
_LOCAL_ROOT_ENV = "OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT"


def _data_file_for_subset(subset: str) -> str:
    local_root = os.environ.get(_LOCAL_ROOT_ENV)
    if not local_root:
        return f"{_HF_BASE}/{subset}/validation.json"
    path = Path(local_root) / subset / "validation.json"
    if not path.is_file():
        raise FileNotFoundError(f"{_LOCAL_ROOT_ENV} is set, but {path} does not exist")
    return str(path)
"""

BASIC_LOAD_OLD = """        data_file = f"{_HF_BASE}/{self._subset}/validation.json"
        ds = load_dataset("json", data_files=data_file, split="train")
"""

BASIC_LOAD_NEW = """        data_file = _data_file_for_subset(self._subset)
        ds = load_dataset("json", data_files=data_file, split="train")
"""

BASIC_DATASOURCE_OLD = """                data_files=f"{_HF_BASE}/{_subtask}/validation.json",
"""

BASIC_DATASOURCE_NEW = """                data_files=_data_file_for_subset(_subtask),
"""

OLMOBASE_ARC_BPB_OLD = """make_suite(
    name="arc:bpb:olmo3base",
    tasks=("arc_challenge:bpb:olmo3base", "arc_easy:bpb:olmo3base"),
)
"""

OLMOBASE_ARC_BPB_NEW = """make_suite(
    name="arc:bpb:olmo3base",
    tasks=("arc_challenge:olmo3base:bpb", "arc_easy:olmo3base:bpb"),
)
"""

OLMOBASE_PIQA_BPB_OLD = """"piqa:bpb:olmo3base","""

OLMOBASE_PIQA_BPB_NEW = """"piqa:olmo3base:bpb","""

OLMOBASE_CSQA_BPB_OLD = """"csqa:bpb:olmo3base","""

OLMOBASE_CSQA_BPB_NEW = """"csqa:olmo3base:bpb","""


def replace_once(text: str, old: str, new: str, *, file: Path) -> str:
    """Replace one exact block; return unchanged text when the patch is already present."""
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one unpatched block in {file}, found {count}")
    return text.replace(old, new, 1)


def patched_file_text(path: Path, replacements: list[tuple[str, str]]) -> tuple[str, bool]:
    """Return patched file text and whether it changed without writing the file."""
    original = path.read_text()
    updated = original
    for old, new in replacements:
        updated = replace_once(updated, old, new, file=path)
    return updated, updated != original


def apply_patches(olmo_eval_dir: Path, *, dry_run: bool = False) -> dict[str, bool]:
    """Apply fanout-safety patches to an OLMo-Eval checkout."""
    runner = olmo_eval_dir / "src" / "olmo_eval" / "runners" / "asynq" / "runner.py"
    basic = olmo_eval_dir / "src" / "olmo_eval" / "evals" / "tasks" / "basic_skills.py"
    olmobase = olmo_eval_dir / "src" / "olmo_eval" / "evals" / "suites" / "olmobase.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    if not basic.is_file():
        raise FileNotFoundError(basic)
    if not olmobase.is_file():
        raise FileNotFoundError(olmobase)
    planned = {
        runner: patched_file_text(
            runner,
            [
                (RUNNER_IMPORT_OLD, RUNNER_IMPORT_NEW),
                (RUNNER_CONSTANT_OLD, RUNNER_CONSTANT_NEW),
                (RUNNER_EXECUTOR_OLD, RUNNER_EXECUTOR_NEW),
            ],
        ),
        basic: patched_file_text(
            basic,
            [
                (BASIC_IMPORT_OLD, BASIC_IMPORT_NEW),
                (BASIC_CONSTANT_OLD, BASIC_CONSTANT_NEW),
                (BASIC_LOAD_OLD, BASIC_LOAD_NEW),
                (BASIC_DATASOURCE_OLD, BASIC_DATASOURCE_NEW),
            ],
        ),
        olmobase: patched_file_text(
            olmobase,
            [
                (OLMOBASE_ARC_BPB_OLD, OLMOBASE_ARC_BPB_NEW),
                (OLMOBASE_CSQA_BPB_OLD, OLMOBASE_CSQA_BPB_NEW),
                (OLMOBASE_PIQA_BPB_OLD, OLMOBASE_PIQA_BPB_NEW),
            ],
        ),
    }
    for path, (updated, changed) in planned.items():
        if changed and not dry_run:
            path.write_text(updated)
    return {str(path): changed for path, (_updated, changed) in planned.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-eval-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changed = apply_patches(args.olmo_eval_dir, dry_run=args.dry_run)
    for path, did_change in changed.items():
        if args.dry_run:
            status = "would_patch" if did_change else "already_patched"
        else:
            status = "patched" if did_change else "already_patched"
        print(f"{status}: {path}")


if __name__ == "__main__":
    main()
