# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load recorded HumanEval joint-decoding rollouts and their grades."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import fsspec

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.framework.schema import (
    COMPLETIONS_FILENAME,
    GRADES_FILENAME,
    PROMPTS_FILENAME,
    Completion,
    Grade,
    completions_file,
    grades_file,
    prompts_file,
    read_completion_rows,
    read_grade_rows,
    read_prompt_rows,
)

TOKEN_PATHS_FILENAME = "token_paths.jsonl.gz"
PROMPTS_STEP_NAME = "downstream_scaling/evals/prompts/humaneval"
DEFAULT_EXECUTOR_PREFIX = "gs://marin-us-central2"


@dataclass(frozen=True)
class Step:
    chunk: bytes
    tokens_a: tuple[int, ...]
    tokens_b: tuple[int, ...]


@dataclass(frozen=True)
class Rollout:
    problem_id: str
    completion_index: int
    sample_rank: int
    advisor_weight: float
    prompt: str
    steps: tuple[Step, ...]
    ended_with_eos: bool
    finish_reason: str | None
    passed: bool


def _parse_steps(record: dict[str, Any], *, context: str) -> tuple[tuple[Step, ...], bool]:
    steps: list[Step] = []
    raw_steps = record["steps"]
    for position, raw in enumerate(raw_steps):
        chunk = bytes.fromhex(raw["bytes_hex"])
        if not chunk:
            if position != len(raw_steps) - 1:
                raise ValueError(f"{context}: empty-bytes step at position {position} is not terminal")
            return tuple(steps), True
        steps.append(Step(chunk=chunk, tokens_a=tuple(raw["tokens_a"]), tokens_b=tuple(raw["tokens_b"])))
    return tuple(steps), False


def _localize(step_dir: str, cache_dir: str, filenames: tuple[str, ...]) -> str:
    if "://" not in step_dir:
        return step_dir
    local_dir = os.path.join(cache_dir, "inputs", os.path.basename(step_dir.rstrip("/")))
    os.makedirs(local_dir, exist_ok=True)
    for filename in filenames:
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            continue
        fs, remote_path = fsspec.url_to_fs(f"{step_dir.rstrip('/')}/{filename}")
        partial_path = f"{local_path}.partial.{os.getpid()}"
        fs.get_file(remote_path, partial_path)
        os.replace(partial_path, local_path)
    return local_dir


def load_rollouts(
    completions_output: str,
    *,
    prompts_output: str,
    grades_output: str,
    advisor_weights: Sequence[float],
    sample_ranks: Sequence[int],
    problem_ids: Sequence[str] | None = None,
    grade_filter: bool | None = None,
    limit: int | None = None,
    cache_dir: str | None = None,
) -> list[Rollout]:
    """Join and select recorded completions, token paths, prompts, and grades."""
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1 or None (got {limit})")
    if any(rank < 0 for rank in sample_ranks):
        raise ValueError(f"sample ranks must be non-negative: {list(sample_ranks)}")
    if cache_dir is not None:
        completions_output = _localize(completions_output, cache_dir, (TOKEN_PATHS_FILENAME, COMPLETIONS_FILENAME))
        prompts_output = _localize(prompts_output, cache_dir, (PROMPTS_FILENAME,))
        grades_output = _localize(grades_output, cache_dir, (GRADES_FILENAME,))

    wanted_ids = None if problem_ids is None else set(problem_ids)
    wanted_weights = set(advisor_weights)
    prompts = {
        row["id"]: row["prompt"]
        for row in read_prompt_rows(prompts_file(prompts_output))
        if wanted_ids is None or row["id"] in wanted_ids
    }
    if wanted_ids is not None and (missing := wanted_ids - prompts.keys()):
        raise ValueError(f"prompt ids not found in {prompts_file(prompts_output)}: {sorted(missing)}")

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    token_paths_file = os.path.join(completions_output, TOKEN_PATHS_FILENAME)
    with fsspec.open(token_paths_file, "rt", compression="infer") as f:
        for line in f:
            record = json.loads(line)
            if record["advisor_weight"] not in wanted_weights:
                continue
            if wanted_ids is not None and record["id"] not in wanted_ids:
                continue
            groups.setdefault((record["id"], record["advisor_weight"]), []).append(record)

    selection_ids = sorted(wanted_ids) if wanted_ids is not None else sorted({problem_id for problem_id, _ in groups})
    if not selection_ids:
        raise ValueError(f"no token paths at weights {sorted(wanted_weights)} in {completions_output}")
    selected: list[tuple[int, dict[str, Any]]] = []
    for problem_id in selection_ids:
        for weight in sorted(wanted_weights):
            group = groups.get((problem_id, weight))
            if group is None:
                raise ValueError(f"no token paths for problem {problem_id!r} at advisor weight {weight}")
            group.sort(key=lambda record: record["completion_index"])
            for rank in sample_ranks:
                if rank >= len(group):
                    raise ValueError(
                        f"sample rank {rank} out of range: {len(group)} samples for {problem_id!r} at weight {weight}"
                    )
                selected.append((rank, group[rank]))

    selected_ids = {record["id"] for _, record in selected}
    completions: dict[tuple[str, int], Completion] = {}
    for row in read_completion_rows(completions_file(completions_output)):
        if row["id"] in selected_ids:
            completions.update({(row["id"], index): completion for index, completion in enumerate(row["completions"])})

    grades: dict[tuple[str, int], Grade] = {}
    for row in read_grade_rows(grades_file(grades_output)):
        if row["id"] not in selected_ids:
            continue
        for index, grade in enumerate(row["grades"]):
            completion_index = int(grade.get("metadata", {}).get("completion_index", index))
            grades[(row["id"], completion_index)] = grade

    rollouts: list[Rollout] = []
    for sample_rank, record in selected:
        problem_id = record["id"]
        completion_index = record["completion_index"]
        weight = record["advisor_weight"]
        context = f"{problem_id} completion {completion_index}"
        if problem_id not in prompts:
            raise ValueError(f"{context}: id missing from {prompts_file(prompts_output)}")
        key = (problem_id, completion_index)
        if key not in completions:
            raise ValueError(f"{context}: missing from {completions_file(completions_output)}")
        if key not in grades:
            raise ValueError(f"{context}: missing from {grades_file(grades_output)}")
        completion = completions[key]
        grade = grades[key]
        if completion["metadata"]["advisor_weight"] != weight:
            raise ValueError(f"{context}: completion and token-path advisor weights differ")
        steps, ended_with_eos = _parse_steps(record, context=context)
        joined = b"".join(step.chunk for step in steps)
        if not joined.startswith(completion["text"].encode("utf-8")):
            raise ValueError(f"{context}: committed bytes do not extend the completion text")
        passed = bool(grade["metadata"]["passed"])
        if grade_filter is not None and passed != grade_filter:
            continue
        rollouts.append(
            Rollout(
                problem_id=problem_id,
                completion_index=completion_index,
                sample_rank=sample_rank,
                advisor_weight=weight,
                prompt=prompts[problem_id],
                steps=steps,
                ended_with_eos=ended_with_eos,
                finish_reason=completion["metadata"].get("finish_reason"),
                passed=passed,
            )
        )
    return rollouts if limit is None else rollouts[:limit]


def verify_shared_tokenizer_path(rollout: Rollout, vocab: xtok_selection.Vocab) -> None:
    """Verify that every recorded chunk is the shared side's single token."""
    for step_index, step in enumerate(rollout.steps, start=1):
        if step.tokens_a != step.tokens_b or len(step.tokens_a) != 1:
            raise ValueError(f"step {step_index}: not a shared-tokenizer path: {step}")
        token_id = step.tokens_a[0]
        piece = vocab.token_bytes[token_id] if token_id < len(vocab.token_bytes) else None
        if piece != step.chunk:
            raise ValueError(f"step {step_index}: chunk bytes do not match token id {token_id}")


def resolve_step_paths(slug: str, *, prefix: str = DEFAULT_EXECUTOR_PREFIX) -> tuple[str, str, str]:
    """Return the sweep's completions, prompts, and grades output paths."""
    from fray.cluster import ANY_REGION  # noqa: PLC0415
    from thalas.execution.context import executor_context  # noqa: PLC0415
    from thalas.execution.executor import Executor  # noqa: PLC0415

    from experiments.downstream_scaling.evals import (  # noqa: PLC0415
        run_delphi_humaneval_joint_decode_avg_xtok_llama as sweep,
    )

    pools = sweep.make_worker_pools(
        tpu_types=list(sweep.TPU_TYPES),
        worker_regions=[ANY_REGION],
        num_workers=sweep.WORKERS_PER_TPU_TYPE,
    )
    executor = Executor(prefix=prefix, executor_info_base_path=os.path.join(prefix, "experiments"))
    with executor_context():
        for step in sweep.build_run_steps(pools):
            executor.compute_version(step, is_pseudo_dep=False)

    completions = {path for step, path in executor.output_paths.items() if step.name.endswith(f"/{slug}/completions")}
    grades = {path for step, path in executor.output_paths.items() if step.name.endswith(f"/{slug}/grade")}
    prompts = {path for step, path in executor.output_paths.items() if step.name == PROMPTS_STEP_NAME}
    if len(completions) != 1 or len(grades) != 1 or len(prompts) != 1:
        raise ValueError(
            f"expected one completions, grade, and prompts path for {slug!r}; "
            f"found {len(completions)}, {len(grades)}, and {len(prompts)}"
        )
    return next(iter(completions)), next(iter(prompts)), next(iter(grades))
