# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mask-hint completion algorithm for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    MirroredValue,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.execution.remote import remote

from experiments.downstream_scaling.evals.algorithms.iid import (
    IIDConfig,
    IIDExecutionConfig,
    IIDSamplingConfig,
    make_iid_completion_step,
)
from experiments.downstream_scaling.evals.framework.core import EvalTask
from experiments.downstream_scaling.evals.framework.schema import (
    COMPLETIONS_FILENAME,
    GRADES_FILENAME,
    PROMPTS_FILENAME,
    prompts_file,
    read_completion_rows,
    read_grade_rows,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskGenerationConfig:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MaskSamplingConfig:
    num_hint_samples: int
    num_final_samples: int
    mask_fraction: float
    mask_text: str
    seed: int
    hint: MaskGenerationConfig
    final: MaskGenerationConfig


@dataclass(frozen=True)
class MaskExecutionConfig:
    num_workers: int
    chunk_size: int
    worker_resources: ResourceConfig


@dataclass(frozen=True)
class MaskConfig:
    hint_model_path: str | InputName | MirroredValue
    task: EvalTask
    sampling: MaskSamplingConfig
    execution: MaskExecutionConfig

    hint_name: str = "masked-hints"


@dataclass(frozen=True)
class MaskPromptsStepConfig:
    output_path: str
    model_path: str
    prompts_path: str
    hint_completions_path: str
    hint_grades_path: str
    mask_fraction: float
    mask_text: str
    seed: int


@dataclass(frozen=True)
class MaskCompletionAlgorithm:
    config: MaskConfig

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        hint_completions = make_hint_completion_step(
            name=f"{self.config.hint_name}/hint-completions",
            hint_model_path=self.config.hint_model_path,
            prompts_path=prompts_path,
            sampling=self.config.sampling,
            execution=self.config.execution,
        )
        hint_grades = self.config.task.make_grade_step(
            name=f"{self.config.hint_name}/hint-grade",
            prompts_path=prompts_path,
            completions_path=output_path_of(hint_completions) / COMPLETIONS_FILENAME,
        )
        masked_prompts = make_mask_prompts_step(
            name=f"{self.config.hint_name}/masked-prompts",
            model_path=model_path,
            prompts_path=prompts_path,
            hint_completions_path=output_path_of(hint_completions) / COMPLETIONS_FILENAME,
            hint_grades_path=output_path_of(hint_grades) / GRADES_FILENAME,
            config=self.config,
        )
        return make_iid_completion_step(
            name=name,
            model_path=model_path,
            prompts_path=output_path_of(masked_prompts) / PROMPTS_FILENAME,
            config=IIDConfig(
                sampling=IIDSamplingConfig(
                    n_samples=self.config.sampling.num_final_samples,
                    temperature=self.config.sampling.final.temperature,
                    top_p=self.config.sampling.final.top_p,
                    top_k=self.config.sampling.final.top_k,
                    max_tokens=self.config.sampling.final.max_tokens,
                    seed=self.config.sampling.seed,
                    stop=self.config.sampling.final.stop,
                ),
                execution=IIDExecutionConfig(
                    num_workers=self.config.execution.num_workers,
                    chunk_size=self.config.execution.chunk_size,
                    worker_resources=self.config.execution.worker_resources,
                ),
            ),
        )


def make_hint_completion_step(
    *,
    name: str,
    hint_model_path: str | InputName | MirroredValue,
    prompts_path: str | InputName | MirroredValue,
    sampling: MaskSamplingConfig,
    execution: MaskExecutionConfig,
) -> ExecutorStep:
    return make_iid_completion_step(
        name=name,
        model_path=hint_model_path,
        prompts_path=prompts_path,
        config=IIDConfig(
            sampling=IIDSamplingConfig(
                n_samples=sampling.num_hint_samples,
                temperature=sampling.hint.temperature,
                top_p=sampling.hint.top_p,
                top_k=sampling.hint.top_k,
                max_tokens=sampling.hint.max_tokens,
                seed=sampling.seed,
                stop=sampling.hint.stop,
            ),
            execution=IIDExecutionConfig(
                num_workers=execution.num_workers,
                chunk_size=execution.chunk_size,
                worker_resources=execution.worker_resources,
            ),
        ),
    )


def make_mask_prompts_step(
    *,
    name: str,
    model_path: str | InputName | MirroredValue,
    prompts_path: str | InputName | MirroredValue,
    hint_completions_path: str | InputName | MirroredValue,
    hint_grades_path: str | InputName | MirroredValue,
    config: MaskConfig,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            write_mask_prompts,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm"],
        ),
        config=MaskPromptsStepConfig(
            output_path=this_output_path(),
            model_path=version_path(model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            hint_completions_path=version_path(hint_completions_path),  # type: ignore[arg-type]
            hint_grades_path=version_path(hint_grades_path),  # type: ignore[arg-type]
            mask_fraction=versioned(config.sampling.mask_fraction),  # type: ignore[arg-type]
            mask_text=versioned(config.sampling.mask_text),  # type: ignore[arg-type]
            seed=versioned(config.sampling.seed),  # type: ignore[arg-type]
        ),
    )


def _correct_hint_indices(grades: list[dict[str, Any]]) -> list[int]:
    return [i for i, grade in enumerate(grades) if grade["score"] == 1.0]


def _mask_completion(tokenizer, text: str, mask_fraction: float, mask_text: str, rng: random.Random) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    mask_token_ids = tokenizer.encode(mask_text, add_special_tokens=False)
    masked_token_ids = []
    for token_id in token_ids:
        if rng.random() < mask_fraction:
            masked_token_ids.extend(mask_token_ids)
        else:
            masked_token_ids.append(token_id)
    return tokenizer.decode(masked_token_ids, skip_special_tokens=False)


def _load_tokenizer(model_path: str):
    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    logger.info("Resolved tokenizer %s -> %s", model_path, resolved_model_path)
    return load_tokenizer(resolved_model_path)


def write_mask_prompts(config: MaskPromptsStepConfig) -> None:
    tokenizer = _load_tokenizer(config.model_path)
    prompts = list(read_prompt_rows(config.prompts_path))
    hint_completions_by_id = {
        row["id"]: row["completions"] for row in read_completion_rows(config.hint_completions_path)
    }
    hint_grades_by_id = {row["id"]: row["grades"] for row in read_grade_rows(config.hint_grades_path)}

    path = prompts_file(config.output_path)
    written = 0
    with fsspec.open(path, "wt", compression="gzip") as f:
        for prompt in prompts:
            prompt_id = prompt["id"]
            hint_completions = hint_completions_by_id.get(prompt_id, [])
            hint_grades = hint_grades_by_id.get(prompt_id, [])
            correct_indices = _correct_hint_indices(hint_grades)
            if not correct_indices:
                continue

            rng = random.Random(f"{config.seed}:{prompt_id}:0")
            hint_index = rng.choice(correct_indices)
            masked_hint = _mask_completion(
                tokenizer,
                hint_completions[hint_index]["text"],
                config.mask_fraction,
                config.mask_text,
                rng,
            )
            f.write(json.dumps({"id": prompt_id, "prompt": prompt["prompt"] + masked_hint}) + "\n")
            written += 1
    logger.info("Wrote %d mask prompt rows to %s", written, path)
