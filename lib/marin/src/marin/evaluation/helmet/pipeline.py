# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from marin.evaluation.helmet.config import HelmetConfig, HelmetEvalName
from marin.evaluation.helmet.data import HelmetDataDownloadConfig, download_helmet_data
from marin.evaluation.helmet.openai_judge import HelmetOpenAiJudgeConfig, judge_with_helmet_scripts
from marin.evaluation.helmet.runner import HelmetRunConfig, run_helmet
from marin.evaluation.helmet.shas import resolve_git_sha, resolve_hf_dataset_sha
from marin.evaluation.helmet.stage_model import StageModelToGcsfuseConfig, stage_model_to_gcsfuse
from marin.evaluation.helmet.summarize import HelmetSummarizeConfig, summarize_helmet
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    get_executor_step,
    output_path_of,
    this_output_path,
)
from marin.utils import get_directory_friendly_name


def _chunked(items: tuple[HelmetEvalName, ...], n: int) -> list[tuple[HelmetEvalName, ...]]:
    if n <= 0:
        raise ValueError(f"evals_per_instance must be >= 1, got {n}")
    return [items[i : i + n] for i in range(0, len(items), n)]


def _looks_like_hf_repo_id(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def _step_already_on_gcsfuse(step: ExecutorStep) -> bool:
    if step.override_output_path and "gcsfuse_mount/" in step.override_output_path:
        return True
    return "gcsfuse_mount/" in step.name


def _needs_gcsfuse_staging(model_path: str | InputName | ExecutorStep) -> bool:
    if isinstance(model_path, str):
        if _looks_like_hf_repo_id(model_path):
            return False
        return "gcsfuse_mount/" not in model_path
    if isinstance(model_path, InputName):
        return not _step_already_on_gcsfuse(get_executor_step(model_path))
    if isinstance(model_path, ExecutorStep):
        return not _step_already_on_gcsfuse(model_path)
    return True


def _staging_prefix(model_path: str | InputName | ExecutorStep) -> str:
    if isinstance(model_path, str):
        return get_directory_friendly_name(model_path)
    if isinstance(model_path, InputName):
        step = get_executor_step(model_path)
        suffix = model_path.name or ""
        joined = f"{step.name}/{suffix}" if suffix else step.name
        return get_directory_friendly_name(joined)
    if isinstance(model_path, ExecutorStep):
        return get_directory_friendly_name(model_path.name)
    raise TypeError(f"Unsupported model_path type: {type(model_path)}")


@dataclass(frozen=True)
class HelmetPipelineConfig:
    """Top-level pipeline config; primarily for per-run metadata and grouping."""

    config_variant: Literal["full", "short"] = "full"
    enable_openai_judging: bool = False
    """Whether to run OpenAI-based judging steps for LongQA/Summ."""

    openai_num_shards: int = 8
    """Number of shards for OpenAI judging steps (smaller units of work)."""


HELMET_PIPELINE_ALL = HelmetPipelineConfig(config_variant="full", enable_openai_judging=True)
HELMET_PIPELINE_AUTOMATIC = HelmetPipelineConfig(config_variant="full", enable_openai_judging=False)
HELMET_PIPELINE_SHORT = HelmetPipelineConfig(config_variant="short", enable_openai_judging=False)


def helmet_steps(
    *,
    model_name: str,
    model_path: str | InputName | ExecutorStep,
    helmet: HelmetConfig,
    pipeline: HelmetPipelineConfig = HelmetPipelineConfig(),
    wandb_tags: list[str] | None = None,
) -> list[ExecutorStep]:
    """Build an Executor DAG that runs HELMET and aggregates results.

    This returns a list of steps in dependency order:
      1) data download
      2) one or more TPU evaluation steps
      3) summarize + W&B logging
    """
    use_chat_template = helmet.require_use_chat_template()

    helmet_repo_sha = helmet.helmet_repo_sha or resolve_git_sha(helmet.helmet_repo_url)
    helmet_data_sha = helmet.helmet_data_sha or resolve_hf_dataset_sha(
        helmet.helmet_data_repo_id, helmet.helmet_data_revision
    )

    stage_step: ExecutorStep | None = None
    staged_model_path: str | InputName | ExecutorStep = model_path
    if _needs_gcsfuse_staging(model_path):
        prefix = _staging_prefix(model_path)
        stage_step = ExecutorStep(
            name=f"gcsfuse_mount/models/helmet-staged/{prefix}",
            description="Stage model checkpoint into gcsfuse-backed storage for TPU-friendly reads.",
            fn=stage_model_to_gcsfuse,
            config=StageModelToGcsfuseConfig(
                input_path=model_path,  # type: ignore[arg-type]
                output_path=this_output_path(),  # type: ignore[arg-type]
            ),
        )
        staged_model_path = output_path_of(stage_step, "model")  # type: ignore[assignment]

    data_override = os.path.join(helmet.data_output_root, helmet_data_sha)
    data_step = ExecutorStep(
        name=f"evaluation/helmet/data/{helmet_data_sha[:8]}",
        description="Download and extract HELMET datasets to the gcsfuse mount.",
        fn=download_helmet_data,
        config=HelmetDataDownloadConfig(
            output_path=this_output_path(),  # type: ignore[arg-type]
            repo_id=helmet.helmet_data_repo_id,
            revision=helmet_data_sha,
            resolved_sha=helmet_data_sha,
        ),
    ).with_output_path(data_override)

    if helmet.evals_per_instance == "all":
        groups = [helmet.evals]
    else:
        groups = _chunked(helmet.evals, int(helmet.evals_per_instance))

    eval_steps: list[ExecutorStep] = []
    for idx, group in enumerate(groups):
        group_name = group[0] if len(group) == 1 else f"group_{idx:02d}"
        step = ExecutorStep(
            name=f"evaluation/helmet/{model_name}/{group_name}",
            description="Run HELMET via vLLM serving on TPUs.",
            fn=run_helmet,
            config=HelmetRunConfig(
                run_name=model_name,
                model_name_or_path=staged_model_path,  # type: ignore[arg-type]
                helmet_repo_url=helmet.helmet_repo_url,
                helmet_repo_sha=helmet_repo_sha,
                helmet_data_output_path=output_path_of(data_step),  # type: ignore[arg-type]
                evals=group,
                config_variant=pipeline.config_variant,
                use_chat_template=use_chat_template,
                seed=helmet.seed,
                tag=helmet.tag,
                output_path=this_output_path(),  # type: ignore[arg-type]
                resource_config=helmet.resource_config,
                vllm_serve_args=helmet.vllm_serve_args,
            ),
            pip_dependency_groups=["eval", "helmet"],
        )
        eval_steps.append(step)

    judge_steps: list[ExecutorStep] = []
    if pipeline.enable_openai_judging:
        eval_paths = [output_path_of(s) for s in eval_steps]  # type: ignore[list-item]
        for shard_idx in range(pipeline.openai_num_shards):
            judge_steps.append(
                ExecutorStep(
                    name=f"evaluation/helmet/{model_name}/judge_longqa/shard_{shard_idx:02d}",
                    description="OpenAI judging for HELMET LongQA (NarrativeQA).",
                    fn=judge_with_helmet_scripts,
                    config=HelmetOpenAiJudgeConfig(
                        kind="longqa",
                        helmet_repo_url=helmet.helmet_repo_url,
                        helmet_repo_sha=helmet_repo_sha,
                        helmet_data_output_path=output_path_of(data_step),  # type: ignore[arg-type]
                        model_name=model_name,
                        tag=helmet.tag,
                        eval_output_paths=eval_paths,  # type: ignore[arg-type]
                        output_path=this_output_path(),  # type: ignore[arg-type]
                        shard_idx=shard_idx,
                        num_shards=pipeline.openai_num_shards,
                    ),
                    pip_dependency_groups=["eval", "helmet"],
                )
            )
            judge_steps.append(
                ExecutorStep(
                    name=f"evaluation/helmet/{model_name}/judge_summ/shard_{shard_idx:02d}",
                    description="OpenAI judging for HELMET Summarization (InfiniteBench + MultiLexSum).",
                    fn=judge_with_helmet_scripts,
                    config=HelmetOpenAiJudgeConfig(
                        kind="summ",
                        helmet_repo_url=helmet.helmet_repo_url,
                        helmet_repo_sha=helmet_repo_sha,
                        helmet_data_output_path=output_path_of(data_step),  # type: ignore[arg-type]
                        model_name=model_name,
                        tag=helmet.tag,
                        eval_output_paths=eval_paths,  # type: ignore[arg-type]
                        output_path=this_output_path(),  # type: ignore[arg-type]
                        shard_idx=shard_idx,
                        num_shards=pipeline.openai_num_shards,
                    ),
                    pip_dependency_groups=["eval", "helmet"],
                )
            )

    summarize_step = ExecutorStep(
        name=f"evaluation/helmet/{model_name}/summarize",
        description="Aggregate HELMET outputs into a single report and log to W&B.",
        fn=summarize_helmet,
        config=HelmetSummarizeConfig(
            model_name=model_name,
            model_path=model_path,
            helmet_repo_url=helmet.helmet_repo_url,
            helmet_repo_sha=helmet_repo_sha,
            helmet_data_sha=helmet_data_sha,
            use_chat_template=use_chat_template,
            config_variant=pipeline.config_variant,
            eval_output_paths=[output_path_of(s) for s in [*eval_steps, *judge_steps]],  # type: ignore[arg-type]
            output_path=this_output_path(),  # type: ignore[arg-type]
            wandb_tags=wandb_tags,
            tag=helmet.tag,
            seed=helmet.seed,
        ),
        pip_dependency_groups=["eval"],
    )

    steps: list[ExecutorStep] = []
    if stage_step is not None:
        steps.append(stage_step)
    steps.append(data_step)
    steps.extend(eval_steps)
    steps.extend(judge_steps)
    steps.append(summarize_step)
    return steps
