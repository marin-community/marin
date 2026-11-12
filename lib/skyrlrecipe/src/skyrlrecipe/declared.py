# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed recipe declarations whose consumers supply execution defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, PlainSerializer, PositiveInt

from .model import FrozenMap, OpenMap, Section, SectionMap, thaw, unset_field
from .sampling import EvalSamplingParams


class TeacherModel(Section):
    path: str
    revision: str


class TeacherEndpoint(Section):
    url: str
    auth: str | None = unset_field()
    max_concurrency: PositiveInt


class TeacherResources(Section):
    num_nodes: PositiveInt
    gpus_per_node: PositiveInt
    tensor_parallel_size: PositiveInt
    colocation_group: str
    max_num_batched_tokens: PositiveInt | None = unset_field()
    gpu_memory_utilization: int | float | None = unset_field()
    data_parallel_size: PositiveInt | None = unset_field()
    expert_parallel_size: PositiveInt | None = unset_field()


class Teacher(Section):
    source: Literal["openai_compatible", "local_inference", "frozen_worker", "resident"]
    placement: Literal["external", "pinned", "rotating", "co_resident"] | None = unset_field()
    model: TeacherModel
    evidence: Literal["chosen_token", "topk_distribution", "student_selected_topk"]
    top_k: PositiveInt | None = unset_field()
    endpoints: tuple[TeacherEndpoint, ...] = unset_field()
    backend: Literal["vllm"] | None = unset_field()
    resources: TeacherResources | None = unset_field()
    tokenizer_fingerprint: str | None = unset_field()
    max_sequence_length: PositiveInt | None = unset_field()
    request_timeout_seconds: int | float | None = unset_field()


class TeacherRoute(Section):
    teacher: str
    weight: int | float


class TeacherRouting(Section):
    revision: str
    routes: SectionMap[TeacherRoute]


class TokenRule(Section):
    kind: Literal["token"]
    action: Literal["truncate", "mask"]
    low: int | float | None = unset_field()
    high: int | float | None = unset_field()


class SequenceRule(Section):
    kind: Literal["sequence"]
    aggregate: Literal["geometric", "product", "extreme_token"]
    action: Literal["truncate", "mask"]
    low: int | float | None = unset_field()
    high: int | float | None = unset_field()


EvaluationSchedule = Literal["every_evaluation", "final"]


class EvaluationSampling(Section):
    sampling_params: EvalSamplingParams | None = None
    n_samples_per_prompt: PositiveInt | None = None
    data_files: Annotated[tuple[Annotated[str, Field(min_length=1)], ...], Field(min_length=1)] | None = None
    schedule: EvaluationSchedule = "every_evaluation"


class EvaluationMinimum(Section):
    minimum: int | float
    min_improvement: None = unset_field()


class EvaluationImprovement(Section):
    min_improvement: int | float
    minimum: None = unset_field()


class CheckpointCallback(Section):
    type: Literal["checkpoint"]
    save_steps: int = unset_field()
    save_on_train_end: bool = unset_field()


class BestCheckpointCallback(Section):
    type: Literal["best_checkpoint"]
    initial_model_uri: str
    expected_tasks: int
    minimum_scored_tasks: int


class DistillationTokenBudgetCallback(Section):
    type: Literal["distillation_token_budget"]
    token_budget: PositiveInt


class EvaluationCallback(Section):
    type: Literal["evaluation"]
    eval_steps: int = unset_field()
    eval_on_train_end: bool = unset_field()
    eval_before_train: bool = unset_field()
    additional_evaluations: SectionMap[EvaluationSampling] | None = unset_field()
    metric_groups: Annotated[Mapping[str, tuple[str, ...]], AfterValidator(FrozenMap), PlainSerializer(thaw)] | None = (
        unset_field()
    )
    stop_when: SectionMap[EvaluationMinimum | EvaluationImprovement] | None = unset_field()


class ReferenceLikelihoodCallback(Section):
    type: Literal["reference_likelihood"]
    reference_paths: tuple[str, ...]
    output_path: str
    batch_size: PositiveInt = unset_field()


class HarborEvaluationCallback(Section):
    type: Literal["harbor_evaluation"]
    cohorts: tuple[OpenMap, ...]
    timeout: int | float
    minimum_coverage: int | float
    daytona_api_key_env: str


class HFModelSaveCallback(Section):
    type: Literal["hf_model_save"]
    save_steps: int = unset_field()
    save_on_train_end: bool = unset_field()


class DatabaseRegistrationCallback(Section):
    type: Literal["database_registration"]
    agent_name: str | None = unset_field()
    enabled: bool = unset_field()


class RefModelUpdateCallback(Section):
    type: Literal["ref_model_update"]
    update_every_epoch: bool = unset_field()


class ProgressCallback(Section):
    type: Literal["progress"]
    log_interval: int = unset_field()


class LoggingCallback(Section):
    type: Literal["logging"]
    log_every_step: bool = unset_field()


class PreflightGateCallback(Section):
    type: Literal["preflight_gate"]
    enabled: bool = unset_field()
    min_reward: int | float = unset_field()
    max_reward: int | float = unset_field()
    on_failure: Literal["abort", "warn"] = unset_field()
    num_trials: int = unset_field()


class InferenceStatsCallback(Section):
    type: Literal["inference_stats"]
    log_every_steps: int = unset_field()
    log_to_console: bool = unset_field()
    log_to_tracker: bool = unset_field()
    console_log_level: str = unset_field()
    poll_interval_seconds: int | float = unset_field()


LoggingBackend = Literal["wandb", "mlflow", "swanlab", "tensorboard", "console", "records"]

CorrectionRule = Annotated[TokenRule | SequenceRule, Field(discriminator="kind")]

Callback = Annotated[
    CheckpointCallback
    | BestCheckpointCallback
    | DistillationTokenBudgetCallback
    | EvaluationCallback
    | ReferenceLikelihoodCallback
    | HarborEvaluationCallback
    | HFModelSaveCallback
    | DatabaseRegistrationCallback
    | RefModelUpdateCallback
    | ProgressCallback
    | LoggingCallback
    | PreflightGateCallback
    | InferenceStatsCallback,
    Field(discriminator="type"),
]
