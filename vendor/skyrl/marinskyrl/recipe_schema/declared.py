"""Typed recipe declarations whose consumers supply execution defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, TYPE_CHECKING

from pydantic import AfterValidator, Field, PlainSerializer, PositiveInt

from .model import FrozenMap, OpenMap, Section, SectionMap, field, thaw, unset_field

if TYPE_CHECKING:
    from .sections import EvalSamplingParams


class TeacherModel(Section):
    path: str = field()
    revision: str = field()


class TeacherEndpoint(Section):
    url: str = field()
    auth: str | None = unset_field()
    max_concurrency: PositiveInt = field()


class TeacherResources(Section):
    num_nodes: PositiveInt = field()
    gpus_per_node: PositiveInt = field()
    tensor_parallel_size: PositiveInt = field()
    colocation_group: str = field()
    max_num_batched_tokens: PositiveInt | None = unset_field()
    gpu_memory_utilization: int | float | None = unset_field()
    data_parallel_size: PositiveInt | None = unset_field()
    expert_parallel_size: PositiveInt | None = unset_field()


class Teacher(Section):
    source: Literal["openai_compatible", "local_inference", "frozen_worker", "resident"] = field()
    placement: Literal["external", "pinned", "rotating", "co_resident"] | None = unset_field()
    model: TeacherModel = field()
    evidence: Literal["chosen_token", "topk_distribution", "student_selected_topk"] = field()
    top_k: PositiveInt | None = unset_field()
    endpoints: tuple[TeacherEndpoint, ...] = unset_field()
    backend: Literal["vllm"] | None = unset_field()
    resources: TeacherResources | None = unset_field()
    tokenizer_fingerprint: str | None = unset_field()
    max_sequence_length: PositiveInt | None = unset_field()
    request_timeout_seconds: int | float | None = unset_field()


class TeacherRoute(Section):
    teacher: str = field()
    weight: int | float = field()


class TeacherRouting(Section):
    revision: str = field()
    routes: SectionMap[TeacherRoute] = field()


class TokenRule(Section):
    kind: Literal["token"] = field()
    action: Literal["truncate", "mask"] = field()
    low: int | float | None = unset_field()
    high: int | float | None = unset_field()


class SequenceRule(Section):
    kind: Literal["sequence"] = field()
    aggregate: Literal["geometric", "product", "extreme_token"] = field()
    action: Literal["truncate", "mask"] = field()
    low: int | float | None = unset_field()
    high: int | float | None = unset_field()


class EvaluationSampling(Section):
    sampling_params: EvalSamplingParams | None = unset_field()
    n_samples_per_prompt: PositiveInt | None = unset_field()


class EvaluationMinimum(Section):
    minimum: int | float = field()
    min_improvement: None = unset_field()


class EvaluationImprovement(Section):
    min_improvement: int | float = field()
    minimum: None = unset_field()


class CheckpointCallback(Section):
    type: Literal["checkpoint"] = field()
    save_steps: int = unset_field()
    save_on_train_end: bool = unset_field()


class DistillationTokenBudgetCallback(Section):
    type: Literal["distillation_token_budget"] = field()
    token_budget: PositiveInt = field()


class EvaluationCallback(Section):
    type: Literal["evaluation"] = field()
    eval_steps: int = unset_field()
    eval_on_train_end: bool = unset_field()
    eval_before_train: bool = unset_field()
    additional_evaluations: SectionMap[EvaluationSampling] | None = unset_field()
    metric_groups: Annotated[Mapping[str, tuple[str, ...]], AfterValidator(FrozenMap), PlainSerializer(thaw)] | None = (
        unset_field()
    )
    stop_when: SectionMap[EvaluationMinimum | EvaluationImprovement] | None = unset_field()


class HarborEvaluationCallback(Section):
    type: Literal["harbor_evaluation"] = field()
    cohorts: tuple[OpenMap, ...] = field()
    timeout: int | float = field()
    minimum_coverage: int | float = field()
    daytona_api_key_env: str = field()


class HFModelSaveCallback(Section):
    type: Literal["hf_model_save"] = field()
    save_steps: int = unset_field()
    save_on_train_end: bool = unset_field()


class DatabaseRegistrationCallback(Section):
    type: Literal["database_registration"] = field()
    agent_name: str | None = unset_field()
    enabled: bool = unset_field()


class RefModelUpdateCallback(Section):
    type: Literal["ref_model_update"] = field()
    update_every_epoch: bool = unset_field()


class ProgressCallback(Section):
    type: Literal["progress"] = field()
    log_interval: int = unset_field()


class LoggingCallback(Section):
    type: Literal["logging"] = field()
    log_every_step: bool = unset_field()


class PreflightGateCallback(Section):
    type: Literal["preflight_gate"] = field()
    enabled: bool = unset_field()
    min_reward: int | float = unset_field()
    max_reward: int | float = unset_field()
    on_failure: Literal["abort", "warn"] = unset_field()
    num_trials: int = unset_field()


class InferenceStatsCallback(Section):
    type: Literal["inference_stats"] = field()
    log_every_steps: int = unset_field()
    log_to_console: bool = unset_field()
    log_to_tracker: bool = unset_field()
    console_log_level: str = unset_field()
    poll_interval_seconds: int | float = unset_field()


LoggingBackend = Literal["wandb", "mlflow", "swanlab", "tensorboard", "console"]

CorrectionRule = Annotated[TokenRule | SequenceRule, Field(discriminator="kind")]

Callback = Annotated[
    CheckpointCallback
    | DistillationTokenBudgetCallback
    | EvaluationCallback
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
