"""
Built-in callbacks for common training operations.

These callbacks provide default implementations for checkpointing, evaluation,
model saving, and other periodic actions that were previously inline in the
training loop.

Supports two configuration styles:
1. Legacy interval configs (ckpt_interval, eval_interval, etc.)
2. New explicit callback configs in YAML:
   ```yaml
   trainer:
     callbacks:
       - type: checkpoint
         save_steps: 10
       - type: evaluation
         eval_steps: 20
   ```
"""

import asyncio
import contextlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from skyrlrecipe.declared import PooledCheckpointEvaluation

from skyrl_train.evaluation_profiles import (
    DEFAULT_EVALUATION_STEPS,
    validate_evaluation_schedule,
    EvaluationProfile,
    evaluation_profile_due,
    resolve_evaluation_profiles,
    additional_evaluation_metrics,
)
from skyrl_train.config.callbacks import has_explicit_callbacks, hf_export_destination_configured
from skyrl_train.distillation import DISTILLATION_SCORED_TOKENS_METRIC
from skyrl_train.json_serialization import to_jsonable
from skyrl_train.io import io
from rolloutengine.models import CUMULATIVE_GROSS_ROLLOUT_TOKENS_METRIC, GROSS_ROLLOUT_TOKENS_METRIC
from marinskyrl.checkpoint_paths import BEST_CHECKPOINT_FILENAME, POLICY_CHECKPOINT_SUBDIRECTORY
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.inference_engines.vllm.stats import VLLM_NUM_ENGINES_METRIC, IntervalReadMode
from skyrl_train.inference_observability import (
    InferenceMetricsSink,
    configured_inference_sinks,
    format_console_summary,
    trainer_metrics,
)

from skyrl_train.contextual_sampling import PromptThompsonOrder
from skyrl_train.dataset.tasks import TASK_EVALUATION_IDENTITY
from skyrl_train.mismatch_probe.callback import MismatchProbeCallback
from skyrl_train.probe_mix import CALLBACK_TYPE as DEVELOPMENT_MIX_TYPE, panel_identity
from skyrl_train.telemetry import record_event
from skyrl_train.callbacks.reference_likelihood import ReferenceLikelihoodCallback

from .base import (
    CallbackHandler,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    checkpoint_save_allowed,
    validate_save_after_fraction,
)
from .types import (
    CallbackErrorBehavior,
    CHECKPOINT_CALLBACK_TYPE,
    HF_MODEL_SAVE_CALLBACK_TYPE,
)

# Registry mapping callback type names to classes
# This enables YAML-based callback configuration
CALLBACK_REGISTRY: Dict[str, Type[TrainerCallback]] = {"reference_likelihood": ReferenceLikelihoodCallback}
EVALUATION_ATTEMPTED_METRIC = "eval/all/num_attempted"
EVALUATION_SCORED_METRIC = "eval/all/num_scored"


def register_callback(name: str):
    """
    Decorator to register a callback class in the registry.

    Args:
        name: The type name to use in YAML configs (e.g., "checkpoint")

    Example:
        @register_callback("my_callback")
        class MyCallback(TrainerCallback):
            ...
    """

    def decorator(cls: Type[TrainerCallback]) -> Type[TrainerCallback]:
        CALLBACK_REGISTRY[name] = cls
        return cls

    return decorator


def _configured_distillation_token_budget(cfg: DictConfig) -> int | None:
    value = cfg.trainer.get("distillation_token_budget")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("trainer.distillation_token_budget must be a positive integer")
    if cfg.trainer.algorithm.get("distillation") is None:
        raise ValueError("trainer.distillation_token_budget requires trainer.algorithm.distillation")
    return value


@register_callback(CHECKPOINT_CALLBACK_TYPE)
class CheckpointCallback(TrainerCallback):
    """
    Callback for saving training checkpoints at regular intervals.

    This replaces the inline `ckpt_interval` logic in the training loop.
    Checkpoints include model weights, optimizer state, and training state
    for resumable training.

    Args:
        save_steps: Save a checkpoint every N steps. Set to -1 or 0 to disable.
        save_on_train_end: Whether to save a final checkpoint when training ends.
        save_after_fraction: Suppress saves before this fraction of planned updates, including early termination.
    """

    def __init__(self, save_steps: int = 10, save_on_train_end: bool = True, save_after_fraction: float = 0.0):
        validate_save_after_fraction(save_after_fraction)
        self.save_steps = save_steps
        self.save_on_train_end = save_on_train_end
        self.save_after_fraction = save_after_fraction

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if (
            self.save_steps > 0
            and state.global_step % self.save_steps == 0
            and checkpoint_save_allowed(state, self.save_after_fraction)
        ):
            control.should_save = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_on_train_end and self.save_steps > 0 and checkpoint_save_allowed(state, self.save_after_fraction):
            control.should_save = True
        return control


@register_callback("distillation_token_budget")
class DistillationTokenBudgetCallback(TrainerCallback):
    """Stop either trainer after reaching a cumulative teacher-scored-token budget."""

    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(self, token_budget: int):
        if token_budget <= 0:
            raise ValueError("distillation token budget must be positive")
        self.token_budget = token_budget

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        scored = state.metrics.get(DISTILLATION_SCORED_TOKENS_METRIC)
        if scored is None:
            raise ValueError("distillation token budget requires teacher-scored token metrics")
        scored_tokens = int(scored)
        if scored_tokens < 0 or scored_tokens != scored:
            raise ValueError(f"{DISTILLATION_SCORED_TOKENS_METRIC} must be a non-negative integer, got {scored!r}")
        trainer = kwargs["trainer"]
        trainer.distillation_scored_tokens_total += scored_tokens
        trainer.all_metrics.update(
            {
                "distillation/scored_tokens_total": float(trainer.distillation_scored_tokens_total),
                "distillation/token_budget": float(self.token_budget),
            }
        )
        if trainer.distillation_scored_tokens_total >= self.token_budget:
            logger.info(
                "Reached distillation token budget: scored_tokens={} budget={}",
                trainer.distillation_scored_tokens_total,
                self.token_budget,
            )
            control.should_training_stop = True
        return control


class RolloutTokenBudgetCallback(TrainerCallback):
    """Schedule evaluations and stop synchronous training by gross rollout spend."""

    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(self, budget: int, eval_milestones: tuple[int, ...] = ()):
        self.budget = budget
        self.eval_milestones = eval_milestones

    def _crossed_milestone(self, previous: int, total: int) -> bool:
        return any(previous < milestone <= total for milestone in self.eval_milestones)

    def on_train_begin(self, state: TrainerState, control: TrainerControl, *, trainer, **kwargs) -> TrainerControl:
        if state.global_step > 0:
            control.should_evaluate = False
        total = trainer.rollout_gross_response_tokens_total
        if total >= self.budget:
            control.should_training_stop = True
        elif state.global_step > 0 and self.eval_milestones:
            # The checkpoint can be saved before this update's evaluation.
            previous = total - trainer.rollout_gross_response_tokens_last_update
            control.should_evaluate = self._crossed_milestone(previous, total)
        return control

    def on_step_end(self, state: TrainerState, control: TrainerControl, *, trainer, **kwargs) -> TrainerControl:
        gross = state.metrics[GROSS_ROLLOUT_TOKENS_METRIC]
        tokens = int(gross)
        if tokens < 0 or tokens != gross:
            raise ValueError(f"{GROSS_ROLLOUT_TOKENS_METRIC} must be a non-negative integer, got {gross!r}")
        trainer.rollout_gross_response_tokens_total += tokens
        trainer.rollout_gross_response_tokens_last_update = tokens
        total = trainer.rollout_gross_response_tokens_total
        trainer.all_metrics[CUMULATIVE_GROSS_ROLLOUT_TOKENS_METRIC] = total
        if total >= self.budget:
            logger.info(
                "Reached rollout token budget at update {}: {} >= {}", state.global_step, total, self.budget
            )
            control.should_training_stop = True
            # Publish the stopping update's metrics before the terminal evaluation can fail.
            control.should_evaluate = False
        elif self._crossed_milestone(total - tokens, total):
            control.should_evaluate = True
        return control

    def on_train_end(self, state: TrainerState, control: TrainerControl, *, trainer, **kwargs) -> TrainerControl:
        total = trainer.rollout_gross_response_tokens_total
        if total < self.budget:
            raise ValueError(
                f"Training ended at update {state.global_step} with {total} gross rollout tokens "
                f"before reaching the configured budget of {self.budget}"
            )
        # A retry can restore a checkpoint written before the stopping update's evaluation.
        control.should_evaluate = True
        return control


@register_callback(DEVELOPMENT_MIX_TYPE)
class DevelopmentProbeMixCallback(TrainerCallback):
    """Apply a development-only baseline or milestone before the next rollout publish."""

    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(self, panel_sha256: str, expected_tasks: int):
        self.panel_sha256 = panel_sha256
        self.expected_tasks = expected_tasks

    def on_train_begin(self, state: TrainerState, control: TrainerControl, *, trainer, **kwargs) -> TrainerControl:
        order = trainer.context.loader.order
        if not isinstance(order, PromptThompsonOrder) or order.development_mix is None:
            raise ValueError("Development probe mix requires a configured source-pass order")
        if state.global_step > 0 and order.development_mix.initial is None:
            raise ValueError("The resumed rollout checkpoint is missing development probe state")
        identities = trainer.eval_dataset.dataframe.with_format(None)[TASK_EVALUATION_IDENTITY]
        digests = {json.loads(value)["task_digest"] for value in identities if value is not None}
        if len(identities) != self.expected_tasks or len(digests) != self.expected_tasks:
            raise ValueError("Development probe requires a unique identity for every pinned task")
        if panel_identity(digests) != self.panel_sha256:
            raise ValueError("The primary evaluation is not the pinned development probe")
        return control

    async def on_evaluate_async(
        self, state: TrainerState, control: TrainerControl, *, trainer, metrics: dict[str, float], **kwargs
    ) -> TrainerControl:
        total = trainer.rollout_gross_response_tokens_total
        milestones = trainer.cfg.trainer.rollout_token_eval_milestones
        last = trainer.rollout_gross_response_tokens_last_update
        if state.global_step and (
            total >= trainer.cfg.trainer.rollout_token_budget or not total - last < milestones[0] <= total
        ):
            return control
        if metrics.get(EVALUATION_ATTEMPTED_METRIC) != self.expected_tasks or (
            metrics.get(EVALUATION_SCORED_METRIC) != self.expected_tasks
        ):
            raise ValueError("Development probe mix requires complete verifier coverage")
        decision = trainer.context.loader.order.observe_development_probe(
            step=state.global_step, accuracy=metrics["eval/all/avg_verifier_score"], gross_tokens=total
        )
        if decision is not None:
            record_event(
                "development_probe_mix_decision",
                {
                    **vars(decision),
                    "panel_sha256": self.panel_sha256,
                    "gross_training_tokens": total,
                    "scored_tasks": self.expected_tasks,
                    "development_response_tokens": metrics["eval/all/response_tokens"],
                },
            )
        return control


@dataclass(frozen=True)
class EvaluationStopConfig:
    minimum: float | None = None
    min_improvement: float | None = None

    def __post_init__(self):
        if (self.minimum is None) == (self.min_improvement is None):
            raise ValueError("evaluation stop requires exactly minimum or min_improvement")
        threshold = self.minimum if self.minimum is not None else self.min_improvement
        if not math.isfinite(threshold):
            raise ValueError("evaluation stop thresholds must be finite")


@register_callback("best_checkpoint")
class BestCheckpointCallback(TrainerCallback):
    """Persist the best sufficiently covered verifier score, retaining the initial policy on ties."""

    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(
        self,
        *,
        initial_model_uri: str,
        expected_tasks: int,
        minimum_scored_tasks: int,
        selection_metric: str = "eval/all/avg_verifier_score",
        save_after_fraction: float = 0.0,
        preservation_max_drop: dict[str, float] | None = None,
    ):
        validate_save_after_fraction(save_after_fraction)
        self.initial_model_uri = initial_model_uri
        self.expected_tasks = expected_tasks
        self.minimum_scored_tasks = minimum_scored_tasks
        self.best_score: float | None = None
        self.selection_metric = selection_metric
        self.save_after_fraction = save_after_fraction
        self.preservation_max_drop = dict(preservation_max_drop or {})
        if any(not math.isfinite(drop) or drop < 0 for drop in self.preservation_max_drop.values()):
            raise ValueError("Preservation drops must be finite and nonnegative")
        self.preservation_baseline: dict[str, float] = {}

    async def on_evaluate_async(self, state, control, *, metrics, trainer, **kwargs):
        root = trainer.cfg.trainer.ckpt_path
        best_path = join_resource_path(root, BEST_CHECKPOINT_FILENAME)
        if self.best_score is None and io.exists(best_path):
            with io.open_file(best_path, "r") as stream:
                saved = json.load(stream)
                self.best_score = saved["accuracy"]
                self.preservation_baseline = saved.get("preservation_baseline", {})
        attempted = metrics[EVALUATION_ATTEMPTED_METRIC]
        scored = metrics[EVALUATION_SCORED_METRIC]
        ineligible = (
            attempted != self.expected_tasks
            or scored < self.minimum_scored_tasks
            or not metrics.get("eval/selection_eligible", 1.0)
        )
        if state.global_step == 0 and self.preservation_max_drop and ineligible:
            raise ValueError("Preservation-gated selection requires an eligible step-zero evaluation")
        if ineligible:
            return control
        if state.global_step == 0:
            self.preservation_baseline = {key: metrics[key] for key in self.preservation_max_drop}
            if any(not math.isfinite(value) for value in self.preservation_baseline.values()):
                raise ValueError("Preservation baseline must be finite")
        if set(self.preservation_baseline) != set(self.preservation_max_drop):
            raise ValueError("Preservation-gated selection requires step-zero evaluation or restored baseline")
        for key, maximum_drop in self.preservation_max_drop.items():
            value = metrics[key]
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite preservation metric: {key}")
            if value < self.preservation_baseline[key] - maximum_drop:
                return control
        score = metrics[self.selection_metric]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Best-checkpoint selection requires a finite normalized verifier score")
        if self.best_score is not None and score <= self.best_score:
            return control
        if state.global_step and not checkpoint_save_allowed(state, self.save_after_fraction):
            return control
        model_uri = self.initial_model_uri
        model_format = "hf"
        if state.global_step:
            await trainer.save_checkpoints()
            model_uri = join_resource_path(
                trainer.cfg.trainer.ckpt_path, f"global_step_{state.global_step}", POLICY_CHECKPOINT_SUBDIRECTORY
            )
            model_format = "native"
        payload = {
            "step": state.global_step,
            "accuracy": score,
            "attempted": attempted,
            "scored": scored,
            "model_uri": model_uri,
            "model_format": model_format,
            "selection": "Highest eligible selected metric; keep the earliest checkpoint on ties",
            "selection_metric": self.selection_metric,
            "preservation_baseline": self.preservation_baseline,
            "preservation_max_drop": self.preservation_max_drop,
            "save_after_fraction": self.save_after_fraction,
        }
        io.makedirs(root, exist_ok=True)
        encoded = json.dumps(payload).encode()
        io.write_bytes_atomic(best_path, encoded)
        export_root = trainer.cfg.trainer.export_path
        io.makedirs(export_root, exist_ok=True)
        io.write_bytes_atomic(join_resource_path(export_root, BEST_CHECKPOINT_FILENAME), encoded)
        self.best_score = score
        return control


def pool_checkpoint_scores(metrics: Dict[str, float], policy: PooledCheckpointEvaluation) -> None:
    """Pool the scores of individual tasks and assess each source against its declared population."""
    attempted = scored = 0.0
    reward_sums = []
    eligible = True
    sources = {"eval/all": policy.primary, **{f"eval/{name}/all": count for name, count in policy.cohorts.items()}}
    for prefix, coverage in sources.items():
        tried = metrics[f"{prefix}/num_attempted"]
        included = metrics.get(f"{prefix}/num_scored", 0.0)
        if tried != coverage.expected_tasks or included < coverage.minimum_scored_tasks:
            eligible = False
            logger.warning(
                "Insufficient checkpoint evaluation coverage for {}: {} scored, {} attempted, {} expected",
                prefix, included, tried, coverage.expected_tasks,
            )
        attempted += tried
        scored += included
        if included:
            reward_sums.append(metrics[f"{prefix}/avg_verifier_score"] * included)
    metrics["eval/selection_eligible"] = float(eligible)
    metrics[EVALUATION_ATTEMPTED_METRIC] = attempted
    metrics[EVALUATION_SCORED_METRIC] = scored
    metrics["eval/all/verifier_score_coverage"] = scored / attempted if attempted else 0.0
    if scored:
        metrics["eval/all/avg_verifier_score"] = math.fsum(reward_sums) / scored
    else:
        metrics.pop("eval/all/avg_verifier_score", None)


@register_callback("evaluation")
class EvaluationCallback(TrainerCallback):
    """Schedule evaluations and stop when configured score requirements are met.

    Args:
        eval_steps: Evaluate every N completed steps; non-positive values disable evaluation.
        eval_on_train_end: Evaluate the final policy. A final profile also requests this evaluation.
        eval_before_train: Evaluate the initial policy.
        additional_evaluations: Named sampling, sample count, dataset and schedule overrides.
            The schedule is every_evaluation or final (the configured last update only).
            A final profile at an off-periodic last update evaluates the primary dataset too.
            Resuming at the last update reruns final evaluation. It requires a primary evaluation
            dataset, and does not run after an early stop.
        metric_groups: Output metric names mapped to source metrics whose mean is reported.
        stop_when: Metrics mapped to minimum scores or gains over the initial evaluation.
        checkpoint_scores: Pool verifier scores across the primary evaluation and the named cohorts.
    """

    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(
        self,
        eval_steps: int = DEFAULT_EVALUATION_STEPS,
        eval_on_train_end: bool = True,
        eval_before_train: bool = True,
        additional_evaluations: Dict[str, EvaluationProfile] | None = None,
        metric_groups: Dict[str, List[str]] | None = None,
        stop_when: Dict[str, Dict[str, float]] | None = None,
        checkpoint_scores: PooledCheckpointEvaluation | None = None,
    ):
        self.eval_steps = eval_steps
        self.eval_on_train_end = eval_on_train_end
        self.eval_before_train = eval_before_train
        self.additional_evaluations = additional_evaluations or {}
        self.checkpoint_scores = checkpoint_scores
        self.metric_groups = metric_groups or {}
        self.stop_when = {name: EvaluationStopConfig(**value) for name, value in (stop_when or {}).items()}
        self._initial_values: Dict[str, float] = {}
        self._initial_step: int | None = None
        if any(not keys for keys in self.metric_groups.values()):
            raise ValueError("evaluation metric groups must be nonempty")
        if checkpoint_scores is not None:
            for name in checkpoint_scores.cohorts:
                profile = self.additional_evaluations.get(name)
                if profile is None or profile.schedule != "every_evaluation" or profile.n_samples_per_prompt != 1:
                    raise ValueError(f"Checkpoint cohort {name} requires one sample per task at every evaluation")

    async def collect_evaluation(self, state, *, trainer, pool):
        metrics = {}

        async def collect(name, profile):
            scoring = self.checkpoint_scores
            timeout = scoring.timeout if scoring is not None and name in scoring.cohorts else None
            async with asyncio.timeout(timeout):
                additional = await trainer.eval(
                    val_set_name=name,
                    profile=profile,
                    evaluation_pool=pool,
                )
            metrics.update(additional_evaluation_metrics(additional, name))

        async with asyncio.TaskGroup() as tasks:
            for name, profile in self.additional_evaluations.items():
                if evaluation_profile_due(profile, step=state.global_step, last_step=state.is_last_step):
                    tasks.create_task(collect(name, profile))
        return metrics

    async def on_evaluate_async(
        self, state: TrainerState, control: TrainerControl, *, metrics: Dict[str, float], trainer, **kwargs
    ) -> TrainerControl:
        if self.checkpoint_scores is not None:
            pool_checkpoint_scores(metrics, self.checkpoint_scores)
        for name, members in self.metric_groups.items():
            metrics[name] = math.fsum(metrics[member] / len(members) for member in members)
            if not math.isfinite(metrics[name]):
                raise ValueError(f"nonfinite evaluation metric group {name}")
        if not self.stop_when:
            return control
        if self._initial_step is None:
            self._initial_values = {name: metrics[name] for name in self.stop_when}
            self._initial_step = state.global_step
        improvements = {name: metrics[name] - initial for name, initial in self._initial_values.items()}
        for name, improvement in improvements.items():
            if not math.isfinite(improvement):
                raise ValueError(f"nonfinite evaluation improvement for {name}")
            metrics[f"{name}_improvement"] = improvement
        if state.global_step > self._initial_step and all(
            metrics[name] >= requirement.minimum
            if requirement.minimum is not None
            else improvements[name] >= requirement.min_improvement
            for name, requirement in self.stop_when.items()
        ):
            logger.info("Evaluation stop requirements reached at step {}", state.global_step)
            control.should_training_stop = True
        return control

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.eval_before_train and self.eval_steps > 0:
            control.should_evaluate = True
        return control

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            control.should_evaluate = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        final_due = any(
            profile.schedule == "final"
            and evaluation_profile_due(profile, step=state.global_step, last_step=state.is_last_step)
            for profile in self.additional_evaluations.values()
        )
        if (self.eval_on_train_end and self.eval_steps > 0) or final_due:
            control.should_evaluate = True
        return control


@register_callback(HF_MODEL_SAVE_CALLBACK_TYPE)
class HFModelSaveCallback(TrainerCallback):
    """
    Callback for requesting Hugging Face exports at the end of training and, optionally, at intervals.

    Normal training records a request beside the immutable sharded checkpoint;
    a dedicated export job later converts and optionally publishes that checkpoint.

    Args:
        save_steps: Request an HF export every N steps. Set to -1 or 0 to request only the
            terminal export.
        save_on_train_end: Whether to request a final HF export when training ends.
        save_after_fraction: Suppress exports before this fraction of planned updates, including early termination.
    """

    def __init__(self, save_steps: int = -1, save_on_train_end: bool = True, save_after_fraction: float = 0.0):
        validate_save_after_fraction(save_after_fraction)
        self.save_steps = save_steps
        self.save_on_train_end = save_on_train_end
        self.save_after_fraction = save_after_fraction

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if (
            self.save_steps > 0
            and state.global_step % self.save_steps == 0
            and checkpoint_save_allowed(state, self.save_after_fraction)
        ):
            control.should_save_hf_model = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_on_train_end and checkpoint_save_allowed(state, self.save_after_fraction):
            control.should_save_hf_model = True
        return control


@register_callback("database_registration")
class DatabaseRegistrationCallback(TrainerCallback):
    """
    Callback for registering trained models to the unified database (Supabase).

    This callback runs at training end and registers the trained model along with:
    - Training timestamps (start/end)
    - Training configuration (hyperparameters, algorithm, etc.)
    - W&B link (if available)
    - Dataset and base model references

    Requirements:
    - KEYS environment variable pointing to Supabase credentials file, OR
    - Direct env vars: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY

    Args:
        agent_name: Name of the agent being trained. Defaults to TRAINING_AGENT_NAME or "skyrl".
        enabled: Whether registration is enabled (default: True, auto-disabled if no credentials)
    """

    def __init__(
        self,
        agent_name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.agent_name = agent_name
        self.enabled = enabled
        self._training_start: Optional[str] = None
        self._supabase_ready = False
        self._cfg = None

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Record training start time and load Supabase credentials."""
        from datetime import datetime, timezone

        # Only register from rank 0
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank != 0:
            self.enabled = False
            return control

        self._training_start = datetime.now(timezone.utc).isoformat()

        # Try to load Supabase credentials
        try:
            from skyrl_train.callbacks.database.utils import load_supabase_keys

            required_keys = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"]
            if not all(os.environ.get(k) for k in required_keys):
                self._supabase_ready = load_supabase_keys()
            else:
                self._supabase_ready = True

            if self._supabase_ready:
                logger.info("DatabaseRegistrationCallback: Supabase credentials loaded")
            else:
                logger.warning(
                    "DatabaseRegistrationCallback: Supabase credentials not available, "
                    "model will not be registered to database"
                )
        except ImportError as e:
            logger.warning(
                f"DatabaseRegistrationCallback: database module not available ({e}), "
                "install supabase-py to enable database registration: pip install supabase"
            )
            self.enabled = False

        # Store config reference
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "cfg"):
            self._cfg = trainer.cfg

        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Register the trained model to the database."""
        if not self.enabled or not self._supabase_ready:
            return control

        from datetime import datetime, timezone

        try:
            from skyrl_train.callbacks.database.utils import register_trained_model
        except ImportError:
            logger.error("DatabaseRegistrationCallback: Cannot import register_trained_model")
            return control

        training_end = datetime.now(timezone.utc).isoformat()

        # Extract configuration
        cfg = self._cfg
        if cfg is None:
            logger.warning("DatabaseRegistrationCallback: No config available, skipping registration")
            return control

        agent_name = self.agent_name or os.environ.get("TRAINING_AGENT_NAME", "skyrl")

        # Get model path and dataset info
        policy_cfg = getattr(cfg.trainer, "policy", None)
        base_model_name = None
        if policy_cfg:
            model_cfg = getattr(policy_cfg, "model", None)
            if model_cfg:
                base_model_name = getattr(model_cfg, "path", None)

        # Get dataset names
        data_cfg = getattr(cfg, "data", None)
        train_data = getattr(data_cfg, "train_data", []) if data_cfg else []
        dataset_names = list(train_data) if isinstance(train_data, (list, tuple)) else [train_data]

        # Get HF repo ID for weights location
        hf_hub_repo_id = getattr(cfg.trainer, "hf_hub_repo_id", None)

        # Get W&B link
        wandb_link = None
        try:
            import wandb

            if wandb.run is not None:
                wandb_link = wandb.run.url
        except Exception:
            pass

        training_params = {
            "trainer": to_jsonable(cfg.trainer) if hasattr(cfg, "trainer") else {},
            "generator": to_jsonable(cfg.generator) if hasattr(cfg, "generator") else {},
            "algorithm": str(getattr(cfg.trainer.algorithm, "advantage_estimator", "unknown")),
        }

        # Build registration record
        record = {
            "agent_name": agent_name,
            "training_start": self._training_start,
            "training_end": training_end,
            "created_by": os.environ.get("JOB_CREATOR", ""),
            "base_model_name": base_model_name,
            "dataset_names": dataset_names,
            "training_type": "RL",
            "training_parameters": training_params,
            "wandb_link": wandb_link or "",
            "traces_location_s3": os.environ.get("TRACE_S3_PATH", ""),
            "model_name": hf_hub_repo_id,
        }

        logger.info(
            f"DatabaseRegistrationCallback: Registering model to database "
            f"(agent={agent_name}, base_model={base_model_name}, datasets={dataset_names})"
        )

        try:
            result = register_trained_model(record)

            if result.get("success"):
                model = result.get("model", {})
                model_name = model.get("name", "unknown")
                if result.get("exists"):
                    logger.info(f"DatabaseRegistrationCallback: Model '{model_name}' already exists in database")
                elif result.get("updated"):
                    logger.info(f"DatabaseRegistrationCallback: Updated existing model '{model_name}'")
                else:
                    logger.info(f"DatabaseRegistrationCallback: Registered new model '{model_name}'")
            else:
                logger.error(f"DatabaseRegistrationCallback: Registration failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"DatabaseRegistrationCallback: Exception during registration: {e}")

        return control


@register_callback("ref_model_update")
class RefModelUpdateCallback(TrainerCallback):
    """
    Callback for updating the reference model with policy weights at epoch boundaries.

    This replaces the inline `update_ref_every_epoch` logic in the training loop.
    The reference model is used for KL divergence calculations in algorithms
    like PPO and GRPO.

    Args:
        update_every_epoch: Whether to update the reference model at the end of each epoch.
    """

    def __init__(self, update_every_epoch: bool = False):
        self.update_every_epoch = update_every_epoch
        self._should_update_ref = False

    def on_epoch_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        # Mark that we should update ref model
        # The actual update is handled by the trainer when it processes this flag
        if self.update_every_epoch and not state.is_last_step:
            # Skip updating ref at the end of the last epoch (as the original code did)
            self._should_update_ref = True
        return control

    @property
    def should_update_ref(self) -> bool:
        """Check if ref model should be updated and reset the flag."""
        result = self._should_update_ref
        self._should_update_ref = False
        return result


@register_callback("progress")
class ProgressCallback(TrainerCallback):
    """
    Callback for tracking and displaying training progress.

    This provides a central place for progress tracking without modifying
    the core training loop.

    Args:
        log_interval: Log progress every N steps. Default is every step.
    """

    def __init__(self, log_interval: int = 1):
        self.log_interval = log_interval

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_interval > 0 and state.global_step % self.log_interval == 0:
            logger.info(
                f"Step {state.global_step}/{state.total_steps} (Epoch {state.epoch + 1}, Step {state.step_in_epoch})"
            )
        return control


@register_callback("logging")
class LoggingCallback(TrainerCallback):
    """
    Callback for logging metrics to tracking systems (WandB, MLflow).

    This callback handles the actual logging to external tracking systems.
    It's always enabled by default.

    Args:
        log_every_step: Whether to log after every step. Default True.
    """

    def __init__(self, log_every_step: bool = True):
        self.log_every_step = log_every_step

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_every_step:
            control.should_log = True
        return control


class PreflightGateError(Exception):
    """Raised when the pre-flight reward gate fails with on_failure='abort'."""


@register_callback("preflight_gate")
class PreflightGateCallback(TrainerCallback):
    """Pre-flight reward gate: abort training when the reward distribution is
    outside the band where RLOO has usable within-group variance.

    DEFAULT-OFF. When enabled, checks mean per-sample reward after the first
    training step's rollouts are scored (before the second step).  On failure
    with ``on_failure="abort"`` raises ``PreflightGateError``; with
    ``on_failure="warn"`` logs and continues.
    """

    def __init__(
        self,
        enabled: bool = False,
        min_reward: float = 0.25,
        max_reward: float = 0.75,
        on_failure: str = "abort",
        num_trials: int = 256,
    ):
        self.enabled = enabled
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.on_failure = on_failure
        self.num_trials = num_trials
        self._checked = False

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self._checked or not self.enabled:
            return control
        self._checked = True

        trainer = kwargs.get("trainer")
        if trainer is None:
            return control

        rewards = self._extract_step_rewards(trainer)
        if not rewards:
            # An enabled gate that quietly does nothing is the failure this whole
            # mechanism exists to prevent, so say so at error level rather than
            # letting the run proceed looking gated.
            logger.error(
                "[preflight] Gate is ENABLED but step 1 exposed no per-sample rewards, "
                "so nothing was checked and this run is UNGATED. Expected the trainer to "
                "set _current_step_rewards during reward post-processing."
            )
            return control

        if len(rewards) < self.num_trials:
            # Not a failure: the sample count is set by train_batch_size x
            # n_samples_per_prompt, not by the dataset. Report it so a verdict drawn
            # from a thin sample is not read as a firm one.
            logger.warning(
                f"[preflight] Checking {len(rewards)} samples, fewer than the requested "
                f"num_trials={self.num_trials}; the estimate is correspondingly noisier."
            )

        from skyrl_train.utils.preflight_gate import check_preflight_gate

        result = check_preflight_gate(rewards, self.min_reward, self.max_reward)
        if not result.passed and self.on_failure == "abort":
            raise PreflightGateError(result.message)

        return control

    @staticmethod
    def _extract_step_rewards(trainer) -> List[float]:
        """Extract per-sample scalar rewards from the trainer's current batch."""
        for attr in ("_current_step_rewards", "step_rewards"):
            val = getattr(trainer, attr, None)
            if val is not None:
                return [float(r) for r in val]
        return []


@register_callback("inference_stats")
class InferenceStatsCallback(TrainerCallback):
    """Collect one canonical inference snapshot and fan it out to independent sinks."""

    def __init__(
        self,
        log_every_steps: int = 1,
        log_to_console: bool = True,
        log_to_tracker: bool = True,
        console_log_level: str = "info",
        poll_interval_seconds: float = 5.0,
        sinks: tuple[InferenceMetricsSink, ...] | None = None,
    ):
        self.log_every_steps = log_every_steps
        self.log_to_console = log_to_console
        self.log_to_tracker = log_to_tracker
        self.console_log_level = console_log_level.lower()
        self.poll_interval_seconds = poll_interval_seconds
        self._sinks = sinks
        self._inference_engine_client = None
        self._poll_task: asyncio.Task | None = None
        self._read_lock = asyncio.Lock()
        self._step = 0

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        trainer = kwargs.get("trainer")
        if trainer is not None:
            self._inference_engine_client = getattr(trainer, "inference_engine_client", None)
            if self._inference_engine_client is None:
                logger.warning(
                    "InferenceStatsCallback: No inference_engine_client found on trainer. Stats collection will be disabled."
                )
        if self._sinks is None:
            self._sinks = configured_inference_sinks()
        return control

    async def on_train_begin_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        self.on_train_begin(state, control, **kwargs)
        self._step = state.global_step
        if self._inference_engine_client is not None and self._sinks:
            try:
                async with self._read_lock:
                    snapshot = await self._inference_engine_client.get_stats(read_mode=IntervalReadMode.PEEK)
                self._publish_sinks(snapshot, self._step)
            except Exception:
                logger.warning("InferenceStatsCallback: initial collection failed", exc_info=True)
            if self.poll_interval_seconds > 0:
                self._poll_task = asyncio.create_task(self._poll(), name="vllm-stats-poll")
        return control

    async def on_train_end_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
        return control

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval_seconds)
            try:
                async with self._read_lock:
                    snapshot = await self._inference_engine_client.get_stats(read_mode=IntervalReadMode.PEEK)
                self._publish_sinks(snapshot, self._step)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("InferenceStatsCallback: periodic collection failed", exc_info=True)

    async def on_step_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_every_steps <= 0 or state.global_step % self.log_every_steps != 0:
            return control
        if self._inference_engine_client is None:
            return control

        self._step = state.global_step
        try:
            async with self._read_lock:
                snapshot = await self._inference_engine_client.get_stats(read_mode=IntervalReadMode.RESET)
        except Exception:
            logger.warning("InferenceStatsCallback: Failed to collect stats", exc_info=True)
            return control

        metrics = trainer_metrics(snapshot)
        if not metrics:
            return control
        if self.log_to_tracker:
            kwargs["trainer"].all_metrics.update(metrics)
        if self.log_to_console and VLLM_NUM_ENGINES_METRIC in metrics:
            log = logger.debug if self.console_log_level == "debug" else logger.info
            log(format_console_summary(metrics, state.global_step))
        self._publish_sinks(snapshot, state.global_step)
        return control

    def _publish_sinks(self, snapshot, step: int) -> None:
        for sink in self._sinks or ():
            try:
                sink.publish(snapshot, step)
            except Exception:
                logger.warning(f"InferenceStatsCallback: {type(sink).__name__} failed", exc_info=True)


def _add_runtime_callbacks(cfg: DictConfig, callbacks: List[TrainerCallback]) -> None:
    budget = cfg.trainer.get("rollout_token_budget")
    if budget is not None:
        callbacks.append(RolloutTokenBudgetCallback(budget, tuple(cfg.trainer.get("rollout_token_eval_milestones", ()))))
    interval = getattr(cfg.generator, "inference_stats_interval", 1)
    if (
        cfg.generator.backend == "vllm"
        and interval > 0
        and not any(isinstance(callback, InferenceStatsCallback) for callback in callbacks)
    ):
        callbacks.append(InferenceStatsCallback(log_every_steps=interval))


def create_default_callbacks(
    cfg: DictConfig, *, evaluation_profiles: Mapping[str, EvaluationProfile] | None = None
) -> List[TrainerCallback]:
    """
    Create the default set of callbacks based on trainer configuration.

    Supports two configuration styles:

    1. **New style** (explicit callbacks list):
       ```yaml
       trainer:
         callbacks:
           - type: checkpoint
             save_steps: 10
           - type: evaluation
             eval_steps: 20
       ```

    2. **Legacy style** (interval configs):
       ```yaml
       trainer:
         ckpt_interval: 10
         eval_interval: 20
       ```

    If explicit 'callbacks' config is present, it takes precedence.
    Otherwise, callbacks are created from legacy interval configs.

    Args:
        cfg: Training configuration (OmegaConf DictConfig)

    Returns:
        List of configured callbacks
    """
    # Check for new-style explicit callback configuration
    if has_explicit_callbacks(cfg):
        logger.info("Using explicit callback configuration from YAML")
        callbacks = create_callbacks_from_config(cfg, evaluation_profiles=evaluation_profiles)
        token_budget = _configured_distillation_token_budget(cfg)
        configured_budget_callbacks = [
            callback for callback in callbacks if isinstance(callback, DistillationTokenBudgetCallback)
        ]
        if len(configured_budget_callbacks) > 1:
            raise ValueError("explicit callback configuration contains multiple distillation token budgets")
        if token_budget is not None:
            if configured_budget_callbacks and any(
                callback.token_budget != token_budget for callback in configured_budget_callbacks
            ):
                raise ValueError("explicit distillation token budget callback disagrees with trainer configuration")
            if not configured_budget_callbacks:
                callbacks.append(DistillationTokenBudgetCallback(token_budget))
        # Always add logging callback if not explicitly configured
        has_logging = any(isinstance(cb, LoggingCallback) for cb in callbacks)
        if not has_logging:
            callbacks.append(LoggingCallback())
        if cfg.trainer.get("mismatch_probe", {}).get("enabled", False):
            callbacks.append(MismatchProbeCallback(cfg))
        _add_runtime_callbacks(cfg, callbacks)
        return callbacks

    # Fall back to legacy interval-based configuration
    logger.debug("Using legacy interval-based callback configuration")
    callbacks = []

    # Checkpoint callback
    ckpt_interval = getattr(cfg.trainer, "ckpt_interval", 10)
    if ckpt_interval > 0:
        callbacks.append(CheckpointCallback(save_steps=ckpt_interval))

    # Evaluation callback
    eval_interval = getattr(cfg.trainer, "eval_interval", 5)
    eval_before_train = getattr(cfg.trainer, "eval_before_train", True)
    if eval_interval > 0:
        callbacks.append(
            EvaluationCallback(
                eval_steps=eval_interval,
                eval_before_train=eval_before_train,
            )
        )

    # HF model save callback
    if hf_export_destination_configured(cfg):
        callbacks.append(HFModelSaveCallback(save_steps=int(cfg.trainer.hf_save_interval)))

    # Reference model update callback
    update_ref_every_epoch = getattr(cfg.trainer, "update_ref_every_epoch", False)
    if update_ref_every_epoch:
        callbacks.append(RefModelUpdateCallback(update_every_epoch=True))

    # Database registration callback (auto-enabled, gracefully disabled if no credentials)
    enable_db_registration = getattr(cfg.trainer, "enable_db_registration", True)
    if enable_db_registration:
        callbacks.append(DatabaseRegistrationCallback())

    # Pre-flight reward gate (default-off)
    gate_cfg = getattr(cfg.trainer, "preflight_gate", None)
    gate_enabled = getattr(gate_cfg, "enabled", False) if gate_cfg else False
    if gate_enabled:
        callbacks.append(
            PreflightGateCallback(
                enabled=True,
                min_reward=getattr(gate_cfg, "min_reward", 0.25),
                max_reward=getattr(gate_cfg, "max_reward", 0.75),
                on_failure=getattr(gate_cfg, "on_failure", "abort"),
                num_trials=getattr(gate_cfg, "num_trials", 256),
            )
        )

    _add_runtime_callbacks(cfg, callbacks)

    token_budget = _configured_distillation_token_budget(cfg)
    if token_budget is not None:
        callbacks.append(DistillationTokenBudgetCallback(token_budget))

    # Logging callback (always enabled)
    callbacks.append(LoggingCallback())

    if cfg.trainer.get("mismatch_probe", {}).get("enabled", False):
        callbacks.append(MismatchProbeCallback(cfg))
    return callbacks


class DefaultCallbackHandler(CallbackHandler):
    """
    A callback handler that initializes with default callbacks based on config.

    This provides backward compatibility by recreating the original inline
    behavior through callbacks.

    Example:
        ```python
        handler = DefaultCallbackHandler(cfg)
        # Adds all default callbacks based on config intervals
        ```
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """
        Initialize with default callbacks from config, plus any custom callbacks.

        Args:
            cfg: Training configuration. If provided, creates default callbacks.
            callbacks: Additional custom callbacks to add after defaults.
        """
        default_callbacks = []
        if cfg is not None:
            default_callbacks = create_default_callbacks(cfg)

        all_callbacks = default_callbacks + (callbacks or [])
        super().__init__(all_callbacks)

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        additional_callbacks: Optional[List[TrainerCallback]] = None,
    ) -> "DefaultCallbackHandler":
        """
        Create a handler from config with optional additional callbacks.

        Args:
            cfg: Training configuration
            additional_callbacks: Custom callbacks to add after defaults

        Returns:
            Configured callback handler
        """
        return cls(cfg=cfg, callbacks=additional_callbacks)


def create_callback_from_config(callback_config: Dict[str, Any]) -> TrainerCallback:
    """
    Create a callback instance from a YAML config dictionary.

    Args:
        callback_config: Dictionary with 'type' key and callback-specific params.
            Example: {"type": "checkpoint", "save_steps": 10}

    Returns:
        Instantiated callback

    Raises:
        ValueError: If callback type is unknown or missing
    """
    if "type" not in callback_config:
        raise ValueError(f"Callback config missing 'type' key: {callback_config}")

    callback_type = callback_config["type"]
    if callback_type not in CALLBACK_REGISTRY:
        available = ", ".join(CALLBACK_REGISTRY.keys())
        raise ValueError(f"Unknown callback type '{callback_type}'. Available types: {available}")

    # Get the callback class and instantiate with remaining params
    callback_cls = CALLBACK_REGISTRY[callback_type]
    params = {k: v for k, v in callback_config.items() if k != "type"}

    try:
        return callback_cls(**params)
    except TypeError as e:
        raise ValueError(f"Invalid parameters for callback '{callback_type}': {e}") from e


def create_callbacks_from_config(
    cfg: DictConfig, *, evaluation_profiles: Mapping[str, EvaluationProfile] | None = None
) -> List[TrainerCallback]:
    """
    Create callbacks from explicit YAML configuration.

    This supports the new-style callback configuration:
    ```yaml
    trainer:
      callbacks:
        - type: checkpoint
          save_steps: 10
        - type: evaluation
          eval_steps: 20
          eval_before_train: false
    ```

    Args:
        cfg: Training configuration with optional 'callbacks' list

    Returns:
        List of instantiated callbacks (empty if no callbacks configured)
    """
    callbacks_config = getattr(cfg.trainer, "callbacks", None)
    if callbacks_config is None:
        return []

    profiles = resolve_evaluation_profiles(cfg) if evaluation_profiles is None else evaluation_profiles
    validate_evaluation_schedule(cfg, profiles)
    callbacks = []
    for callback_config in callbacks_config:
        # Convert OmegaConf to dict if needed
        if hasattr(callback_config, "items"):
            config_dict = dict(callback_config)
        else:
            config_dict = callback_config

        try:
            if config_dict.get("type") == "evaluation":
                config_dict["additional_evaluations"] = {
                    name: profiles[name] for name in config_dict.get("additional_evaluations") or {}
                }
                if (score_pool := config_dict.get("checkpoint_scores")) is not None:
                    if OmegaConf.is_config(score_pool):
                        score_pool = OmegaConf.to_container(score_pool, resolve=True)
                    config_dict["checkpoint_scores"] = PooledCheckpointEvaluation.model_validate(score_pool)
            callback = create_callback_from_config(config_dict)
            callbacks.append(callback)
            logger.debug(f"Created callback: {callback.__class__.__name__}")
        except ValueError as e:
            logger.error(f"Failed to create callback: {e}")
            raise

    return callbacks
