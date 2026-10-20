"""Apply Harbor launch settings to tasks executed by the common rollout engine."""

from dataclasses import dataclass, replace
from functools import partial
import os

from daytona import AsyncDaytona, DaytonaConfig
from omegaconf import DictConfig
from shellbox.backends.daytona.machine import DaytonaMachineFactory, DaytonaNetworkMode, DaytonaNetworkPolicy
from shellbox.machine import Backend, MachineFactory
from taskcompendium.models import NoGrader
from rolloutengine.models import MachineRuntimeSpec, RolloutFailure, TaskRolloutSpec
from taskcompendium.grading_result import GradingFailure, Outcome
from skyrl_train.rollouts.training_record import TrainingRollout

from skyrl_train.rollouts.harbor_settings import (
    HARBOR_EXCEPTION,
    EnvironmentConfig,
    EnvironmentType,
    RetryConfig,
    VerifierConfig,
)

from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder
from skyrl_train.trajectory_runners.harbor.identity_aware_reward import (
    IDENTITY_AWARE_SHAPER,
    identity_aware_pass_ratios,
)
from skyrl_train.trajectory_runners.harbor.truncation_penalty import apply_truncation_penalty, detect_turn_truncation
from skyrl_train.trajectory_runners.types import RESPONSE_SPAN_TAGS_KEY, TrajectoryRequestBatch
from skyrl_train.metric_names import IDENTITY_AWARE_REWARD_METRIC_PREFIX
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig, ErrorTreatment, resolve_error_treatment
from skyrl_train.utils.pbs_shaping import compute_pbs_token_shaping
from skyrl_train.utils.span_tagger import tag_generated_tokens
from skyrl_train.utils.reward_shaping import (
    RewardOutputParseError,
    parse_test_output_with_parser,
    shape_reward_from_output,
    shape_reward_with_components,
    verifier_test_collection,
)

HARBOR_MACHINE_BACKEND = "harbor"
HARBOR_EVAL_MACHINE_BACKEND = "harbor_eval"


@dataclass(frozen=True)
class HarborTaskSettings:
    environment: EnvironmentConfig
    verifier: VerifierConfig
    agent_timeout: float | None
    max_agent_timeout: float | None
    max_turns: int | None
    timeout_multiplier: float
    eval_timeout: float | None
    concurrent_trials: int
    error_handling: ErrorHandlingConfig
    reward_shaping: dict
    retry: RetryConfig
    attempt_timeout: float | None

    @classmethod
    def from_config(cls, config: DictConfig):
        builder = HarborConfigBuilder(config)
        agent, options = builder.agent_fields()
        trial = builder.trial_fields()
        return cls(
            builder.environment_config(),
            builder.verifier_config(),
            agent.get("override_timeout_sec"),
            agent.get("max_timeout_sec"),
            options.get("max_turns"),
            float(trial.get("timeout_multiplier", 1)),
            builder.get_eval_timeout_override_sec(),
            builder.get_n_concurrent_trials(),
            builder.get_error_handling_config(),
            builder.get_reward_shaping_config(),
            builder.build_retry_config(),
            trial.get("trial_attempt_timeout_sec"),
        )

    def retry_delay(self, failure: RolloutFailure | None, retries: int) -> float | None:
        """Return the backoff delay, or keep a terminal result without another attempt."""
        if failure is None or retries >= self.retry.max_retries:
            return None
        name = failure.exception_type
        if name in (self.retry.exclude_exceptions or ()):
            return None
        if self.retry.include_exceptions and name not in self.retry.include_exceptions:
            return None
        if resolve_error_treatment(name, self.error_handling) is not ErrorTreatment.MASK:
            return None
        return min(self.retry.min_wait_sec * self.retry.wait_multiplier**retries, self.retry.max_wait_sec)

    def grader_override(self) -> NoGrader | None:
        return NoGrader(reason="Harbor verification is disabled") if self.verifier.disable else None

    def machine_factory(self, *, phase: str = "train") -> MachineFactory:
        match self.environment.type:
            case EnvironmentType.DAYTONA:
                policy = self.environment.kwargs.get("network_policy")
                key_env = self.environment.kwargs.get("eval_api_key_env" if phase == "eval" else "api_key_env")
                client_factory = AsyncDaytona
                if key_env is not None:
                    api_key = os.environ.get(key_env)
                    if not api_key:
                        raise ValueError(f"Missing Daytona credential environment variable: {key_env}")
                    client_factory = partial(AsyncDaytona, DaytonaConfig(api_key=api_key))
                return DaytonaMachineFactory(
                    client_factory=client_factory,
                    ttl_minutes=self.environment.kwargs.get("ttl_minutes", 360),
                    network_policy=None
                    if policy is None
                    else DaytonaNetworkPolicy(
                        mode=DaytonaNetworkMode(policy["mode"]),
                        value=policy.get("value"),
                    ),
                )
            case other:
                raise ValueError(f"No Shellbox factory is configured for Harbor backend {other!r}")

    def configured_spec(self, spec: TaskRolloutSpec, *, phase: str) -> TaskRolloutSpec:
        """Apply deployment overrides without changing the task definition."""
        if self.verifier.disable and not isinstance(spec.task.grader, NoGrader):
            raise ValueError("Disabled Harbor verification must be selected before task import")

        def machine(original: MachineRuntimeSpec | None) -> MachineRuntimeSpec | None:
            if original is None:
                return None
            updates = {
                name: value
                for name in ("cpus", "memory_mb", "storage_mb", "gpus")
                if (value := getattr(self.environment, f"override_{name}")) is not None
            }
            if original.backend == Backend.DOCKER:
                updates["backend"] = (
                    HARBOR_EVAL_MACHINE_BACKEND
                    if phase == "eval" and self.environment.kwargs.get("eval_api_key_env")
                    else HARBOR_MACHINE_BACKEND
                )
            if original.startup_timeout is not None:
                updates["startup_timeout"] = original.startup_timeout * self.timeout_multiplier
            return original.model_copy(update=updates)

        def timeout(original: float | None, override: float | None, ceiling: float | None) -> float | None:
            value = original if override is None else override
            if value is None:
                return None
            value *= self.timeout_multiplier
            return value if ceiling is None else min(value, ceiling)

        limits = spec.limits.model_copy(
            update={
                "max_turns": spec.limits.max_turns if self.max_turns is None else self.max_turns,
                "total_turn_timeout": timeout(
                    spec.limits.total_turn_timeout,
                    self.eval_timeout if phase == "eval" else self.agent_timeout,
                    self.max_agent_timeout,
                ),
                "verifier_timeout": timeout(
                    spec.limits.verifier_timeout,
                    self.verifier.override_timeout_sec,
                    self.verifier.max_timeout_sec,
                ),
                "attempt_timeout": spec.limits.attempt_timeout
                if self.attempt_timeout is None
                else self.attempt_timeout,
            }
        )
        return spec.model_copy(
            update={
                "task_machine": machine(spec.task_machine),
                "verifier_machine": machine(spec.verifier_machine),
                "limits": limits,
            }
        )


_GRADING_FAILURE_TYPES = {
    GradingFailure.TIMEOUT: "VerifierTimeoutError",
    GradingFailure.MISSING_REWARD: "RewardFileNotFoundError",
    GradingFailure.EMPTY_REWARD: "RewardFileEmptyError",
    GradingFailure.INVALID_REWARD: "VerifierOutputParseError",
    GradingFailure.EXECUTION: "VerifierRuntimeError",
}


def harbor_grading_failure(rollout: TrainingRollout) -> TrainingRollout:
    """Translate structured verifier failures to the configured Harbor error policy."""
    if rollout.failure is not None:
        return rollout
    exception_type = rollout.grade.diagnostics.get(HARBOR_EXCEPTION)
    if exception_type is not None:
        return replace(
            rollout,
            failure=RolloutFailure(exception_type, diagnostics={"timed_out": exception_type == "AgentTimeoutError"}),
        )
    if rollout.grade.status != Outcome.INFRA_ERROR:
        return rollout
    return replace(
        rollout,
        failure=RolloutFailure(_GRADING_FAILURE_TYPES.get(rollout.grade.failure, "VerifierRuntimeError")),
    )


def shape_harbor_rollouts(
    rollouts: list[TrainingRollout],
    request: TrajectoryRequestBatch,
    settings: HarborTaskSettings,
    max_generate_length: int,
    tokenizer,
    *,
    harbor_tasks: list[bool],
) -> list[TrainingRollout]:
    """Keep verifier grades and assign shaped rewards to the last valid action."""
    if not any(harbor_tasks):
        return rollouts
    rollouts = list(rollouts)
    config = settings.reward_shaping
    enabled = config["enable_reward_shaping"]
    shaper = config["reward_shaper"]
    kwargs = config["shaper_kwargs"]
    rewards = [rollout.grade.reward if rollout.grade.status == Outcome.GRADED else 0.0 for rollout in rollouts]
    components = [{} for _ in rollouts]
    collections = [None for _ in rollouts]
    identifiers = request.trajectory_ids
    metadata = request.batch_metadata
    eligible = [
        harbor and rollout.grade.status == Outcome.GRADED and rollout.failure is None
        for harbor, rollout in zip(harbor_tasks, rollouts, strict=True)
    ]
    for index, rollout in enumerate(rollouts):
        if not eligible[index] or not enabled:
            continue
        original = rollout.grade.reward
        assert original is not None
        stdout = rollout.grade.diagnostics.get("stdout", "")
        history = list(rollout.messages)
        try:
            if shaper == IDENTITY_AWARE_SHAPER:
                parsed, parser = parse_test_output_with_parser(stdout, config.get("reward_parser"))
                if parsed is None:
                    if not config["reward_shaping_fallback"]:
                        raise RewardOutputParseError("Cannot parse the verifier output for reward shaping")
                else:
                    if identifiers is None:
                        raise ValueError("Identity-aware reward shaping requires trajectory IDs")
                    identity = identifiers[index]
                    collections[index] = verifier_test_collection(
                        parsed,
                        parser_name=parser,
                        instance_id=identity.instance_id,
                        repetition_id=identity.repetition_id,
                    )
                    rewards[index] = parsed.pass_ratio
            elif shaper in {"composite", "composite_loop"}:
                rewards[index], components[index] = shape_reward_with_components(
                    stdout,
                    original,
                    config.get("reward_parser"),
                    kwargs,
                    history,
                    shaper,
                    trajectory_context={
                        "mark_complete": rollout.stop_reason == "stop",
                        "premature_stop": rollout.stop_reason != "stop",
                        "verifier_reward": original,
                    },
                )
            else:
                rewards[index] = shape_reward_from_output(
                    stdout,
                    original,
                    parser_name=config.get("reward_parser"),
                    shaper_name=shaper,
                    shaper_kwargs=kwargs,
                    fallback_to_original=config["reward_shaping_fallback"],
                    chat_history=history,
                )
        except RewardOutputParseError:
            eligible[index] = False
            rollouts[index] = replace(rollout, failure=RolloutFailure("VerifierOutputParseError"))
    group_metrics = [{} for _ in rollouts]
    if enabled and shaper == IDENTITY_AWARE_SHAPER:
        if identifiers is None:
            raise ValueError("Identity-aware reward shaping requires trajectory IDs")
        groups = {}
        for index, identity in enumerate(identifiers):
            if not harbor_tasks[index]:
                continue
            groups.setdefault(identity.instance_id, []).append(index)
        for indices in groups.values():
            shaped = identity_aware_pass_ratios(
                [collections[index] for index in indices],
                [rewards[index] for index in indices],
                [eligible[index] for index in indices],
                exact_weights=kwargs.get("test_weights"),
            )
            for index, reward in zip(indices, shaped.rewards, strict=True):
                rewards[index] = reward
            group_metrics[indices[0]] = {
                f"{IDENTITY_AWARE_REWARD_METRIC_PREFIX}/groups": 1.0,
                f"{IDENTITY_AWARE_REWARD_METRIC_PREFIX}/informative_tests": float(shaped.informative_test_count),
                f"{IDENTITY_AWARE_REWARD_METRIC_PREFIX}/fallback_groups": float(shaped.fallback_reason is not None),
            }
    result = []
    for index, rollout in enumerate(rollouts):
        if not eligible[index]:
            result.append(rollout)
            continue
        truncated = detect_turn_truncation(
            [len(step.turn.response_token_ids) for step in rollout.steps], max_generate_length
        )
        reward, penalized = apply_truncation_penalty(
            rewards[index], rollout.grade.reward, truncated, config["truncation_penalty"]
        )
        last_step = next(
            (
                step_index
                for step_index in reversed(range(len(rollout.steps)))
                if rollout.loss_mask[rollout.steps[step_index].response_end]
            ),
            None,
        )
        component_rewards = components[index]
        steps = tuple(
            replace(
                step,
                transition=replace(
                    step.transition,
                    reward=reward if step_index == last_step else 0.0,
                    reward_components=component_rewards if step_index == last_step else {},
                ),
            )
            for step_index, step in enumerate(rollout.steps)
        )
        if config.get("enable_token_reward_channel") and (metadata is None or metadata.training_phase != "eval"):
            tags = [0] * len(rollout.response_token_ids)
            if config["enable_span_tagging"]:
                for step in steps:
                    start = step.response_end + 1 - len(step.turn.response_token_ids)
                    values = tag_generated_tokens(list(step.turn.response_token_ids), tokenizer)
                    tags[start : step.response_end + 1] = [
                        value if rollout.loss_mask[start + offset] else 0 for offset, value in enumerate(values)
                    ]
            credit = (
                compute_pbs_token_shaping(
                    list(rollout.messages),
                    tags,
                    gamma=config["pbs_gamma"],
                    max_total_shaping=config["pbs_max_total_shaping"],
                    potential_shape=config["pbs_potential_shape"],
                )
                if config["enable_pbs_shaping"] and config["enable_span_tagging"]
                else [0.0] * len(tags)
            )
            annotated = []
            for step in steps:
                start = step.response_end + 1 - len(step.turn.response_token_ids)
                turn = step.turn
                if config["enable_span_tagging"]:
                    turn = replace(
                        turn, metadata={**turn.metadata, RESPONSE_SPAN_TAGS_KEY: tags[start : step.response_end + 1]}
                    )
                annotated.append(
                    replace(
                        step,
                        turn=turn,
                        transition=replace(
                            step.transition,
                            token_credit=tuple(credit[start : step.response_end + 1]),
                        ),
                    )
                )
            steps = tuple(annotated)
        result.append(
            replace(
                rollout,
                steps=steps,
                grade=replace(
                    rollout.grade, diagnostics={**rollout.grade.diagnostics, "verifier_tests": collections[index]}
                ),
                metrics={**rollout.metrics, **group_metrics[index], "truncation_penalized": float(penalized)},
            )
        )
    return result
