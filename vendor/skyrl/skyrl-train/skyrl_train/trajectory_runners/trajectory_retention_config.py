from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from skyrl_train.trajectory_runners.trajectory_reward_shaping import DEFAULT_ACCEPTED_STOP_REASONS


@dataclass(frozen=True)
class TrajectoryRetentionConfig:
    enabled: bool = True
    output_path: str = ""
    run_id: str = ""
    phases: tuple[str, ...] = ("train",)
    sample_count_per_step: int = 1
    sample_fraction: float = 0.0
    always_retain_failures: bool = True
    always_retain_non_terminating: bool = True
    always_retain_loops: bool = True
    accepted_stop_reasons: tuple[str, ...] = DEFAULT_ACCEPTED_STOP_REASONS
    reward_below: float | None = None
    reward_above: float | None = None
    max_bytes_per_step: int | None = 800 * 1024 * 1024
    max_bytes_per_run: int | None = 25 * 1024 * 1024 * 1024
    required: bool = False
    publish_timeout_seconds: float = 120.0
    shutdown_timeout_seconds: float = 30.0
    redact_fields: tuple[str, ...] = ()
    model_path: str | None = None
    model_source_identity: str | None = None
    resume_path: str | None = None
    inference_backend: str | None = None


def parse_trajectory_retention_config(config: Mapping[str, Any] | None) -> TrajectoryRetentionConfig:
    """Parse and validate the shared trajectory-retention policy."""
    defaults = TrajectoryRetentionConfig()
    if config is None:
        return TrajectoryRetentionConfig(enabled=False)
    if not isinstance(config, Mapping):
        raise ValueError("generator.trajectory_retention must be a mapping")

    phases_value = config.get("phases", defaults.phases)
    redact_value = config.get("redact_fields", defaults.redact_fields)
    accepted_stops_value = config.get("accepted_stop_reasons", defaults.accepted_stop_reasons)
    if not isinstance(phases_value, Sequence) or isinstance(phases_value, (str, bytes)):
        raise ValueError("trajectory_retention.phases must be a sequence")
    if not isinstance(redact_value, Sequence) or isinstance(redact_value, (str, bytes)):
        raise ValueError("trajectory_retention.redact_fields must be a sequence")
    if not isinstance(accepted_stops_value, Sequence) or isinstance(accepted_stops_value, (str, bytes)):
        raise ValueError("trajectory_retention.accepted_stop_reasons must be a sequence")

    parsed = TrajectoryRetentionConfig(
        enabled=bool(config.get("enabled", defaults.enabled)),
        output_path=str(config.get("output_path", defaults.output_path) or ""),
        run_id=str(config.get("run_id", defaults.run_id) or ""),
        phases=tuple(str(phase).lower() for phase in phases_value),
        sample_count_per_step=int(config.get("sample_count_per_step", defaults.sample_count_per_step)),
        sample_fraction=float(config.get("sample_fraction", defaults.sample_fraction)),
        always_retain_failures=bool(config.get("always_retain_failures", defaults.always_retain_failures)),
        always_retain_non_terminating=bool(
            config.get("always_retain_non_terminating", defaults.always_retain_non_terminating)
        ),
        always_retain_loops=bool(config.get("always_retain_loops", defaults.always_retain_loops)),
        accepted_stop_reasons=tuple(str(reason).strip().lower() for reason in accepted_stops_value),
        reward_below=_optional_float(config.get("reward_below", defaults.reward_below)),
        reward_above=_optional_float(config.get("reward_above", defaults.reward_above)),
        max_bytes_per_step=_optional_int(config.get("max_bytes_per_step", defaults.max_bytes_per_step)),
        max_bytes_per_run=_optional_int(config.get("max_bytes_per_run", defaults.max_bytes_per_run)),
        required=bool(config.get("required", defaults.required)),
        publish_timeout_seconds=float(config.get("publish_timeout_seconds", defaults.publish_timeout_seconds)),
        shutdown_timeout_seconds=float(config.get("shutdown_timeout_seconds", defaults.shutdown_timeout_seconds)),
        redact_fields=tuple(str(field) for field in redact_value),
        model_path=_optional_string(config.get("model_path", defaults.model_path)),
        model_source_identity=_optional_string(config.get("model_source_identity", defaults.model_source_identity)),
        resume_path=_optional_string(config.get("resume_path", defaults.resume_path)),
        inference_backend=_optional_string(config.get("inference_backend", defaults.inference_backend)),
    )
    _validate_config(parsed)
    return parsed


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _validate_config(config: TrajectoryRetentionConfig) -> None:
    if config.enabled and (not config.output_path or not config.run_id):
        raise ValueError("enabled trajectory retention requires output_path and run_id")
    if not set(config.phases).issubset({"train", "eval"}) or not config.phases:
        raise ValueError("trajectory_retention.phases must contain train and/or eval")
    if config.sample_count_per_step < 0:
        raise ValueError("trajectory_retention.sample_count_per_step must be non-negative")
    if not 0.0 <= config.sample_fraction <= 1.0:
        raise ValueError("trajectory_retention.sample_fraction must be between 0 and 1")
    byte_bounds = (config.max_bytes_per_step, config.max_bytes_per_run)
    if any(bound is not None and bound < 0 for bound in byte_bounds):
        raise ValueError("trajectory retention byte bounds must be non-negative")
    if (
        config.max_bytes_per_step is not None
        and config.max_bytes_per_run is not None
        and config.max_bytes_per_step > config.max_bytes_per_run
    ):
        raise ValueError("trajectory retention per-step bound cannot exceed its run bound")
    if config.publish_timeout_seconds <= 0 or config.shutdown_timeout_seconds <= 0:
        raise ValueError("trajectory retention publication timeouts must be positive")
    if not config.accepted_stop_reasons:
        raise ValueError("trajectory_retention.accepted_stop_reasons cannot be empty")
