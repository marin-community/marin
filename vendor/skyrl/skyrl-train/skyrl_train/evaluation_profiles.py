"""Resolve trainer.callbacks[*].additional_evaluations into recorded native requests.

Named results use eval/<name>/ metrics and <trainer.export_path>/<name>/ outputs.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args

from marinskyrl.resource_locator import join_resource_path
from omegaconf import DictConfig, OmegaConf


EvaluationSchedule = Literal["every_evaluation", "final"]


@dataclass(frozen=True)
class EvaluationProfile:
    sampling_params: dict[str, Any] | None = None
    n_samples_per_prompt: int | None = None
    data_files: list[str] | None = None
    schedule: EvaluationSchedule = "every_evaluation"

    def __post_init__(self):
        if self.schedule not in get_args(EvaluationSchedule):
            raise ValueError(f"unknown evaluation schedule: {self.schedule}")
        if self.n_samples_per_prompt is not None and (
            type(self.n_samples_per_prompt) is not int or self.n_samples_per_prompt <= 0
        ):
            raise ValueError("n_samples_per_prompt must be a positive integer")
        if self.data_files is not None and not self.data_files:
            raise ValueError("data_files must be nonempty when provided")


@cache
def _sampling_fields() -> frozenset[str]:
    config = OmegaConf.load(Path(__file__).parent / "config" / "ppo_base_config.yaml")
    # Per-request seeds and minimum lengths are consumed by the native sampling adapters.
    return frozenset(config.generator.eval_sampling_params) | {"seed", "min_tokens"}


def _declared_profiles(config: Mapping[str, Any]) -> dict[str, EvaluationProfile]:
    profiles = {}
    allowed = {field.name for field in fields(EvaluationProfile)}
    for callback in config.get("trainer", {}).get("callbacks") or []:
        if callback.get("type") != "evaluation":
            continue
        for name, parameters in (callback.get("additional_evaluations") or {}).items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"evaluation name must be an identifier: {name!r}")
            if name in profiles:
                raise ValueError(f"duplicate evaluation profile: {name}")
            unknown = set(parameters) - allowed
            if unknown:
                raise ValueError(f"evaluation profile {name} has unknown fields: {sorted(unknown)}")
            try:
                profile = EvaluationProfile(**parameters)
            except (TypeError, ValueError) as error:
                raise ValueError(f"evaluation profile {name}: {error}") from error
            unknown_sampling = set(profile.sampling_params or {}) - _sampling_fields()
            if unknown_sampling:
                raise ValueError(f"evaluation profile {name} has unknown sampling keys: {sorted(unknown_sampling)}")
            profiles[name] = profile
    return profiles


def validate_evaluation_data(config: DictConfig, *, source_recipe: bool) -> None:
    """Validate named launch-data bindings against their declared evaluation profiles."""
    profiles = _declared_profiles(config.skyrl)
    for name in config.inputs.get("evaluation_data", {}):
        if name not in profiles:
            raise ValueError(f"evaluation data has no additional evaluation profile: {name}")
        if source_recipe and profiles[name].data_files is not None:
            raise ValueError(f"evaluation profile {name} has both data_files and evaluation_data")


def evaluation_profiles(
    config: DictConfig, *, evaluation_data: tuple[tuple[str, tuple[str, ...]], ...] = ()
) -> dict[str, EvaluationProfile]:
    """Resolve and record named requests in one complete native SkyRL configuration."""
    profiles = _declared_profiles(config)
    data = dict(evaluation_data)
    resolved = {}
    for name, profile in profiles.items():
        if name in data:
            profile = EvaluationProfile(**{**asdict(profile), "data_files": list(data[name])})
        merged = evaluation_profile_config(config, profile, name)
        resolved[name] = EvaluationProfile(
            sampling_params=OmegaConf.to_container(merged.generator.eval_sampling_params, resolve=True),
            n_samples_per_prompt=int(merged.generator.eval_n_samples_per_prompt),
            data_files=profile.data_files,
            schedule=profile.schedule,
        )
    for callback in config.get("trainer", {}).get("callbacks") or []:
        if callback.get("type") == "evaluation":
            for name in callback.get("additional_evaluations") or {}:
                callback.additional_evaluations[name] = asdict(resolved[name])
    return resolved


def evaluation_profile_due(profile: EvaluationProfile, *, step: int, last_step: bool) -> bool:
    """Whether a named request runs at this evaluation event."""
    return profile.schedule == "every_evaluation" or last_step and step > 0


def named_evaluation_metrics(metrics: Mapping[str, float], name: str) -> dict[str, float]:
    return {f"eval/{name}/{key.removeprefix('eval/')}": value for key, value in metrics.items()}


def evaluation_profile_config(cfg: DictConfig, profile: EvaluationProfile, name: str | None = None) -> DictConfig:
    overrides = {"generator": {}, "trainer": {}}
    if profile.n_samples_per_prompt is not None:
        overrides["generator"]["eval_n_samples_per_prompt"] = profile.n_samples_per_prompt
    if name is not None:
        overrides["trainer"]["export_path"] = join_resource_path(cfg.trainer.export_path, name)
    if profile.data_files is not None:
        overrides["data"] = {"val_data": profile.data_files}
    merged = OmegaConf.merge(cfg, overrides)
    for key, value in (profile.sampling_params or {}).items():
        OmegaConf.update(merged, f"generator.eval_sampling_params.{key}", value, force_add=True)
    return merged
