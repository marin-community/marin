"""Resolve trainer.callbacks[*].additional_evaluations into recorded native configuration.

Named results use eval/<name>/ metrics and <trainer.export_path>/<name>/ outputs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any, Protocol

from marinskyrl.resource_locator import join_resource_path
from omegaconf import DictConfig, OmegaConf
from skyrlrecipe.declared import EvaluationSampling as EvaluationProfile


DEFAULT_EVALUATION_STEPS = 5


def _declared_profiles(config: Mapping[str, Any]) -> dict[str, EvaluationProfile]:
    profiles = {}
    callbacks = config.get("trainer", {}).get("callbacks")
    if callbacks is None:
        callbacks = ()
    if not isinstance(callbacks, Sequence) or isinstance(callbacks, str):
        raise ValueError("trainer.callbacks must be a sequence")
    for index, callback in enumerate(callbacks):
        if not isinstance(callback, Mapping):
            raise ValueError(f"trainer.callbacks[{index}] must be a mapping")
        if callback.get("type") != "evaluation":
            continue
        declarations = callback.get("additional_evaluations")
        if declarations is None:
            continue
        if not isinstance(declarations, Mapping):
            raise ValueError("evaluation additional_evaluations must be a mapping")
        for name, parameters in declarations.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"evaluation name must be an identifier: {name!r}")
            if name in profiles:
                raise ValueError(f"duplicate evaluation profile: {name}")
            try:
                document = (
                    OmegaConf.to_container(parameters, resolve=True) if OmegaConf.is_config(parameters) else parameters
                )
                profile = EvaluationProfile.model_validate_json(json.dumps(document))
            except (TypeError, ValueError) as error:
                raise ValueError(f"evaluation profile {name}: {error}") from error
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
    for name, paths in evaluation_data:
        profiles[name] = EvaluationProfile.model_validate_json(
            json.dumps(profiles[name].to_skyrl() | {"data_files": paths})
        )
    resolved = {}
    for name, profile in profiles.items():
        merged = evaluation_profile_config(config, profile, name)
        resolved[name] = EvaluationProfile.model_validate_json(
            json.dumps(
                {
                    "sampling_params": OmegaConf.to_container(merged.generator.eval_sampling_params, resolve=True),
                    "n_samples_per_prompt": int(merged.generator.eval_n_samples_per_prompt),
                    "data_files": profile.data_files,
                    "schedule": profile.schedule,
                }
            )
        )
    for callback in config.get("trainer", {}).get("callbacks") or []:
        if callback.get("type") == "evaluation":
            for name in callback.get("additional_evaluations") or {}:
                callback.additional_evaluations[name] = resolved[name].to_skyrl()
    return resolved


def evaluation_profile_due(profile: EvaluationProfile, *, step: int, last_step: bool) -> bool:
    """Whether a named request runs at this evaluation event."""
    return profile.schedule == "every_evaluation" or last_step and step > 0


def validate_generation_profiles(profiles: Mapping[str, EvaluationProfile]) -> None:
    if any(not evaluation_profile_due(profile, step=0, last_step=False) for profile in profiles.values()):
        raise ValueError("final evaluation profiles require training updates")


class PeriodicEvaluation(Protocol):
    @property
    def eval_steps(self) -> int: ...
    @property
    def additional_evaluations(self) -> Mapping[str, EvaluationProfile]: ...
    @property
    def metric_groups(self) -> Mapping[str, Sequence[str]]: ...
    @property
    def stop_when(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class PeriodicEvaluationConfig:
    eval_steps: int = DEFAULT_EVALUATION_STEPS
    additional_evaluations: Mapping[str, EvaluationProfile] = field(default_factory=dict)
    metric_groups: Mapping[str, Sequence[str]] = field(default_factory=dict)
    stop_when: Mapping[str, object] = field(default_factory=dict)


def validate_evaluation_schedule(config: DictConfig, *, generation: bool) -> None:
    profiles = _declared_profiles(config)
    if generation:
        validate_generation_profiles(profiles)
    evaluations = [
        PeriodicEvaluationConfig(
            eval_steps=callback.get("eval_steps", DEFAULT_EVALUATION_STEPS),
            additional_evaluations={name: profiles[name] for name in callback.get("additional_evaluations") or {}},
            metric_groups=callback.get("metric_groups") or {},
            stop_when=callback.get("stop_when") or {},
        )
        for callback in config.trainer.get("callbacks") or ()
        if callback.get("type") == "evaluation"
    ]
    validate_periodic_evaluation_inputs(evaluations)


def validate_periodic_evaluation_inputs(evaluations: Sequence[PeriodicEvaluation]) -> None:
    if any(callback.eval_steps > 0 for callback in evaluations):
        final_prefixes = tuple(
            f"eval/{name}/"
            for callback in evaluations
            for name, profile in callback.additional_evaluations.items()
            if profile.schedule == "final"
        )
        for callback in evaluations:
            inputs = set(callback.stop_when)
            inputs.update(key for members in callback.metric_groups.values() for key in members)
            final_inputs = sorted(key for key in inputs if key.startswith(final_prefixes))
            if final_inputs:
                raise ValueError(
                    "final-only metrics in metric_groups or stop_when require eval_steps=0 on all evaluation callbacks "
                    f"({', '.join(final_inputs)})"
                )


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
    for key, value in (profile.sampling_params.to_skyrl() if profile.sampling_params is not None else {}).items():
        OmegaConf.update(merged, f"generator.eval_sampling_params.{key}", value, force_add=True)
    return merged
