"""Resolve trainer.callbacks[*].additional_evaluations into recorded native configuration.

Named results use eval/<name>/ metrics and <trainer.export_path>/<name>/ outputs.
"""

from collections.abc import Mapping
import json
from typing import Any

from marinskyrl.resource_locator import join_resource_path
from omegaconf import DictConfig, OmegaConf
from skyrlrecipe.declared import EvaluationSampling as EvaluationProfile


DEFAULT_EVALUATION_STEPS = 5


def _declared_profiles(config: Mapping[str, Any]) -> dict[str, Any]:
    profiles = {}
    for callback in config.get("trainer", {}).get("callbacks") or ():
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
            profiles[name] = (
                OmegaConf.to_container(parameters, resolve=True) if OmegaConf.is_config(parameters) else parameters
            )
    return profiles


def validate_evaluation_data(config: DictConfig, *, source_recipe: bool) -> None:
    """Validate named launch-data bindings against their declared evaluation profiles."""
    profiles = _declared_profiles(config.skyrl)
    for name in config.inputs.get("evaluation_data", {}):
        if name not in profiles:
            raise ValueError(f"evaluation data has no additional evaluation profile: {name}")
        if source_recipe and isinstance(profiles[name], Mapping) and profiles[name].get("data_files") is not None:
            raise ValueError(f"evaluation profile {name}: both data_files and evaluation_data")


def evaluation_profiles(
    config: DictConfig, *, evaluation_data: tuple[tuple[str, tuple[str, ...]], ...] = ()
) -> dict[str, EvaluationProfile]:
    """Resolve and record named requests in one complete native SkyRL configuration."""
    staged = dict(evaluation_data)
    resolved = {}
    for name, declaration in _declared_profiles(config).items():
        document = declaration
        if isinstance(declaration, Mapping):
            sampling = declaration.get("sampling_params")
            if sampling is None or isinstance(sampling, Mapping):
                sampling = OmegaConf.to_container(config.generator.eval_sampling_params, resolve=True) | (
                    sampling or {}
                )
            document = {
                "schedule": "every_evaluation",
                "data_files": None,
                **declaration,
                "sampling_params": sampling,
                "n_samples_per_prompt": (
                    int(config.generator.eval_n_samples_per_prompt)
                    if declaration.get("n_samples_per_prompt") is None
                    else declaration["n_samples_per_prompt"]
                ),
            }
            if name in staged:
                document["data_files"] = staged[name]
        try:
            resolved[name] = EvaluationProfile.model_validate_json(json.dumps(document))
        except (TypeError, ValueError) as error:
            raise ValueError(f"evaluation profile {name}: {error}") from error
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


def validate_evaluation_schedule(
    config: DictConfig, profiles: Mapping[str, EvaluationProfile], *, generation: bool = False
) -> None:
    if generation:
        validate_generation_profiles(profiles)
    evaluations = [
        callback for callback in config.trainer.get("callbacks") or () if callback.get("type") == "evaluation"
    ]
    if any(callback.get("eval_steps", DEFAULT_EVALUATION_STEPS) > 0 for callback in evaluations):
        final_prefixes = tuple(f"eval/{name}/" for name, profile in profiles.items() if profile.schedule == "final")
        for callback in evaluations:
            inputs = set(callback.get("stop_when") or {})
            inputs.update(key for members in (callback.get("metric_groups") or {}).values() for key in members)
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
