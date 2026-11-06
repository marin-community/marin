"""Named dataset and sampling overrides for policy evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from marinskyrl.resource_locator import join_resource_path
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class EvaluationProfile:
    sampling_params: dict[str, Any] | None = None
    n_samples_per_prompt: int | None = None
    data_files: list[str] | None = None


def parse_evaluation_profiles(values: Mapping[str, Any] | None) -> dict[str, EvaluationProfile]:
    profiles = {name: EvaluationProfile(**parameters) for name, parameters in (values or {}).items()}
    if any(not name.isidentifier() for name in profiles):
        raise ValueError("additional evaluation names must be identifiers")
    return profiles


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
