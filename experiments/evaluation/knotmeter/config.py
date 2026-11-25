# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate serving presets and resolve their shared measurement inputs."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from marin.inference.config import VLLM_TOPOLOGY_OPTIONS, ServingGeometry
from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.evaluation.knotmeter.recipe import DIRECTORY, load_series


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage(StrEnum):
    PROBE = "probe"
    SCREEN = "screen"
    CONFIRM = "confirm"


class MeasurementOverrides(Config):
    duration: int | None = Field(default=None, gt=0)
    warmup_requests_per_lane: int | None = Field(default=None, ge=0)
    concurrency: int | None = Field(default=None, gt=0)
    unsafe_override: bool | None = None


class Model(Config):
    repo: str
    revision: str
    tokenizer: str
    tokenizer_revision: str
    path: str | None = None
    config_sha256: str | None = None
    chat_template: Literal["marin"] | None = None
    eos_token_id: int | None = None


class Gpus(Config):
    type: Literal["H100", "GB200"]
    count: int = Field(gt=0)
    per_host: int = Field(gt=0)


class Runtime(Config):
    release: str | None = None
    manifest: str | None = None
    repo: str | None = None
    revision: str | None = None


class Serving(Config):
    dtype: str = "bfloat16"
    tensor_parallel_size: int = Field(default=1, gt=0)
    max_model_len: int = Field(gt=0)
    max_num_seqs: int = Field(gt=0)
    max_num_batched_tokens: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, lt=1)
    prefix_caching: bool = True
    router: bool = False
    extra_args: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)


class Workload(Config):
    public_dataset: str


class Check(Config):
    name: str
    command: list[str] = Field(default_factory=list)
    fixture: str | None = None
    timeout: int = Field(default=900, gt=0)
    kind: Literal["serving", "numerical"] = "numerical"
    checkpoint_revision: str | None = None
    runtime_source: str | None = None
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def at_most_one_implementation(self) -> "Check":
        if self.fixture and self.command:
            raise ValueError("Use a shared fixture or an external command for a check")
        return self


class Profile(Config):
    model: Model
    gpus: Gpus
    cluster: str
    priority: Literal["batch", "interactive"]
    session_concurrency: int = Field(gt=0)
    quantization: str | None = None
    weight_precision: str | None = None
    serving: Serving
    workload: Workload
    vllm: Runtime = Field(default_factory=Runtime)
    checks: list[Check] = Field(default_factory=list)
    cpu: int = 32
    memory: str = "768GB"
    disk: str = "1TB"

    @property
    def geometry(self) -> ServingGeometry:
        return ServingGeometry(
            tensor_parallel_size=self.serving.tensor_parallel_size,
            data_parallel_size=self.gpus.count // self.serving.tensor_parallel_size,
            pipeline_parallel_size=1,
            gpus_per_task=self.gpus.per_host,
        )

    @model_validator(mode="after")
    def valid_serving_profile(self) -> "Profile":
        if self.geometry.total_gpus != self.gpus.count:
            raise ValueError("GPU count must divide into tensor-parallel groups")
        if self.vllm.release and self.vllm.manifest:
            raise ValueError("Choose vllm.release or vllm.manifest")
        if (self.vllm.repo or self.vllm.revision) and not self.vllm.manifest:
            raise ValueError("Source overrides require the resulting build's vllm.manifest")
        owned = VLLM_TOPOLOGY_OPTIONS | {
            "--model",
            "--revision",
            "--tokenizer",
            "--tokenizer-revision",
            "--served-model-name",
            "--dtype",
            "--quantization",
            "--max-model-len",
            "--max-num-seqs",
            "--max-num-batched-tokens",
            "--gpu-memory-utilization",
            "--enable-prefix-caching",
            "--no-enable-prefix-caching",
        }
        duplicates = owned.intersection(arg.partition("=")[0] for arg in self.serving.extra_args)
        if duplicates:
            raise ValueError(f"Use profile fields instead of duplicate serving flags: {sorted(duplicates)}")
        return self


def load_profile(path: Path, overrides: list[str]) -> Profile:
    """Apply YAML-valued dotted overrides, then reject unknown or inconsistent inputs."""
    if not path.is_file():
        path = DIRECTORY / "profiles" / (str(path) if path.suffix in {".yaml", ".yml"} else f"{path}.yaml")
    values = load_series(path)
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected --set name=value, got {override!r}")
        cursor = values
        *parents, leaf = key.split(".")
        for parent in parents:
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot index a list or scalar in {key!r}; replace its whole value")
            cursor = cursor.setdefault(parent, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot index a list or scalar in {key!r}; replace its whole value")
        cursor[leaf] = yaml.safe_load(value)
    return Profile.model_validate(values)


def is_confirm_recipe(series: dict) -> bool:
    """Require Confirm's minimum recipe independently of native window validity."""
    recipe = load_series()
    return (
        series.get("stage") == Stage.CONFIRM
        and series["duration"] >= recipe["duration"]
        and series["warmup_requests_per_lane"] >= recipe["warmup_requests_per_lane"]
        and not series["unsafe_override"]
    )


def resolved_series(
    profile: Profile,
    stage: Stage,
    overrides: MeasurementOverrides = MeasurementOverrides(),
    *,
    measure_failed_checks: bool = False,
) -> dict:
    recipe = load_series()
    datasets = recipe.pop("datasets")
    stages = recipe.pop("stages")
    stage_defaults = stages[stage]
    limit = stage_defaults.pop("concurrency_limit", profile.session_concurrency)
    requested = overrides.model_dump(mode="json")
    recipe |= stage_defaults | {"concurrency": min(profile.session_concurrency, limit)}
    recipe |= overrides.model_dump(exclude_none=True)
    if recipe["duration"] < recipe["minimum_duration"] and not recipe["unsafe_override"]:
        raise ValueError(f"AgentX requires at least {recipe['minimum_duration']} seconds without --unsafe-override")
    selector = profile.workload.public_dataset
    if selector not in datasets:
        raise ValueError(f"No pinned corpus for {selector!r}; add its reviewed identity to recipe.yaml")
    hardware = profile.gpus
    name = "h100" if hardware.type == "H100" else "gb200"
    series = (
        recipe
        | datasets[selector]
        | {
            "model": profile.model.repo,
            "model_prefix": Path(profile.model.repo).name.lower(),
            "model_revision": profile.model.revision,
            "tokenizer": profile.model.tokenizer,
            "tokenizer_revision": profile.model.tokenizer_revision,
            "dataset_loader": selector,
            "max_context": profile.serving.max_model_len,
            "precision": profile.weight_precision or profile.quantization or profile.serving.dtype,
            "tensor_parallel": profile.serving.tensor_parallel_size,
            "profile": profile.model_dump(mode="json"),
            "profiles": {
                name: {"gpu": hardware.type, "gpus": hardware.per_host, "hosts": hardware.count // hardware.per_host}
            },
            "stage": stage.value,
            "measurement_requested": (
                {"stage": stage.value, "target_concurrency": profile.session_concurrency} | requested
            ),
        }
    )
    series["diagnostic"] = not is_confirm_recipe(series)
    if measure_failed_checks:
        series["measure_failed_checks"] = True
    return series
