"""Typed configuration contracts for EAGLE speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
import math
import os
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from marinskyrl.resource_locator import is_cloud_uri, is_hugging_face_repo_id


_HF_SOURCE_SCHEME = "hf"
_HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
STANDARD_TRAINING_ENTRYPOINT = "skyrl_train.entrypoints.main_base"


def is_hugging_face_commit(value: str) -> bool:
    """Return whether ``value`` is a full immutable Hugging Face commit SHA."""
    return _HF_COMMIT_PATTERN.fullmatch(value) is not None


class SpeculativeDecodingMethod(StrEnum):
    """Speculative methods supported by MarinSkyRL's managed lifecycle."""

    EAGLE3 = "eagle3"


class SpeculatorModelSourceKind(StrEnum):
    """Storage kinds supported by the draft-model lifecycle."""

    HUGGING_FACE = "hugging_face"
    LOCAL = "local"
    ARTIFACT = "artifact"


class SpeculativeDecodingConfigError(ValueError):
    """A managed speculative-decoding configuration is invalid."""


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpeculativeDecodingConfigError(f"{field} must be a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise SpeculativeDecodingConfigError(f"unknown {field} fields: {', '.join(sorted(unknown))}")


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpeculativeDecodingConfigError(f"{field} must be a positive integer, got {value!r}")
    return value


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SpeculativeDecodingConfigError(f"{field} must be a positive finite number, got {value!r}")
    return float(value)


def _nonnegative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise SpeculativeDecodingConfigError(f"{field} must be a nonnegative finite number, got {value!r}")
    return float(value)


def _fraction(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise SpeculativeDecodingConfigError(f"{field} must be in (0, 1), got {value!r}")
    return float(value)


def hugging_face_repo_from_source_uri(source_uri: str) -> str | None:
    """Return the repo ID encoded by ``hf://namespace/repository``."""
    parsed = urlsplit(source_uri)
    if parsed.scheme != _HF_SOURCE_SCHEME:
        return None
    if parsed.query or parsed.fragment:
        raise SpeculativeDecodingConfigError("Hugging Face speculator source_uri cannot contain a query or fragment")
    repo_id = f"{parsed.netloc}{parsed.path}"
    if not is_hugging_face_repo_id(repo_id):
        raise SpeculativeDecodingConfigError(
            "Hugging Face speculator source_uri must have the form hf://namespace/repository"
        )
    return repo_id


def runai_model_uri(source_uri: str) -> str:
    """Return an object-store URI accepted by vLLM's RunAI loader."""
    if source_uri.startswith("gcs://"):
        return f"gs://{source_uri.removeprefix('gcs://')}"
    if source_uri.startswith(("s3://", "gs://")):
        return source_uri
    raise SpeculativeDecodingConfigError(f"RunAI model loading requires an s3:// or gs:// URI, got {source_uri!r}")


@dataclass(frozen=True)
class SpeculatorModelConfig:
    """One immutable local, Hugging Face, or object-store draft model."""

    source_uri: str
    source_identity: str

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> "SpeculatorModelConfig":
        mapping = _mapping(value, context)
        _reject_unknown(mapping, {"source_uri", "source_identity"}, context)
        missing = {field for field in ("source_uri", "source_identity") if not mapping.get(field)}
        if missing:
            raise SpeculativeDecodingConfigError(f"missing {context} fields: {', '.join(sorted(missing))}")

        source_uri = mapping["source_uri"]
        source_identity = mapping["source_identity"]
        if not isinstance(source_uri, str):
            raise SpeculativeDecodingConfigError(f"{context}.source_uri must be a string")
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise SpeculativeDecodingConfigError(f"{context}.source_identity must be a nonempty string")

        hf_repo = hugging_face_repo_from_source_uri(source_uri)
        if hf_repo is not None:
            if not is_hugging_face_commit(source_identity):
                raise SpeculativeDecodingConfigError(
                    f"{context}.source_identity must be a full 40-character lowercase commit SHA for {hf_repo}"
                )
        elif not is_cloud_uri(source_uri) and not os.path.isabs(source_uri):
            raise SpeculativeDecodingConfigError(
                f"{context}.source_uri must be an absolute local path or use hf://, s3://, gs://, or gcs://, "
                f"got {source_uri!r}"
            )
        elif is_cloud_uri(source_uri):
            parsed = urlsplit(source_uri)
            if not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
                raise SpeculativeDecodingConfigError(f"{context}.source_uri is not a complete object-store URI")

        return cls(
            source_uri=source_uri,
            source_identity=source_identity,
        )

    @property
    def hugging_face_repo_id(self) -> str | None:
        return hugging_face_repo_from_source_uri(self.source_uri)

    @property
    def local_source_path(self) -> str | None:
        """Return the source path when this model already exists locally."""
        return self.source_uri if os.path.isabs(self.source_uri) else None

    @property
    def source_kind(self) -> SpeculatorModelSourceKind:
        if self.local_source_path is not None:
            return SpeculatorModelSourceKind.LOCAL
        if self.hugging_face_repo_id is not None:
            return SpeculatorModelSourceKind.HUGGING_FACE
        return SpeculatorModelSourceKind.ARTIFACT

    def vllm_source_config(self) -> dict[str, Any]:
        """Return the vLLM fields needed to load this draft source."""
        if self.source_kind is SpeculatorModelSourceKind.LOCAL:
            assert self.local_source_path is not None
            return {"model": self.local_source_path}
        if self.source_kind is SpeculatorModelSourceKind.HUGGING_FACE:
            assert self.hugging_face_repo_id is not None
            model_id = self.hugging_face_repo_id
            return {"model": model_id, "revision": self.source_identity}
        return {
            "model": runai_model_uri(self.source_uri),
            "draft_load_config": {"load_format": "runai_streamer"},
        }


@dataclass(frozen=True)
class SpeculatorTrainingConfig:
    """Bounded single-rank online EAGLE update settings."""

    interval_steps: int = 1
    # Bound the deterministic all-DP merge while leaving enough admission
    # headroom for rollout response lengths to vary between steps.
    max_tokens_per_update: int = 16_384
    # A request-local history cannot be split without changing its attention
    # semantics. This is therefore a separate hard forward bound; the packed
    # microbatch setting below is a target that a single admitted window may
    # exceed.
    max_window_tokens: int = 16_384
    max_tokens_per_micro_batch: int = 2_048
    max_sequences_per_prompt_group: int = 2
    min_train_sequences: int = 6
    holdout_fraction: float = 0.25
    min_holdout_sequences: int = 3
    epochs_per_update: int = 1
    learning_rate: float = 5e-5
    max_validation_loss_increase: float = 0
    max_validation_agreement_decrease: float = 0
    reserved_gpu_memory_gib: float = 8

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> "SpeculatorTrainingConfig":
        mapping = _mapping(value, context)
        field_names = {field.name for field in fields(cls)}
        _reject_unknown(mapping, field_names, context)
        defaults = cls()
        return cls(
            interval_steps=_positive_integer(
                mapping.get("interval_steps", defaults.interval_steps), f"{context}.interval_steps"
            ),
            max_tokens_per_update=_positive_integer(
                mapping.get("max_tokens_per_update", defaults.max_tokens_per_update),
                f"{context}.max_tokens_per_update",
            ),
            max_window_tokens=_positive_integer(
                mapping.get("max_window_tokens", defaults.max_window_tokens),
                f"{context}.max_window_tokens",
            ),
            max_tokens_per_micro_batch=_positive_integer(
                mapping.get("max_tokens_per_micro_batch", defaults.max_tokens_per_micro_batch),
                f"{context}.max_tokens_per_micro_batch",
            ),
            max_sequences_per_prompt_group=_positive_integer(
                mapping.get("max_sequences_per_prompt_group", defaults.max_sequences_per_prompt_group),
                f"{context}.max_sequences_per_prompt_group",
            ),
            min_train_sequences=_positive_integer(
                mapping.get("min_train_sequences", defaults.min_train_sequences), f"{context}.min_train_sequences"
            ),
            holdout_fraction=_fraction(
                mapping.get("holdout_fraction", defaults.holdout_fraction), f"{context}.holdout_fraction"
            ),
            min_holdout_sequences=_positive_integer(
                mapping.get("min_holdout_sequences", defaults.min_holdout_sequences),
                f"{context}.min_holdout_sequences",
            ),
            epochs_per_update=_positive_integer(
                mapping.get("epochs_per_update", defaults.epochs_per_update), f"{context}.epochs_per_update"
            ),
            learning_rate=_positive_number(
                mapping.get("learning_rate", defaults.learning_rate), f"{context}.learning_rate"
            ),
            max_validation_loss_increase=_nonnegative_number(
                mapping.get("max_validation_loss_increase", defaults.max_validation_loss_increase),
                f"{context}.max_validation_loss_increase",
            ),
            max_validation_agreement_decrease=_nonnegative_number(
                mapping.get("max_validation_agreement_decrease", defaults.max_validation_agreement_decrease),
                f"{context}.max_validation_agreement_decrease",
            ),
            reserved_gpu_memory_gib=_positive_number(
                mapping.get("reserved_gpu_memory_gib", defaults.reserved_gpu_memory_gib),
                f"{context}.reserved_gpu_memory_gib",
            ),
        )


@dataclass(frozen=True)
class SpeculativeDecodingConfig:
    """Managed serving and optional online-training configuration."""

    method: SpeculativeDecodingMethod
    model: SpeculatorModelConfig
    num_speculative_tokens: int
    training: SpeculatorTrainingConfig | None

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str = "generator.speculative_decoding",
    ) -> "SpeculativeDecodingConfig":
        mapping = _mapping(value, context)
        _reject_unknown(mapping, {"method", "model", "num_speculative_tokens", "training"}, context)
        missing = {field for field in ("method", "model", "num_speculative_tokens") if field not in mapping}
        if missing:
            raise SpeculativeDecodingConfigError(f"missing {context} fields: {', '.join(sorted(missing))}")
        try:
            method = SpeculativeDecodingMethod(mapping["method"])
        except (TypeError, ValueError) as error:
            raise SpeculativeDecodingConfigError(f"{context}.method must be eagle3") from error

        raw_training = mapping.get("training")
        training = (
            None
            if raw_training is None
            else SpeculatorTrainingConfig.from_mapping(raw_training, context=f"{context}.training")
        )
        return cls(
            method=method,
            model=SpeculatorModelConfig.from_mapping(mapping["model"], context=f"{context}.model"),
            num_speculative_tokens=_positive_integer(
                mapping["num_speculative_tokens"], f"{context}.num_speculative_tokens"
            ),
            training=training,
        )

    def vllm_speculative_config(self) -> dict[str, Any]:
        """Return the serving fields understood by vLLM."""
        return {
            "method": self.method.value,
            **self.model.vllm_source_config(),
            "num_speculative_tokens": self.num_speculative_tokens,
        }


def parse_speculative_decoding_config(
    value: object,
    *,
    backend: str,
    run_engines_locally: bool,
    entrypoint: str,
    colocate_all: bool,
    num_inference_engines: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    engine_init_kwargs: Mapping[str, Any] | None = None,
    context: str = "generator.speculative_decoding",
) -> SpeculativeDecodingConfig | None:
    """Parse the optional public block and reject unsupported execution modes."""
    if value is None:
        return None
    config = SpeculativeDecodingConfig.from_mapping(value, context=context)
    if backend != "vllm":
        raise SpeculativeDecodingConfigError(f"{context} requires generator.backend=vllm")
    if not run_engines_locally:
        raise SpeculativeDecodingConfigError(f"{context} requires generator.run_engines_locally=true")
    if colocate_all:
        raise SpeculativeDecodingConfigError(f"{context} requires trainer.placement.colocate_all=false")
    if config.training is not None and entrypoint != STANDARD_TRAINING_ENTRYPOINT:
        raise SpeculativeDecodingConfigError(f"{context}.training is not supported by entrypoint {entrypoint!r}")
    if config.training is not None and num_inference_engines != 1:
        raise SpeculativeDecodingConfigError(f"{context}.training requires generator.num_inference_engines=1")
    if config.training is not None and tensor_parallel_size != 1:
        raise SpeculativeDecodingConfigError(
            f"{context}.training requires generator.inference_engine_tensor_parallel_size=1"
        )
    if config.training is not None and pipeline_parallel_size != 1:
        raise SpeculativeDecodingConfigError(
            f"{context}.training requires generator.inference_engine_pipeline_parallel_size=1"
        )
    if config.training is not None and (engine_init_kwargs or {}).get("async_scheduling", False):
        raise SpeculativeDecodingConfigError(f"{context}.training does not support vLLM async_scheduling")
    return config
