# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validated model, serving, generation, and agent configuration."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path

import draccus
import fsspec


class ServeBackend(StrEnum):
    """Inference backend used for evaluation."""

    VLLM = "vllm"
    LEVANTER = "levanter"


class VariantUnset(StrEnum):
    """Sentinel distinguishing an omitted variant field from an explicit default value."""

    VALUE = "__unset__"


@dataclass(frozen=True)
class ServeVariant:
    """Fields a hardware variant overlays onto its model's base serve configuration."""

    backend: ServeBackend | VariantUnset = VariantUnset.VALUE
    hbm_gb: int | None | VariantUnset = VariantUnset.VALUE
    fixed_gpu: tuple[str, int] | None | VariantUnset = VariantUnset.VALUE
    gpu_only: bool | VariantUnset = VariantUnset.VALUE
    tensor_parallel_size: int | None | VariantUnset = VariantUnset.VALUE
    data_parallel_size: int | None | VariantUnset = VariantUnset.VALUE
    max_model_len: int | None | VariantUnset = VariantUnset.VALUE
    swap_space_gb: int | None | VariantUnset = VariantUnset.VALUE
    trust_remote_code: bool | VariantUnset = VariantUnset.VALUE
    hf_overrides: str | None | VariantUnset = VariantUnset.VALUE
    limit_mm_per_prompt: str | None | VariantUnset = VariantUnset.VALUE
    tool_call_parser: str | None | VariantUnset = VariantUnset.VALUE
    reasoning_parser: str | None | VariantUnset = VariantUnset.VALUE
    vllm_extra_args: tuple[str, ...] | VariantUnset = VariantUnset.VALUE
    chat_template: str | None | VariantUnset = VariantUnset.VALUE
    serve_memory: str | None | VariantUnset = VariantUnset.VALUE
    target_cluster: str | None | VariantUnset = VariantUnset.VALUE
    auto_overrides: bool | VariantUnset = VariantUnset.VALUE


@dataclass(frozen=True)
class ServeConfig:
    """How a model is served: its slice budget, parallelism, and vLLM serve knobs.

    Sizing: ``hbm_gb`` is the serving HBM budget the hardware selector turns into a slice; ``fixed_gpu``
    pins an exact ``(gpu_type, count)`` shape instead, and ``gpu_only`` forces the GPU path for a model
    the TPU stack cannot serve (a quantized checkpoint, a fork-only architecture). ``target_cluster``
    names the CoreWeave peer a GPU job routes to.

    vLLM knobs map onto ``vllm serve`` flags through :func:`serve_config_vllm_args`.
    ``auto_serve_overrides`` fills unset fields from the model's ``config.json`` and may clamp an
    explicit context length to the model's native limit. ``vllm_extra_args`` is the escape hatch for
    flags without a typed field. ``variants`` carries per-hardware overrides;
    :func:`resolve_serve_variant` applies one when the slice label matches.
    """

    backend: ServeBackend = ServeBackend.VLLM
    hbm_gb: int | None = None
    fixed_gpu: tuple[str, int] | None = None
    gpu_only: bool = False
    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    max_model_len: int | None = None
    swap_space_gb: int | None = None
    trust_remote_code: bool = False
    hf_overrides: str | None = None
    limit_mm_per_prompt: str | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    vllm_extra_args: tuple[str, ...] = ()
    chat_template: str | None = None
    serve_memory: str | None = None
    """Host-memory request for the serve child, overriding the serve default. Large object-store
    exports need it: weight streaming stages shards through host buffers, so the pod's memory limit
    must cover the full weight volume or the kernel OOM-kills the server."""
    target_cluster: str | None = None
    auto_overrides: bool = True
    variants: Mapping[str, ServeVariant] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    """Model-specific generation overrides."""

    max_gen_toks: int | None = None
    extra_gen_kwargs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    """Model-specific Harbor agent arguments."""

    agent_kwargs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """A model the launcher can serve and evaluate: where its weights live and how to serve/query it.

    ``location`` is an HF repo id or an object-store (``gs://``/``s3://``) HF-format export directory;
    an object-store location requires ``tokenizer`` (the eval client loads its tokenizer through HF).
    ``revision`` pins an immutable checkpoint for a base HF model. ``apply_chat_template`` controls
    whether Evalchemy formats requests with the tokenizer's chat template.
    """

    name: str
    location: str
    revision: str | None = None
    tokenizer: str | None = None
    apply_chat_template: bool = True
    serve: ServeConfig = field(default_factory=ServeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def has_vllm_option(args: tuple[str, ...], option: str) -> bool:
    """Whether ``args`` already specifies a vLLM option in either CLI spelling."""
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def serve_config_vllm_args(serve: ServeConfig) -> tuple[str, ...]:
    """Render a :class:`ServeConfig`'s typed serve knobs into ``vllm serve`` flags.

    The typed knobs (``swap_space_gb``, ``trust_remote_code``, ``hf_overrides``, ``limit_mm_per_prompt``,
    ``tool_call_parser``, ``reasoning_parser``, ``data_parallel_size``) come first, then the explicit
    ``vllm_extra_args`` escape hatch. An explicit ``vllm_extra_args`` entry wins: a typed knob is
    skipped when its flag is already present there, so a hand-tuned value is never duplicated.
    ``tensor_parallel_size``, ``max_model_len``, and ``chat_template`` are omitted -- they are
    first-class serve fields the launcher passes through the served-model config, not extra flags.
    """
    explicit = tuple(serve.vllm_extra_args)
    derived: list[str] = []

    def add(option: str, *values: str) -> None:
        if not has_vllm_option(explicit, option):
            derived.extend((option, *values))

    if serve.trust_remote_code:
        add("--trust-remote-code")
    if serve.swap_space_gb is not None:
        add("--swap-space", str(serve.swap_space_gb))
    if serve.data_parallel_size is not None:
        add("--data-parallel-size", str(serve.data_parallel_size))
    if serve.hf_overrides is not None:
        add("--hf-overrides", serve.hf_overrides)
    if serve.limit_mm_per_prompt is not None:
        add("--limit-mm-per-prompt", serve.limit_mm_per_prompt)
    if serve.reasoning_parser is not None:
        add("--reasoning-parser", serve.reasoning_parser)
    if serve.tool_call_parser is not None:
        # vLLM only honors a tool-call parser when auto tool choice is enabled.
        add("--enable-auto-tool-choice")
        add("--tool-call-parser", serve.tool_call_parser)
    return (*derived, *explicit)


def resolve_serve_variant(serve: ServeConfig, hardware_label: str | None) -> ServeConfig:
    """Overlay ``serve.variants[hardware_label]`` onto ``serve`` when the served slice matches.

    Omitted fields leave the base value intact. Explicit defaults, including ``false`` and ``null``,
    replace the base value. No matching variant returns ``serve`` unchanged.
    """
    if hardware_label is None or hardware_label not in serve.variants:
        return serve
    variant = serve.variants[hardware_label]
    overrides = {
        f.name: getattr(variant, f.name)
        for f in fields(ServeVariant)
        if getattr(variant, f.name) is not VariantUnset.VALUE
    }
    return dataclasses.replace(serve, **overrides)


def load_model_config(path: Path) -> ModelConfig:
    """Load one model catalog file, rejecting fields outside the configuration schema."""
    with fsspec.open(str(path), "r") as handle:
        return draccus.load(ModelConfig, handle)


def scan_model_configs(root: Path) -> dict[str, ModelConfig]:
    """Load every ``*.yaml`` under ``root`` into a ``{name: ModelConfig}`` registry.

    Files and directories whose names start with ``_`` or ``.`` are skipped (``_patterns.yaml``,
    ``README``-adjacent scratch). A ``name`` collision across two files is an error: the catalog keys
    by ``ModelConfig.name``, so a duplicate would silently shadow one entry.
    """
    configs: dict[str, ModelConfig] = {}
    for path in sorted(root.rglob("*.yaml")):
        if any(part.startswith(("_", ".")) for part in path.relative_to(root).parts):
            continue
        config = load_model_config(path)
        if config.name in configs:
            raise ValueError(f"duplicate model name {config.name!r} in catalog: {path} and an earlier file")
        configs[config.name] = config
    return configs
