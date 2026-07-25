# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validated model, serving, generation, and agent configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import draccus
from rigging.filesystem import StoragePath


class ServeBackend(StrEnum):
    """Inference backend used for evaluation."""

    VLLM = "vllm"
    LEVANTER = "levanter"


@dataclass(frozen=True)
class ResourceHint:
    """Serving resources required by a model.

    Set ``hbm_gb`` for a model that can run on either TPU or GPU; the experiment fleet chooses a
    slice with enough usable HBM. Set ``gpu`` instead for a GPU-required model. Its keys are canonical
    uppercase GPU types and its values are acceptable exact device counts, for example
    ``{"H100": 8}``. A CLI accelerator override may change the GPU shape but cannot move a
    GPU-required model onto TPU.

    ``cpu``, ``memory``, and ``disk`` override the inference worker's host-resource defaults.
    """

    hbm_gb: int | None = None
    gpu: Mapping[str, int] = field(default_factory=dict)
    cpu: float | None = None
    memory: str | None = None
    disk: str | None = None

    def __post_init__(self) -> None:
        if self.hbm_gb is not None and self.gpu:
            raise ValueError("resource_hint requires at most one of hbm_gb or gpu")
        if self.hbm_gb is not None and self.hbm_gb <= 0:
            raise ValueError("resource_hint.hbm_gb must be positive")
        normalized_gpu: dict[str, int] = {}
        for gpu_type, count in self.gpu.items():
            canonical_type = gpu_type.upper()
            if canonical_type in normalized_gpu:
                raise ValueError(f"duplicate resource_hint.gpu type after normalization: {canonical_type}")
            if count <= 0 or count & (count - 1):
                raise ValueError(f"resource_hint.gpu[{gpu_type!r}] must be a positive power of two")
            normalized_gpu[canonical_type] = count
        object.__setattr__(self, "gpu", normalized_gpu)
        if self.cpu is not None and self.cpu <= 0:
            raise ValueError("resource_hint.cpu must be positive")


@dataclass(frozen=True)
class ServeConfig:
    """Model-server behavior independent of scheduler placement.

    ``backend`` selects vLLM or Levanter. Parallelism, context, and engine limits become first-class
    inference settings. The remaining typed vLLM fields map onto ``vllm serve`` flags through
    :func:`serve_config_vllm_args`; ``vllm_extra_args`` is the escape hatch for flags without a typed
    field and wins when it names the same option.

    When ``auto_overrides`` is true, the lowering path inspects the Hugging Face ``config.json`` to
    fill portable architecture-specific vLLM flags and clamp an explicit context length to the
    checkpoint's native limit. ``chat_template`` is literal Jinja content passed to the server.
    """

    backend: ServeBackend = ServeBackend.VLLM
    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    hf_overrides: str | None = None
    limit_mm_per_prompt: str | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    vllm_extra_args: tuple[str, ...] = ()
    chat_template: str | None = None
    auto_overrides: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    """Generation overrides applied when an experiment resolves an Evalchemy definition."""

    max_gen_toks: int | None = None
    extra_gen_kwargs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    """Harbor agent arguments applied when an experiment resolves an agentic definition."""

    agent_kwargs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """A model the launcher can serve and evaluate: where its weights live and how to serve/query it.

    ``location`` is an HF repo id or an object-store (``gs://``/``s3://``) HF-format export directory;
    an object-store location requires ``tokenizer`` (the eval client loads its tokenizer through HF).
    ``revision`` pins an immutable checkpoint for a base HF model. ``apply_chat_template`` controls
    whether Evalchemy formats requests with the tokenizer's chat template. ``resource_hint`` states
    where the model is compatible; ``serve`` states how its inference server behaves. ``generation``
    and ``agent`` are experiment-definition inputs and never affect inference placement.
    """

    name: str
    location: str
    revision: str | None = None
    tokenizer: str | None = None
    apply_chat_template: bool = True
    resource_hint: ResourceHint = field(default_factory=ResourceHint)
    serve: ServeConfig = field(default_factory=ServeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def has_vllm_option(args: tuple[str, ...], option: str) -> bool:
    """Whether ``args`` already specifies a vLLM option in either CLI spelling."""
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def serve_config_vllm_args(serve: ServeConfig) -> tuple[str, ...]:
    """Render a :class:`ServeConfig`'s typed serve knobs into ``vllm serve`` flags.

    The typed knobs (``hf_overrides``, ``limit_mm_per_prompt``, ``tool_call_parser``,
    ``reasoning_parser``, ``data_parallel_size``) come first, then the explicit ``vllm_extra_args``
    escape hatch. An explicit ``vllm_extra_args`` entry wins: a typed knob is skipped when its flag is
    already present there, so a hand-tuned value is never duplicated. ``tensor_parallel_size``,
    ``max_model_len``, ``max_num_batched_tokens``, ``max_num_seqs``, ``chat_template``, and
    ``auto_overrides`` are consumed by serving configuration or lowering rather than rendered as
    extra flags. ``--trust-remote-code`` is not rendered here: the native server forces it on for
    every evaluated model.
    """
    explicit = tuple(serve.vllm_extra_args)
    derived: list[str] = []

    def add(option: str, *values: str) -> None:
        if not has_vllm_option(explicit, option):
            derived.extend((option, *values))

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


def load_model_config(path: Path) -> ModelConfig:
    """Load one model catalog file, rejecting fields outside the configuration schema."""
    with StoragePath(str(path)).open("r") as handle:
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
