# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The canonical model/serve configuration protocol for the eval launcher.

One :class:`ModelConfig` is the single in-memory contract every eval consumer imports: the group
launcher (evalchemy and Harbor), the serve path, and the agentic benchmarks all read the same object.
It supersedes the earlier per-launcher ``EvalModelConfig`` and folds in the per-model serve catalog
extracted from OT-Agent's agentic-evals package.

Two population paths produce one identical ``ModelConfig``:

- **YAML** (:func:`load_model_config` / :func:`scan_model_configs`) is the validated front-door for
  the bulk catalog. draccus decodes ``serve/models/<org>/<model>.yaml`` against the dataclass, so an
  unknown field or a mistyped value fails at load, not at serve time. One file per model mirrors HF's
  ``<org>/<model>`` namespacing and keeps the catalog diff-friendly.
- **Python factories** (in ``experiments.evaluation.models``) stay for the parametric entries whose
  serve options are computed rather than curated (e.g. the 256-expert MoE that resolves to
  ``tensor_parallel_size=1`` plus expert parallelism).

Both feed the same normalization boundary: :func:`serve_config_vllm_args` renders the typed serve
knobs into ``vllm serve`` flags, and the launcher's ``auto_serve_overrides`` fills the remaining gaps
from the model's ``config.json`` (an explicit value always wins).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path

import draccus


class ServeBackend(StrEnum):
    """Which marin-serve backend serves the model under eval. Both expose the same OpenAI API, so the
    eval client is identical either way."""

    VLLM = "vllm"
    LEVANTER = "levanter"


@dataclass(frozen=True)
class ServeConfig:
    """How a model is served: its slice budget, parallelism, and vLLM serve knobs.

    Sizing: ``hbm_gb`` is the serving HBM budget the hardware selector turns into a slice; ``fixed_gpu``
    pins an exact ``(gpu_type, count)`` shape instead, and ``gpu_only`` forces the GPU path for a model
    the TPU stack cannot serve (a quantized checkpoint, a fork-only architecture). ``target_cluster``
    names the CoreWeave peer a GPU job routes to.

    vLLM knobs map onto ``vllm serve`` flags through :func:`serve_config_vllm_args`; every explicit
    value here wins over what ``auto_serve_overrides`` would derive from the model's ``config.json``.
    ``vllm_extra_args`` is the escape hatch for flags without a typed field. ``variants`` carries
    per-hardware overrides (e.g. ``gh200``) from the imported catalog; :func:`resolve_serve_variant`
    applies one when the served slice's label matches.
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
    variants: Mapping[str, ServeConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    """Generation knobs for the evalchemy client: the token budget and extra sampler settings.

    ``max_gen_toks`` overrides a suite's default generation budget for a verbose reasoning model whose
    chain would otherwise truncate before the answer. ``extra_gen_kwargs`` are forwarded verbatim into
    lm-eval's ``--gen_kwargs`` (``key=value`` pairs); ``snowball-sft`` needs
    ``skip_special_tokens=false`` so its ``<|start_think|>`` delimiters survive, plus
    ``repetition_penalty=1.1``.
    """

    max_gen_toks: int | None = None
    extra_gen_kwargs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    """Agent knobs for the Harbor/agentic path, forwarded to the agent driving each sandbox trial.

    ``agent_kwargs`` are passed through to the Harbor agent (``enable_thinking``, an ``extra_body``
    template, ...); they flow into the OpenAI request the agent makes against the served endpoint.
    """

    agent_kwargs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """A model the launcher can serve and evaluate: where its weights live and how to serve/query it.

    ``location`` is an HF repo id or an object-store (``gs://``/``s3://``) HF-format export directory;
    an object-store location requires ``tokenizer`` (the eval client loads its tokenizer through HF).
    ``revision`` pins an immutable checkpoint for a base HF model. ``apply_chat_template`` selects the
    chat benchmarks (a base model with no chat template runs the NLP suite instead).
    """

    name: str
    location: str
    revision: str | None = None
    tokenizer: str | None = None
    apply_chat_template: bool = True
    serve: ServeConfig = field(default_factory=ServeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _has_vllm_option(args: tuple[str, ...], option: str) -> bool:
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
        if not _has_vllm_option(explicit, option):
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

    A variant carries per-hardware overrides (e.g. a ``gh200`` slice collapsing tensor parallelism to
    1). Only the variant fields that differ from a fresh :class:`ServeConfig` default overlay the base,
    so a partial variant leaves the base's other knobs intact. No matching variant returns ``serve``
    unchanged; the marin clusters (H100/GB200/TPU) do not match the imported ``gh200`` variants, so
    those are inert until a matching hardware label exists.
    """
    if hardware_label is None or hardware_label not in serve.variants:
        return serve
    variant = serve.variants[hardware_label]
    defaults = ServeConfig()
    overrides = {
        f.name: getattr(variant, f.name)
        for f in fields(ServeConfig)
        if f.name != "variants" and getattr(variant, f.name) != getattr(defaults, f.name)
    }
    return dataclasses.replace(serve, **overrides)


def load_model_config(path: str | Path) -> ModelConfig:
    """Decode one ``serve/models/<org>/<model>.yaml`` into a :class:`ModelConfig`.

    draccus validates the YAML against the dataclass schema, so an unknown field or a mistyped value
    raises at load rather than surfacing as a bad serve flag later.
    """
    with open(path) as handle:
        return draccus.load(ModelConfig, handle)


def scan_model_configs(root: str | Path) -> dict[str, ModelConfig]:
    """Load every ``*.yaml`` under ``root`` into a ``{name: ModelConfig}`` registry.

    Files and directories whose names start with ``_`` or ``.`` are skipped (``_patterns.yaml``,
    ``README``-adjacent scratch). A ``name`` collision across two files is an error: the catalog keys
    by ``ModelConfig.name``, so a duplicate would silently shadow one entry.
    """
    root = Path(root)
    configs: dict[str, ModelConfig] = {}
    for path in sorted(root.rglob("*.yaml")):
        if any(part.startswith(("_", ".")) for part in path.relative_to(root).parts):
            continue
        config = load_model_config(path)
        if config.name in configs:
            raise ValueError(f"duplicate model name {config.name!r} in catalog: {path} and an earlier file")
        configs[config.name] = config
    return configs
