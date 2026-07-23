# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The model registry for the eval launcher.

Each entry is a :class:`~marin.evaluation.model_config.ModelConfig` -- the one contract the launcher,
the serve path, and the agentic benchmarks share. Two population paths feed one registry:

- **Python factories** (``_snowball``, ``_base_hf``) for the parametric entries whose serve options are
  computed (a 256-expert MoE that resolves to ``tensor_parallel_size=1`` plus expert parallelism, a
  base model pinned to an immutable revision).
- **YAML catalog** under :data:`MODEL_CATALOG_DIR` (``serve/models/<org>/<model>.yaml``), scanned by
  :func:`~marin.evaluation.model_config.scan_model_configs` -- the bulk per-model serve catalog
  imported from OT-Agent's agentic-evals package.

:mod:`experiments.evaluation.hardware` turns ``serve.hbm_gb`` into a slice; :mod:`experiments.evaluation.launch`
turns the rest into a ``ServeSpec``. HBM sizes follow the bf16 rule of thumb:
``params_billions * 2 GB * ~1.3`` for weights plus runtime overhead.
"""

from __future__ import annotations

from pathlib import Path

from marin.evaluation.model_config import (
    AgentConfig,
    GenerationConfig,
    ModelConfig,
    ServeConfig,
    scan_model_configs,
)
from marin.inference.backend import CONCAT_CHAT_TEMPLATE

# The imported per-model YAML catalog. One file per model, mirroring HF's <org>/<model> namespacing.
MODEL_CATALOG_DIR = Path(__file__).parent / "serve" / "models"

# The 256-expert Grug MoE fork serves data-parallel + expert-parallel with tensor_parallel_size=1; the
# per-head TP heuristic cannot infer this, and the loader streams shards distributed across the ranks.
_SNOWBALL_VLLM_ARGS = ("--enable-expert-parallel", "--model-loader-extra-config", '{"distributed":true}')


def _snowball(name: str, location: str, *, chat_template: str | None = None, generation: GenerationConfig | None = None) -> ModelConfig:
    """A Grug 67B-A2B export served on a CoreWeave 8xH100 node via the marin vLLM fork.

    Data-parallel + expert-parallel sharding for the 256-expert MoE with ``tensor_parallel_size=1``.
    The tokenizer is an HF id because the eval client cannot load a tokenizer from the s3:// export;
    ~134GB of bf16 shards stream from object storage through host buffers on load, so the serve child
    gets a generous memory limit (it owns the node).
    """
    return ModelConfig(
        name=name,
        location=location,
        tokenizer="marin-community/marin-tokenizer",
        apply_chat_template=True,
        serve=ServeConfig(
            hbm_gb=175,
            gpu_only=True,
            fixed_gpu=("H100", 8),
            tensor_parallel_size=1,
            data_parallel_size=8,
            serve_memory="512g",
            target_cluster="cw-us-east-02a",
            chat_template=chat_template,
            vllm_extra_args=_SNOWBALL_VLLM_ARGS,
        ),
        generation=generation or GenerationConfig(),
    )


def _base_hf(name: str, location: str, revision: str, hbm_gb: int) -> ModelConfig:
    """A base (non-chat) HF model, pinned to an immutable revision.

    ``apply_chat_template=False`` (base models ship no chat template), so these run the NLP (lm-eval)
    suite, not the chat benchmarks. The revision is served through ``vllm serve --revision`` so results
    are reproducible against a fixed checkpoint rather than the HF branch head.
    """
    return ModelConfig(
        name=name,
        location=location,
        revision=revision,
        apply_chat_template=False,
        serve=ServeConfig(hbm_gb=hbm_gb),
    )


# The parametric / curated entries defined in Python. Simple curated checkpoints could equally live as
# YAML; they stay here so the migration off the earlier registry is a straight rename + grouping and
# the auto_serve_overrides parity is asserted in one place (tests/evaluation/test_model_config.py).
_FACTORY_MODELS: tuple[ModelConfig, ...] = (
    # Base reference models, pinned to the revisions used elsewhere in experiments/models.py.
    _base_hf("llama-3.1-8b-base", "meta-llama/Llama-3.1-8B", "d04e592", 21),
    _base_hf("olmo-2-7b-base", "allenai/OLMo-2-1124-7B", "7df9a82", 18),
    # Qwen3.5-9B is a verbose hybrid-GDN reasoning model; its chains exceed the 8192-token chat default
    # and truncate before the boxed answer (OlympiadBench scored 0), so give it a 32k budget.
    ModelConfig(
        name="qwen3.5-9b",
        location="Qwen/Qwen3.5-9B",
        apply_chat_template=True,
        serve=ServeConfig(hbm_gb=24),
        generation=GenerationConfig(max_gen_toks=32768),
    ),
    # The single unified Qwen3-8B entry: evalchemy sizing (TPU-servable at hbm_gb=21) plus the agentic
    # serve/agent knobs the OT-Agent catalog carried for it (a hermes tool-call parser, and thinking
    # enabled for the agent). auto_serve_overrides derives the qwen3 reasoning parser.
    ModelConfig(
        name="qwen3-8b",
        location="Qwen/Qwen3-8B",
        apply_chat_template=True,
        serve=ServeConfig(hbm_gb=21, trust_remote_code=True, tool_call_parser="hermes"),
        agent=AgentConfig(agent_kwargs={"extra_body": '{"chat_template_kwargs":{"enable_thinking":true}}'}),
    ),
    ModelConfig(
        name="llama3.1-8b-instruct",
        location="meta-llama/Llama-3.1-8B-Instruct",
        apply_chat_template=True,
        serve=ServeConfig(hbm_gb=21),
    ),
    ModelConfig(
        name="olmo2-7b-instruct",
        location="allenai/OLMo-2-1124-7B-Instruct",
        apply_chat_template=True,
        serve=ServeConfig(hbm_gb=18),
    ),
    ModelConfig(name="qwen3-0.6b", location="Qwen/Qwen3-0.6B", apply_chat_template=True, serve=ServeConfig(hbm_gb=3)),
    ModelConfig(name="qwen3-1.7b", location="Qwen/Qwen3-1.7B", apply_chat_template=True, serve=ServeConfig(hbm_gb=5)),
    # The June pretrain cooldown export (the input to the SFT stages). Its tokenizer ships no chat
    # template and the delphi chat protocol is established by the SFT
    # (experiments/june_tpu_67b_a2b/moe/sft_67b_a2b_2stage.py), so messages-based evals serve the concat
    # template: a message list rendered as the raw text a base model expects.
    _snowball(
        "snowball",
        "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/",
        chat_template=CONCAT_CHAT_TEMPLATE,
    ),
    # The stage-2 (thinking) SFT of the same checkpoint. Its export ships a chat_template.jinja that vLLM
    # loads from the model directory, so no template override. Its <|start_think|> delimiters are special
    # tokens, so skip_special_tokens=false keeps the chain-of-thought from being stripped before scoring
    # (it otherwise scores 0), and a light repetition penalty curbs the thinking loops.
    _snowball(
        "snowball-sft",
        "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm/",
        generation=GenerationConfig(extra_gen_kwargs={"skip_special_tokens": "false", "repetition_penalty": "1.1"}),
    ),
)


def _build_registry() -> dict[str, ModelConfig]:
    """The full model registry: the Python factory entries plus the scanned YAML catalog.

    A ``name`` collision between a factory entry and a catalog file is an error -- one would silently
    shadow the other, and both paths produce the same ``ModelConfig`` type so nothing signals the clash.
    """
    registry = {model.name: model for model in _FACTORY_MODELS}
    for name, config in scan_model_configs(MODEL_CATALOG_DIR).items():
        if name in registry:
            raise ValueError(f"catalog model {name!r} collides with a Python factory entry of the same name")
        registry[name] = config
    return registry


MODELS: dict[str, ModelConfig] = _build_registry()
