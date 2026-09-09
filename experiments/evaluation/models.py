# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Models available to the evaluation launcher."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from marin.evaluation.model_config import (
    AgentConfig,
    GenerationConfig,
    ModelConfig,
    ResourceHint,
    ServeConfig,
    scan_model_configs,
)
from marin.inference.backend import CONCAT_CHAT_TEMPLATE
from rigging.filesystem.storage_path import prefix_join

MODEL_CATALOG_DIR = Path(__file__).parent / "serve" / "models"

# HF-format bf16 vLLM shards for the Snowball 67B-A2B stage-2 (thinking) SFT;
# the export ships its own chat_template.jinja and tokenizes with
# marin-community/marin-tokenizer.
SNOWBALL_SFT_EXPORT_URI = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm/"

# The 256-expert Grug MoE fork serves data-parallel + expert-parallel with tensor_parallel_size=1; the
# per-head TP heuristic cannot infer this, and the loader streams shards distributed across the ranks.
SNOWBALL_VLLM_ARGS = ("--enable-expert-parallel", "--model-loader-extra-config", '{"distributed":true}')
# vLLM's 92% default left only ~0.5 GiB free after allocating the Snowball KV cache.
# Real eval batches need more activation headroom than vLLM's startup profile observed.
SNOWBALL_FINAL_VLLM_ARGS = ("--enable-expert-parallel", "--gpu-memory-utilization", "0.85")
SNOWBALL_FINAL_VERSION = "2026.09.08.6"
SNOWBALL_FINAL_GENERATION = GenerationConfig(
    max_gen_toks=32768,
    extra_gen_kwargs={
        "temperature": "0.7",
        "top_p": "1.0",
        "repetition_penalty": "1.1",
        "skip_special_tokens": "false",
    },
)
SNOWBALL_FINAL_BASES: tuple[tuple[str, str, str], ...] = (
    (
        "qk157",
        "open-athena/snowball-67b-a2b-base-262k-qk157",
        "2b1f526273b8968b307a0098c08fb4321bb91e35",
    ),
    (
        "qk175",
        "open-athena/snowball-67b-a2b-base-262k-qk175",
        "1934e71f2bb0fbeb19e5ce82372136e5297bf0a4",
    ),
    (
        "qk175-skew2",
        "open-athena/snowball-67b-a2b-base-262k-qk175-skew2",
        "ce41c24df0afc10079210521ea7e231115ad5a92",
    ),
    (
        "qk175-skew4",
        "open-athena/snowball-67b-a2b-base-262k-qk175-skew4",
        "5052e68c4d88c9e0de87f7595a25ee4005aef1cf",
    ),
    (
        "qk175-skew8",
        "open-athena/snowball-67b-a2b-base-262k-qk175-skew8",
        "058ecaf27b9e4f37219df221a51e7d490d58ec3d",
    ),
)
SNOWBALL_FINAL_STAGES: tuple[tuple[str, int], ...] = (
    ("chat", 257),
    ("thinking", 630),
    ("opencode", 1888),
    ("nemotron-terminal", 1888),
)


def _snowball(
    name: str, location: str, *, chat_template: str | None = None, generation: GenerationConfig | None = None
) -> ModelConfig:
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
        resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            chat_template=chat_template,
            vllm_extra_args=SNOWBALL_VLLM_ARGS,
        ),
        generation=generation or GenerationConfig(),
    )


def _snowball_final(name: str, location: str, *, revision: str | None = None, base: bool = False) -> ModelConfig:
    """One checkpoint in the fixed five-base Snowball SFT evaluation matrix."""
    return ModelConfig(
        name=name,
        location=location,
        revision=revision,
        # Evalchemy's external runtime accepts an HF repository name but not ``repo@revision``.
        # The campaign separately records the validated tokenizer.json digest in its fixed policy.
        tokenizer="marin-community/marin-tokenizer",
        apply_chat_template=True,
        # The 39-shard checkpoint is larger than the generic 100 GB serve disk; model staging
        # otherwise exhausts the worker volume before vLLM can start.
        resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g", disk="512g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=65536,
            max_num_batched_tokens=7168,
            max_num_seqs=32,
            tool_call_parser="hermes",
            chat_template=CONCAT_CHAT_TEMPLATE if base else None,
            auto_overrides=False,
            # HF repositories are staged to a local directory before vLLM starts. The RunAI
            # streamer's ``distributed`` option is invalid with vLLM's resulting ``auto`` loader.
            vllm_extra_args=SNOWBALL_FINAL_VLLM_ARGS,
        ),
        generation=SNOWBALL_FINAL_GENERATION,
    )


def _snowball_final_models() -> tuple[ModelConfig, ...]:
    models: list[ModelConfig] = []
    prefix = "s3://marin-us-east-02a/marin/snowball-final"
    for base, repository, revision in SNOWBALL_FINAL_BASES:
        models.append(
            _snowball_final(
                f"snowball-final-{base}-base",
                repository,
                revision=revision,
                base=True,
            )
        )
        for stage, step in SNOWBALL_FINAL_STAGES:
            suffix = f"{base}/{stage}/{SNOWBALL_FINAL_VERSION}/hf/step-{step}"
            location = prefix_join(prefix, suffix)
            models.append(_snowball_final(f"snowball-final-{base}-{stage}", location))
    return tuple(models)


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
        resource_hint=ResourceHint(hbm_gb=hbm_gb),
    )


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
        resource_hint=ResourceHint(hbm_gb=24),
        generation=GenerationConfig(max_gen_toks=32768),
    ),
    ModelConfig(
        name="qwen3-8b",
        location="Qwen/Qwen3-8B",
        apply_chat_template=True,
        resource_hint=ResourceHint(hbm_gb=21),
        serve=ServeConfig(tool_call_parser="hermes"),
        agent=AgentConfig(agent_kwargs={"extra_body": '{"chat_template_kwargs":{"enable_thinking":true}}'}),
    ),
    ModelConfig(
        name="llama3.1-8b-instruct",
        location="meta-llama/Llama-3.1-8B-Instruct",
        apply_chat_template=True,
        resource_hint=ResourceHint(hbm_gb=21),
    ),
    ModelConfig(
        name="olmo2-7b-instruct",
        location="allenai/OLMo-2-1124-7B-Instruct",
        apply_chat_template=True,
        resource_hint=ResourceHint(hbm_gb=18),
    ),
    ModelConfig(
        name="qwen3-0.6b",
        location="Qwen/Qwen3-0.6B",
        apply_chat_template=True,
        resource_hint=ResourceHint(hbm_gb=3),
    ),
    ModelConfig(
        name="qwen3-1.7b",
        location="Qwen/Qwen3-1.7B",
        apply_chat_template=True,
        resource_hint=ResourceHint(hbm_gb=5),
    ),
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
        SNOWBALL_SFT_EXPORT_URI,
        generation=GenerationConfig(extra_gen_kwargs={"skip_special_tokens": "false", "repetition_penalty": "1.1"}),
    ),
    *_snowball_final_models(),
)


def _build_registry() -> dict[str, ModelConfig]:
    registry = {model.name: model for model in _FACTORY_MODELS}
    for name, config in scan_model_configs(MODEL_CATALOG_DIR).items():
        if name in registry:
            raise ValueError(f"catalog model {name!r} collides with a Python factory entry of the same name")
        registry[name] = config
    return registry


@cache
def _cached_models() -> dict[str, ModelConfig]:
    return _build_registry()


def models() -> dict[str, ModelConfig]:
    """Return the registered evaluation models keyed by launcher name."""
    return dict(_cached_models())
