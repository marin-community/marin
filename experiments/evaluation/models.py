# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The model registry for the eval launcher.

Each :class:`EvalModelConfig` names a model, where its weights live, how much HBM serving it needs, and
any serving specifics (extra vLLM flags, a pinned GPU shape). :mod:`experiments.evaluation.hardware`
turns ``hbm_gb`` into a slice; :mod:`experiments.evaluation.launch` turns the rest into a
``ServeSpec``. Sizes follow the bf16 rule of thumb: ``params_billions * 2 GB * ~1.3`` for weights plus
runtime overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.evals.evalchemy.serve_and_eval import ServeBackend

# Snowball is the June 67B-A2B Grug MoE export; serving it needs the marin vLLM fork's data-parallel +
# expert-parallel path (the GPU serve path always uses the fork). The tokenizer is an HF id because the
# eval client cannot load a tokenizer from the s3:// export.
SNOWBALL_EXPORT = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/781bc3291c81ce28/"
SNOWBALL_TOKENIZER = "marin-community/marin-tokenizer"
SNOWBALL_CLUSTER = "cw-us-east-02a"
# Data-parallel + expert-parallel sharding for the 256-expert MoE, with tensor_parallel_size=1. The
# per-head TP heuristic cannot infer this, so it is passed verbatim (see ServeSpec.vllm_extra_args).
SNOWBALL_VLLM_ARGS = (
    "--data-parallel-size",
    "8",
    "--enable-expert-parallel",
    "--model-loader-extra-config",
    '{"distributed":true}',
)
# A plain concatenation template: the marin tokenizer defines none (base-flavored model), and
# messages-based evals (math500) need the chat endpoint to accept requests -- for a base model the
# faithful rendering of a message list is its raw text.
SNOWBALL_CHAT_TEMPLATE = "{%- for message in messages -%}{{ message['content'] }}\n\n{%- endfor -%}"


@dataclass(frozen=True)
class EvalModelConfig:
    """A model the launcher can serve and evaluate.

    ``location`` is an HF repo id or an object-store (``gs://``/``s3://``) HF-format export directory;
    an object-store location requires ``tokenizer`` (the eval client loads its tokenizer through HF).
    ``hbm_gb`` is the serving HBM budget used to size a slice. ``fixed_gpu`` pins an exact GPU
    type/count (bypassing the sizing heuristic), and ``target_cluster`` names the CoreWeave peer the
    GPU job routes to.
    """

    name: str
    location: str
    hbm_gb: int
    apply_chat_template: bool
    backend: ServeBackend = ServeBackend.VLLM
    gpu_only: bool = False
    vllm_extra_args: tuple[str, ...] = ()
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    tokenizer: str | None = None
    fixed_gpu: tuple[str, int] | None = None
    target_cluster: str | None = None
    serve_memory: str | None = None
    """Host-memory request for the serve child, overriding the ``ServeSpec`` default. Large
    object-store exports need it: weight streaming stages shards through host buffers, so the
    pod's memory limit must cover the full weight volume or the kernel OOM-kills the server."""
    chat_template: str | None = None
    """A jinja chat template served in place of the tokenizer's own (``ServeSpec.chat_template_content``),
    for models whose tokenizer ships none."""


MODELS: dict[str, EvalModelConfig] = {
    "qwen3.5-9b": EvalModelConfig(
        name="qwen3.5-9b",
        location="Qwen/Qwen3.5-9B",
        hbm_gb=24,
        apply_chat_template=True,
    ),
    "qwen3-8b": EvalModelConfig(
        name="qwen3-8b",
        location="Qwen/Qwen3-8B",
        hbm_gb=21,
        apply_chat_template=True,
    ),
    "llama3.1-8b-instruct": EvalModelConfig(
        name="llama3.1-8b-instruct",
        location="meta-llama/Llama-3.1-8B-Instruct",
        hbm_gb=21,
        apply_chat_template=True,
    ),
    "olmo2-7b-instruct": EvalModelConfig(
        name="olmo2-7b-instruct",
        location="allenai/OLMo-2-1124-7B-Instruct",
        hbm_gb=18,
        apply_chat_template=True,
    ),
    "qwen3-1.7b": EvalModelConfig(
        name="qwen3-1.7b",
        location="Qwen/Qwen3-1.7B",
        hbm_gb=5,
        apply_chat_template=True,
    ),
    "snowball": EvalModelConfig(
        name="snowball",
        location=SNOWBALL_EXPORT,
        hbm_gb=175,
        # The concat chat template renders messages as raw text, so the chat route degrades to a
        # plain completion for this base-flavored model (messages-based evals need the route).
        apply_chat_template=True,
        gpu_only=True,
        vllm_extra_args=SNOWBALL_VLLM_ARGS,
        tensor_parallel_size=1,
        tokenizer=SNOWBALL_TOKENIZER,
        fixed_gpu=("H100", 8),
        target_cluster=SNOWBALL_CLUSTER,
        # ~134GB of bf16 shards stream from object storage through host buffers on load; the
        # serve pod owns the whole 8xH100 node, so a generous limit costs nothing.
        serve_memory="512g",
        chat_template=SNOWBALL_CHAT_TEMPLATE,
    ),
}
