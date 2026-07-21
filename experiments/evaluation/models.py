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

from marin.inference.serving_backend import CONCAT_CHAT_TEMPLATE

from experiments.evals.evalchemy.serve_and_eval import ServeBackend


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
    max_gen_toks: int | None = None
    """Per-model override of a suite's generation budget. A verbose reasoning model needs a longer
    budget than the suite default or its chain truncates before the final answer (scoring it wrong)."""
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


def _snowball(name: str, location: str, chat_template: str | None = None) -> EvalModelConfig:
    """A Grug 67B-A2B export served on a CoreWeave 8xH100 node via the marin vLLM fork.

    Data-parallel + expert-parallel sharding for the 256-expert MoE with ``tensor_parallel_size=1``
    (the per-head TP heuristic cannot infer this). The tokenizer is an HF id because the eval client
    cannot load a tokenizer from the s3:// export; ~134GB of bf16 shards stream from object storage
    through host buffers on load, so the serve child gets a generous memory limit (it owns the node).
    """
    return EvalModelConfig(
        name=name,
        location=location,
        hbm_gb=175,
        apply_chat_template=True,
        gpu_only=True,
        vllm_extra_args=(
            "--data-parallel-size",
            "8",
            "--enable-expert-parallel",
            "--model-loader-extra-config",
            '{"distributed":true}',
        ),
        tensor_parallel_size=1,
        tokenizer="marin-community/marin-tokenizer",
        fixed_gpu=("H100", 8),
        target_cluster="cw-us-east-02a",
        serve_memory="512g",
        chat_template=chat_template,
    )


def _base_hf(name: str, location: str, revision: str, hbm_gb: int) -> EvalModelConfig:
    """A base (non-chat) HF model, pinned to an immutable revision.

    ``apply_chat_template=False`` (base models ship no chat template), so these run the NLP
    (lm-eval) suite, not the chat benchmarks. The revision is pinned through ``vllm serve
    --revision`` so results are reproducible against a fixed checkpoint rather than the HF branch head.
    """
    return EvalModelConfig(
        name=name,
        location=location,
        hbm_gb=hbm_gb,
        apply_chat_template=False,
        vllm_extra_args=("--revision", revision),
    )


MODELS: dict[str, EvalModelConfig] = {
    # Base reference models, pinned to the revisions used elsewhere in experiments/models.py.
    "llama-3.1-8b-base": _base_hf("llama-3.1-8b-base", "meta-llama/Llama-3.1-8B", "d04e592", 21),
    "olmo-2-7b-base": _base_hf("olmo-2-7b-base", "allenai/OLMo-2-1124-7B", "7df9a82", 18),
    # Qwen3.5-9B is a verbose hybrid-GDN reasoning model; its chains exceed the 8192-token chat
    # default and truncate before the boxed answer (OlympiadBench scored 0), so give it a 32k budget.
    "qwen3.5-9b": EvalModelConfig(
        name="qwen3.5-9b",
        location="Qwen/Qwen3.5-9B",
        hbm_gb=24,
        apply_chat_template=True,
        max_gen_toks=32768,
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
    # The June pretrain cooldown export (the input to the SFT stages). Its tokenizer ships no chat
    # template and the delphi chat protocol is established by the SFT
    # (experiments/june_tpu_67b_a2b/moe/sft_67b_a2b_2stage.py), so messages-based evals serve the
    # concat template: a message list rendered as the raw text a base model expects.
    "snowball": _snowball(
        "snowball",
        "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/781bc3291c81ce28/",
        chat_template=CONCAT_CHAT_TEMPLATE,
    ),
    # The stage-2 (thinking) SFT of the same checkpoint. Its export ships a chat_template.jinja that
    # vLLM loads from the model directory, so no template override.
    "snowball-sft": _snowball(
        "snowball-sft",
        "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm/",
    ),
}
