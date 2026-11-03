# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-model serving bindings for Mini840 evaluation runs.

Each entry binds one evaluated checkpoint to the tokenizer, chat-template
behavior, tool parser, context length and service topology its family needs on
the pinned Marin vLLM wheel. Qwen3.6 stays in the registry because the runtime
candidate in these instructions launches through the same code path.

Serving bindings follow each checkpoint's own training configuration:

- Tool parser: the chat-aligned bases emit tool calls in their training
  template's wire format. Qwen3.5/3.6 keep ``qwen3_coder`` (the validated
  execution path); Qwen3-30B, Nemotron and Snowball emit hermes-style JSON and
  use ``hermes``, matching the frozen policy model configs for those families.
- Context length: the uniform profile requests 65,536 tokens; when a model's
  native context is smaller, the policy adopts the model's maximum (Qwen3-30B
  is natively 32,768).
- Snowball 67B-A2B (134 GB BF16) serves one TP1 engine with DP4 expert
  parallelism over four H100s: the fork's GrugMoE GPU path requires tensor
  parallelism 1 and shards experts over the data-parallel group instead, which
  is how the campaign's TP1 Grug serving configs fit the same checkpoint
  family. Four DP ranks keep each GPU near 36 GB of weights with ample KV.
- ``chat_template_file`` prepends a tools system section to a checkpoint whose
  training template drops the API ``tools`` array (Snowball, Nemotron). The
  vendored files are the checkpoints' exact training templates plus that
  section; the Qwen families' own templates already render tools.
- Provenance arms added 2026-11-01: ``snowball-67b-a2b-sft8225`` is the #8225
  final Nemotron-Terminal export (step 1,888 on the 5.7T cooldown), pinned at
  the pipeline's own HF upload (config-identical to the S3 export at
  ``marin-us-east-02a``) so the standard manifest-writing mirror path applies;
  its embedded template renders ``tools``, so no vendored file and no
  reasoning kwarg (its training template has no reasoning modes;
  ``max_position_embeddings`` is 32,768, so the policy adopts that as the
  context). The raw ``Qwen3-30B-A3B-Base`` and ``Qwen3.5-35B-A3B-Base`` bases
  run with no additional training; their parsers match their families'
  validated paths even though a raw base rarely emits parseable calls.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CohortModel:
    """Serving bindings for one evaluated checkpoint."""

    key: str
    repo: str
    revision: str
    tool_parser: str
    context_length: int = 65536
    gpus_per_node: int = 2
    tensor_parallel_size: int = 2
    num_inference_engines: int = 1
    inference_engine_data_parallel_size: int = 1
    expert_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 8192
    chat_template_file: str | None = None
    chat_template_kwargs: dict | None = None
    # "hub" stages tokenizer metadata from the pinned Hub revision; "model-artifact"
    # reads it from the object-store mirror, which is what a gated checkpoint needs
    # because the evaluation containers carry no Hub credentials.
    tokenizer_source: Literal["hub", "model-artifact"] = "hub"


COHORT_MODELS: dict[str, CohortModel] = {
    "qwen36-35b-a3b": CohortModel(
        key="qwen36-35b-a3b",
        repo="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        tool_parser="qwen3_coder",
        chat_template_kwargs={"enable_thinking": True},
    ),
    "qwen35-35b-a3b": CohortModel(
        key="qwen35-35b-a3b",
        repo="open-athena/Qwen3.5-35B-A3B-Chat-Aligned-Base-2026.10.31",
        revision="05f2dfeb06fb0c0a5bf5bad93ed633d1f756dcb7",
        tool_parser="qwen3_coder",
        chat_template_kwargs={"enable_thinking": True},
    ),
    "qwen3-30b-a3b": CohortModel(
        key="qwen3-30b-a3b",
        repo="open-athena/Qwen3-30B-A3B-Chat-Aligned-Base-2026.10.30",
        revision="8e5812791d78603ea84f5cfb89a5ccd5f3e64a63",
        tool_parser="hermes",
        context_length=32768,
        chat_template_kwargs={"enable_thinking": True},
    ),
    "nemotron3-nano-30b-a3b": CohortModel(
        key="nemotron3-nano-30b-a3b",
        repo="open-athena/Nemotron-3-Nano-30B-A3B-Chat-Aligned-Base-2026.10.30",
        revision="82661839330fb2a3a6498a5af08a76f774c0191e",
        tool_parser="hermes",
        chat_template_file="nemotron3-nano-30b-a3b.jinja",
        chat_template_kwargs={"enable_thinking": True},
    ),
    "snowball-67b-a2b": CohortModel(
        key="snowball-67b-a2b",
        repo="open-athena/Snowball-67B-A2B-Chat-Aligned-Base-2026.10.22",
        revision="72ef8bf91f13a4d384ec3efc889ea90856510d61",
        tool_parser="hermes",
        gpus_per_node=4,
        tensor_parallel_size=1,
        num_inference_engines=1,
        inference_engine_data_parallel_size=4,
        expert_parallel_size=4,
        gpu_memory_utilization=0.92,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        chat_template_file="snowball-67b-a2b.jinja",
    ),
    "snowball-67b-a2b-sft8225": CohortModel(
        key="snowball-67b-a2b-sft8225",
        repo="laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888",
        revision="680bec3ade494b61ead6861fa24fc4071f8dc294",
        tool_parser="hermes",
        context_length=32768,
        gpus_per_node=4,
        tensor_parallel_size=1,
        num_inference_engines=1,
        inference_engine_data_parallel_size=4,
        expert_parallel_size=4,
        gpu_memory_utilization=0.92,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
    ),
    "qwen3-30b-a3b-base": CohortModel(
        key="qwen3-30b-a3b-base",
        repo="Qwen/Qwen3-30B-A3B-Base",
        revision="1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        tool_parser="hermes",
        context_length=32768,
        chat_template_kwargs={"enable_thinking": True},
    ),
    "qwen35-35b-a3b-base": CohortModel(
        key="qwen35-35b-a3b-base",
        repo="Qwen/Qwen3.5-35B-A3B-Base",
        revision="0f0813072d2358973511097385626f21fcb6d422",
        tool_parser="qwen3_coder",
        chat_template_kwargs={"enable_thinking": True},
    ),
    "grug-67b-a2b-datakit-sft": CohortModel(
        key="grug-67b-a2b-datakit-sft",
        repo="open-athena/Grug-67B-A2B-Datakit-SFT-Repro-262K-Step471-2026.10.25",
        revision="51669ee19b8972fc2e74c10b39eaef2eb244c44a",
        tool_parser="hermes",
        tokenizer_source="model-artifact",
        gpus_per_node=4,
        tensor_parallel_size=1,
        num_inference_engines=1,
        inference_engine_data_parallel_size=4,
        expert_parallel_size=4,
        gpu_memory_utilization=0.92,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
    ),
}

CHAT_ALIGNED_BASE_MODELS = (
    "qwen35-35b-a3b",
    "qwen3-30b-a3b",
    "nemotron3-nano-30b-a3b",
    "snowball-67b-a2b",
)


def cohort_model(key: str) -> CohortModel:
    """Resolve a registry key to its bindings; unknown keys must fail before staging."""
    try:
        return COHORT_MODELS[key]
    except KeyError:
        raise ValueError(f"Unknown cohort model {key!r}; expected one of {sorted(COHORT_MODELS)}") from None


def engine_init_kwargs(model: CohortModel) -> dict:
    """SkyRL generator engine arguments bound to the model's parser bindings.

    SkyRL sets the actor-level engine kwargs (expert parallel, concurrency,
    prefill budget) itself from the generator config, so none are passed here.
    """
    return {"enable_auto_tool_choice": True, "tool_call_parser": model.tool_parser}


def chat_template_config(model: CohortModel) -> dict:
    """SkyRL generator chat-template configuration for the model."""
    if model.chat_template_file is None:
        return {"source": "name", "name_or_path": None}
    # The evaluation container mounts the repository at /app.
    path = f"/app/experiments/evaluation/mini_policy_runtime/chat-templates/{model.chat_template_file}"
    return {"source": "file", "name_or_path": path}
