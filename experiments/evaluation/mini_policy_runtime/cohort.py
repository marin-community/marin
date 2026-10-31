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
- Snowball 67B-A2B serves TP1 because the Marin vLLM fork's GrugMoE GPU path
  requires tensor parallelism 1; two TP1 engines cover the two allocated H100s,
  with memory utilization raised for KV headroom.
- ``chat_template_file`` prepends a tools system section to a checkpoint whose
  training template drops the API ``tools`` array (Snowball, Nemotron). The
  vendored files are the checkpoints' exact training templates plus that
  section; the Qwen families' own templates already render tools.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CohortModel:
    """Serving bindings for one evaluated checkpoint."""

    key: str
    repo: str
    revision: str
    tool_parser: str
    context_length: int = 65536
    tensor_parallel_size: int = 2
    num_inference_engines: int = 1
    gpu_memory_utilization: float = 0.85
    expert_parallel: bool = False
    chat_template_file: str | None = None
    chat_template_kwargs: dict | None = None


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
        tensor_parallel_size=1,
        num_inference_engines=2,
        gpu_memory_utilization=0.95,
        expert_parallel=True,
        chat_template_file="snowball-67b-a2b.jinja",
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
        raise ValueError(
            f"Unknown cohort model {key!r}; expected one of {sorted(COHORT_MODELS)}"
        ) from None


def engine_init_kwargs(model: CohortModel) -> dict:
    """SkyRL generator engine arguments bound to the model's parser bindings."""
    kwargs = {"enable_auto_tool_choice": True, "tool_call_parser": model.tool_parser}
    if model.expert_parallel:
        kwargs["enable_expert_parallel"] = True
    return kwargs


def chat_template_config(model: CohortModel) -> dict:
    """SkyRL generator chat-template configuration for the model."""
    if model.chat_template_file is None:
        return {"source": "name", "name_or_path": None}
    # The evaluation container mounts the repository at /app.
    path = f"/app/experiments/evaluation/mini_policy_runtime/chat-templates/{model.chat_template_file}"
    return {"source": "file", "name_or_path": path}
