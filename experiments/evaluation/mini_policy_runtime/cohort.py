# Copyright The Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-model serving bindings for Mini840 evaluation runs.

Each entry binds one evaluated checkpoint to the tokenizer, chat-template
behavior and tool parser its family needs on the pinned Marin vLLM wheel.
Qwen3.6 stays in the registry because the validated runtime candidate and the
E2E instructions launch through the same code path.

Serving bindings follow the frozen policy model configs
(``eval-campaign-09-25-unlabeled/model-configs``): every family keeps its
catalog tool parser and no reasoning parser, matching the validated Qwen3.6
execution path. The Snowball 67B-A2B chat template carries no thinking kwarg;
Pi's ``thinking_format: chat-template`` drives thinking for that family.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CohortModel:
    """Serving bindings for one evaluated checkpoint."""

    key: str
    repo: str
    revision: str
    tool_parser: str
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
        chat_template_kwargs={"enable_thinking": True},
    ),
    "nemotron3-nano-30b-a3b": CohortModel(
        key="nemotron3-nano-30b-a3b",
        repo="open-athena/Nemotron-3-Nano-30B-A3B-Chat-Aligned-Base-2026.10.30",
        revision="82661839330fb2a3a6498a5af08a76f774c0191e",
        tool_parser="qwen3_coder",
        chat_template_kwargs={"enable_thinking": True},
    ),
    "snowball-67b-a2b": CohortModel(
        key="snowball-67b-a2b",
        repo="open-athena/Snowball-67B-A2B-Chat-Aligned-Base-2026.10.22",
        revision="72ef8bf91f13a4d384ec3efc889ea90856510d61",
        tool_parser="hermes",
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
    return {"enable_auto_tool_choice": True, "tool_call_parser": model.tool_parser}
