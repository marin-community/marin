# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen inputs shared by Mini840 preparation, budgeting and execution."""

DATASET = "open-athena/Marin-Mini-Eval-840"
SOURCE_REVISION = "bae388a44fc0ba9ffa551a4c6347fbe03a9fefd7"
REVISION = SOURCE_REVISION
DESTINATION = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/e2e-20261031"
SEED = 20261030
TOTAL_CONTEXT_LENGTH = 65536
MAX_OUTPUT_TOKENS = 16384
MAX_INPUT_TOKENS = TOTAL_CONTEXT_LENGTH - MAX_OUTPUT_TOKENS
MODEL = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"

POLICY_INDEX = "open-athena/Marin-Mini-Eval-Policy-Index"
POLICY_INDEX_REVISION = "d5d2f92ce4866c431063bd123ea31a3865530809"


def pi_agent_config() -> dict:
    return {
        "name": "pi",
        "override_setup_timeout_sec": 360,
        "kwargs": {
            "version": "0.87.0",
            "mcp_adapter_version": "7.0.0",
            "thinking": "high",
            "thinking_format": "chat-template",
            "model_info": {"max_input_tokens": TOTAL_CONTEXT_LENGTH, "max_output_tokens": MAX_OUTPUT_TOKENS},
        },
    }


def counterpart_environment() -> dict[str, str]:
    return {
        "OPENAI_BASE_URL": "https://api.together.xyz/v1",
        "TAU2_USER_MODEL": "openai/gpt-oss-120b",
        "TAU2_NL_ASSERTIONS_MODEL": "openai/gpt-oss-120b",
        "TAU2_USER_REASONING_EFFORT": "low",
    }
