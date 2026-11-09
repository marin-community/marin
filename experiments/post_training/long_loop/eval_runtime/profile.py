# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen inputs shared by Micro210 preparation, budgeting and execution."""

DATASET = "open-athena/Marin-Micro-Eval-210"
SOURCE_REVISION = "4564ed081c316bc3231ae5c3a53a46b39248c819"
REVISION = SOURCE_REVISION
DESTINATION = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/e2e-20261031"
SEED = 20261030
TOTAL_CONTEXT_LENGTH = 65536
MAX_OUTPUT_TOKENS = 16384
MAX_INPUT_TOKENS = TOTAL_CONTEXT_LENGTH - MAX_OUTPUT_TOKENS
AGENT_TIMEOUT = 1800
VERIFIER_TIMEOUT = 300
SETUP_TIMEOUT = 360
CLEANUP_TIMEOUT = 60
ATTEMPT_TIMEOUT = SETUP_TIMEOUT + AGENT_TIMEOUT + VERIFIER_TIMEOUT
MODEL = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
COUNTERPART_MODEL = "openai/openai/gpt-oss-120b"

POLICY_INDEX = "open-athena/Marin-Mini-Eval-Policy-Index"
POLICY_INDEX_REVISION = "d5d2f92ce4866c431063bd123ea31a3865530809"


def pi_agent_config() -> dict:
    return {
        "name": "pi",
        "override_setup_timeout_sec": SETUP_TIMEOUT,
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
        # LiteLLM consumes the first openai/ as its provider prefix. Together's
        # actual model ID retains the second openai/ in the HTTP request.
        "TAU2_USER_MODEL": COUNTERPART_MODEL,
        "TAU2_NL_ASSERTIONS_MODEL": COUNTERPART_MODEL,
        "TAU2_USER_REASONING_EFFORT": "low",
    }


# Temporary user-authorized smoke exclusion; preserve the full release unchanged.
SKIPPED_BENCHMARKS = {"tau3-pi": "Unsupported Compose/network contract; marin issue #11083"}
SMOKE_PATH_COUNT = 21 - len(SKIPPED_BENCHMARKS)
