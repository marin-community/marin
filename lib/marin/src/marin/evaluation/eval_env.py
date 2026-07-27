# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allowlisted launch-environment variables supplied to evaluation jobs."""

import os

# Credentials and W&B metadata propagated from an eval orchestrator into its serve and mechanism children.
EVAL_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "HF_TOKEN",
    # RunAI object-store tuning. These are allowlisted so a prefix assignment on the evaluation
    # launcher reaches the remote inference child.
    "RUNAI_STREAMER_CHUNK_BYTESIZE",
    "RUNAI_STREAMER_CONCURRENCY",
    "RUNAI_STREAMER_LOG_LEVEL",
    "RUNAI_STREAMER_LOG_TO_STDERR",
    "RUNAI_STREAMER_S3_MAX_INFLIGHT_MIB",
    "RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS",
    "RUNAI_STREAMER_S3_TRACE",
)

# Forwarded verbatim from the launch environment into an eval job (present ones only).
EVAL_ENV_KEYS: tuple[str, ...] = (
    *EVAL_RUNTIME_ENV_KEYS,
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "E2B_API_KEY",
    "MODAL_API_KEY",
    "TPU_CI",
    "MARIN_PREFIX",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION",
    "VLLM_TPU_SKIP_PRECOMPILE",
)


def env_vars_from_keys(keys: tuple[str, ...]) -> dict[str, str]:
    """The subset of ``keys`` present in the launch environment, as an env-var mapping."""
    return {key: os.environ[key] for key in keys if os.environ.get(key)}
