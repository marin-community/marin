# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch-environment variables forwarded to evaluation jobs.

An eval job (the group orchestrator, an evalchemy client, a Harbor runner) runs on a cluster worker
with none of the launcher's ambient credentials. The launcher forwards a fixed allowlist of variables
from its own environment into the job's :class:`EnvironmentSpec`; this module is that allowlist.
"""

import os

# Credentials and W&B metadata propagated from an eval orchestrator into its serve and mechanism children.
EVAL_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "HF_TOKEN",
)

# Forwarded verbatim from the launch environment into an eval job (present ones only).
EVAL_ENV_KEYS: tuple[str, ...] = (
    *EVAL_RUNTIME_ENV_KEYS,
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "E2B_API_KEY",
    "MODAL_API_KEY",
    "DAYTONA_API_KEY",
    "TPU_CI",
    "MARIN_PREFIX",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION",
    "VLLM_TPU_SKIP_PRECOMPILE",
)


def env_vars_from_keys(keys: tuple[str, ...]) -> dict[str, str]:
    """The subset of ``keys`` present in the launch environment, as an env-var mapping."""
    return {key: os.environ[key] for key in keys if os.environ.get(key)}
