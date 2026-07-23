# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Which launch-environment variables an eval job needs, and how they reach it.

An eval job (the group orchestrator, an evalchemy client, a Harbor runner) runs on a cluster worker
with none of the launcher's ambient credentials. The launcher forwards a fixed allowlist of variables
from its own environment into the job's :class:`EnvironmentSpec`; this module is the single definition
of that allowlist plus the one credential that needs renaming on the way in.

The Daytona SDK reads ``DAYTONA_API_KEY``, but the eval key is stored in Google Secret Manager (and so
in the launch environment) under ``DAYTONA_EVAL_API_KEY`` to keep it distinct from the data/RL keys.
:func:`daytona_sdk_env` bridges the two names so the control-plane credential is injected under the
name the SDK expects, rather than shipping the raw GSM name into every sandbox.
"""

import os

# Forwarded verbatim from the launch environment into an eval job (present ones only).
EVAL_ENV_KEYS: tuple[str, ...] = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "HF_TOKEN",
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

# The Google Secret Manager / launch-env name for the Daytona eval key.
DAYTONA_EVAL_API_KEY_ENV = "DAYTONA_EVAL_API_KEY"
# The name the Daytona SDK itself reads.
DAYTONA_SDK_API_KEY_ENV = "DAYTONA_API_KEY"


def env_vars_from_keys(keys: tuple[str, ...]) -> dict[str, str]:
    """The subset of ``keys`` present in the launch environment, as an env-var mapping."""
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def daytona_sdk_env() -> dict[str, str]:
    """Map the launch env's ``DAYTONA_EVAL_API_KEY`` to the ``DAYTONA_API_KEY`` the Daytona SDK reads.

    Empty when the eval key is absent, so an evalchemy-only launch never requires it; a
    Harbor-on-Daytona launch checks for it up front (see :func:`daytona_eval_api_key`).
    """
    key = os.environ.get(DAYTONA_EVAL_API_KEY_ENV)
    return {DAYTONA_SDK_API_KEY_ENV: key} if key else {}


def daytona_eval_api_key() -> str:
    """The Daytona eval key, or a clear error naming the GSM secret to load into the launch env."""
    key = os.environ.get(DAYTONA_EVAL_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{DAYTONA_EVAL_API_KEY_ENV} is not set; load the Google Secret Manager secret of that name "
            "into the launch environment before launching a Harbor-on-Daytona eval."
        )
    return key
