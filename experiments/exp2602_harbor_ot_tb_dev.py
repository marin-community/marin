# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Harbor OpenThoughts-TB-dev (terminus-2 + Qwen3-8B) (dev)

This is a smoke-test experiment for running OpenThoughts' Terminal-Bench dev split
through Marin's Harbor evaluator.

Tracked in: https://github.com/marin-community/marin/issues/2602
Based on: https://github.com/marin-community/marin/issues/2536#issuecomment-3838665091

Usage (Ray Cluster with Daytona):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars MARIN_PREFIX gs://marin-us-central1 \
        --env_vars DAYTONA_API_KEY ${DAYTONA_API_KEY} \
        --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT harbor \
        --env_vars HF_TOKEN ${HF_TOKEN} \
        --cluster us-central1 \
        --extra harbor,vllm \
        --no_wait \
        -- python experiments/exp2602_harbor_ot_tb_dev.py

Environment variables:
    - ENV_TYPE: "local" | "daytona" | "e2b" | "modal" (default: "daytona")
    - HARBOR_DATASET: Harbor dataset name or HF dataset id (default: open-thoughts/OpenThoughts-TB-dev)
    - HARBOR_DATASET_VERSION: Harbor version label (default: hf)
    - HARBOR_AGENT: Harbor agent name (default: "terminus-2")
    - HARBOR_MODEL_NAME: model identifier (default: "qwen3-8b")
    - HARBOR_MODEL_PATH: model path (default: gs://marin-us-central1/models/Qwen--Qwen3-8B--main/)
    - HARBOR_MAX_INSTANCES: number of tasks to run (default: 70)
    - HARBOR_N_CONCURRENT: number of parallel trials (default: 25)
    - HARBOR_AGENT_KWARGS_JSON: JSON dict forwarded to Harbor AgentConfig.kwargs
"""

import json
import logging
import os

from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _escape_for_path(value: str) -> str:
    return value.replace("://", "__").replace("/", "__")


def _optional_int_from_env(var_name: str, *, default: int | None) -> int | None:
    value = os.environ.get(var_name)
    if value is None:
        return default

    if value.strip().lower() in {"none", "null", "all"}:
        return None

    return int(value)


def _optional_str_from_env(var_name: str, *, default: str | None) -> str | None:
    value = os.environ.get(var_name)
    if value is None:
        return default

    stripped = value.strip()
    if not stripped:
        return None

    if stripped.lower() in {"none", "null"}:
        return None

    return stripped


def _optional_json_dict_from_env(var_name: str) -> dict:
    value = os.environ.get(var_name)
    if value is None or not value.strip():
        return {}

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"{var_name} must be a JSON object (dict), got {type(parsed)}")
    return parsed


DATASET = os.environ.get("HARBOR_DATASET", "open-thoughts/OpenThoughts-TB-dev")
VERSION = os.environ.get("HARBOR_DATASET_VERSION", "hf")

ENV_TYPE = os.environ.get("ENV_TYPE", "daytona")
HARBOR_AGENT = os.environ.get("HARBOR_AGENT", "terminus-2")

MODEL_NAME = os.environ.get("HARBOR_MODEL_NAME", "qwen3-8b")
MODEL_PATH = _optional_str_from_env(
    "HARBOR_MODEL_PATH",
    default="gs://marin-us-central1/models/Qwen--Qwen3-8B--main/",
)

# TPU jobs typically do not have Docker available; use native vLLM by default.
if MODEL_PATH:
    os.environ.setdefault("MARIN_VLLM_MODE", "native")

HARBOR_MAX_INSTANCES = _optional_int_from_env("HARBOR_MAX_INSTANCES", default=70)
HARBOR_N_CONCURRENT = _optional_int_from_env("HARBOR_N_CONCURRENT", default=25) or 1
HARBOR_AGENT_KWARGS = _optional_json_dict_from_env("HARBOR_AGENT_KWARGS_JSON")

DATASET_ESCAPED = _escape_for_path(DATASET)
MODEL_NAME_ESCAPED = _escape_for_path(MODEL_NAME)
OUTPUT_DIR = os.path.join("evaluation", "harbor", DATASET_ESCAPED, MODEL_NAME_ESCAPED, HARBOR_AGENT)

if ENV_TYPE == "daytona" and not os.environ.get("DAYTONA_API_KEY"):
    logger.warning(
        "DAYTONA_API_KEY not set but ENV_TYPE=daytona! "
        "The evaluation will likely fail. Set DAYTONA_API_KEY or use ENV_TYPE=local for Docker."
    )

if MODEL_PATH is None and MODEL_NAME.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not set but MODEL_NAME=%s and MODEL_PATH=None.", MODEL_NAME)

logger.info("=" * 80)
logger.info("Harbor OpenThoughts-TB-dev (terminus-2 + Qwen3-8B) (dev)")
logger.info("=" * 80)
logger.info("Dataset: %s@%s", DATASET, VERSION)
logger.info("Model: %s", MODEL_NAME)
logger.info("Agent: %s", HARBOR_AGENT)
logger.info("Env: %s", ENV_TYPE)
logger.info("Max tasks: %s", HARBOR_MAX_INSTANCES)
logger.info("Concurrent trials: %s", HARBOR_N_CONCURRENT)
logger.info("Output dir (under MARIN_PREFIX): %s", OUTPUT_DIR)
logger.info("=" * 80)


if __name__ == "__main__":
    resource_config = ResourceConfig.with_tpu("v5p-8") if MODEL_PATH else ResourceConfig.with_cpu()
    step = evaluate_harbor(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        dataset=DATASET,
        version=VERSION,
        max_eval_instances=HARBOR_MAX_INSTANCES,
        resource_config=resource_config,
        wandb_tags=["harbor", DATASET, "dev", HARBOR_AGENT, ENV_TYPE, MODEL_NAME, "vllm" if MODEL_PATH else "api"],
        agent=HARBOR_AGENT,
        n_concurrent=HARBOR_N_CONCURRENT,
        env=ENV_TYPE,
        agent_kwargs=HARBOR_AGENT_KWARGS,
    ).with_output_path(OUTPUT_DIR)

    executor_main(steps=[step])
