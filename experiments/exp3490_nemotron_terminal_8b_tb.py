# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Reproduce NemotronTerminal-8B on Terminal-Bench 2.0 (13.0 +/- 2.2)

Model: nvidia/Nemotron-Terminal-8B (SFT from Qwen3-8B)
Agent: Terminus 2 (model-agnostic reference agent from Terminal-Bench 2.0)
Benchmark: Terminal-Bench 2.0 (89 tasks) or TBLite (100-task curated subset)
Paper: https://arxiv.org/abs/2602.21193

Tracked in: https://github.com/marin-community/marin/issues/3490

Usage (Iris with Daytona):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 \
        --extra harbor --extra vllm \
        -e MARIN_PREFIX gs://marin-us-central1 \
        -e DAYTONA_API_KEY ${DAYTONA_API_KEY} \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT harbor \
        -e HF_TOKEN ${HF_TOKEN} \
        --no-wait \
        -- python experiments/exp3490_nemotron_terminal_8b_tb.py

    For TBLite (open-thoughts/OpenThoughts-TBLite):
        Add -e HARBOR_DATASET_LITE true to the command above.

    To evaluate a custom checkpoint:
        Add -e HARBOR_MODEL_NAME <name> -e HARBOR_MODEL_PATH <gs://path/to/hf/checkpoint/>.

Environment variables:
    - ENV_TYPE: "local" | "daytona" | "e2b" | "modal" (default: "daytona")
    - HARBOR_DATASET_LITE: "true" | "1" to use TBLite instead of TB2 (default: false)
    - HARBOR_MAX_INSTANCES: number of tasks to run (default: all)
    - HARBOR_N_CONCURRENT: number of parallel trials (default: 25)
    - HARBOR_AGENT_KWARGS_JSON: JSON dict forwarded to Harbor AgentConfig.kwargs
    - HARBOR_TASK_NAMES_JSON: JSON list of task names to run (for sharding across workers)
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


def _optional_int_from_env(var_name: str, *, default: int | None) -> int | None:
    value = os.environ.get(var_name)
    if value is None:
        return default

    if value.strip().lower() in {"none", "null", "all"}:
        return None

    return int(value)


def _optional_json_dict_from_env(var_name: str) -> dict:
    value = os.environ.get(var_name)
    if value is None or not value.strip():
        return {}

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"{var_name} must be a JSON object (dict), got {type(parsed)}")
    return parsed


USE_TBLITE = os.environ.get("HARBOR_DATASET_LITE", "").strip().lower() in {"true", "1", "yes"}

if USE_TBLITE:
    DATASET = "open-thoughts/OpenThoughts-TBLite"
    VERSION = "hf"
else:
    DATASET = "terminal-bench"
    VERSION = "2.0"

ENV_TYPE = os.environ.get("ENV_TYPE", "daytona")
HARBOR_AGENT = "terminus-2"

MODEL_NAME = os.environ.get("HARBOR_MODEL_NAME", "nemotron-terminal-8b")
MODEL_PATH = os.environ.get("HARBOR_MODEL_PATH", "gs://marin-us-central1/models/nvidia--Nemotron-Terminal-8B--main/")

HARBOR_MAX_INSTANCES = _optional_int_from_env("HARBOR_MAX_INSTANCES", default=None)
HARBOR_N_CONCURRENT = _optional_int_from_env("HARBOR_N_CONCURRENT", default=25) or 1
HARBOR_AGENT_KWARGS = _optional_json_dict_from_env("HARBOR_AGENT_KWARGS_JSON")
HARBOR_TASK_NAMES: list[str] | None = None
_task_names_raw = os.environ.get("HARBOR_TASK_NAMES_JSON", "").strip()
if _task_names_raw:
    HARBOR_TASK_NAMES = json.loads(_task_names_raw)
    if not isinstance(HARBOR_TASK_NAMES, list):
        raise TypeError(f"HARBOR_TASK_NAMES_JSON must be a JSON list, got {type(HARBOR_TASK_NAMES)}")

# Sampling parameters from the model's generation_config.json to match the paper.
# Terminus-2 defaults to temperature=0.7; the model was evaluated with temperature=0.6.
HARBOR_AGENT_KWARGS.setdefault("temperature", 0.6)

DATASET_PATH_SEGMENT = DATASET.replace("/", "__") if USE_TBLITE else "terminal-bench"
OUTPUT_DIR = os.path.join("evaluation", "harbor", DATASET_PATH_SEGMENT, MODEL_NAME, HARBOR_AGENT)
if HARBOR_TASK_NAMES:
    import hashlib as _hl
    _shard_hash = _hl.sha256("|".join(sorted(HARBOR_TASK_NAMES)).encode()).hexdigest()[:8]
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"shard_{_shard_hash}")

if ENV_TYPE == "daytona" and not os.environ.get("DAYTONA_API_KEY"):
    logger.warning(
        "DAYTONA_API_KEY not set but ENV_TYPE=daytona! "
        "The evaluation will likely fail. Set DAYTONA_API_KEY or use ENV_TYPE=local for Docker."
    )

logger.info("=" * 80)
logger.info("Reproduce NemotronTerminal-8B on Terminal-Bench 2.0")
logger.info("=" * 80)
logger.info("Dataset: %s@%s", DATASET, VERSION)
logger.info("Model: %s", MODEL_NAME)
logger.info("Model path: %s", MODEL_PATH)
logger.info("Agent: %s", HARBOR_AGENT)
logger.info("Env: %s", ENV_TYPE)
logger.info("Max tasks: %s", HARBOR_MAX_INSTANCES)
logger.info("Concurrent trials: %s", HARBOR_N_CONCURRENT)
logger.info("Task names (shard): %s", f"{len(HARBOR_TASK_NAMES)} tasks" if HARBOR_TASK_NAMES else "all")
logger.info("Output dir (under MARIN_PREFIX): %s", OUTPUT_DIR)
logger.info("=" * 80)


if __name__ == "__main__":
    step = evaluate_harbor(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        dataset=DATASET,
        version=VERSION,
        max_eval_instances=HARBOR_MAX_INSTANCES,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        wandb_tags=["harbor", DATASET, "terminus-2", ENV_TYPE, MODEL_NAME, "vllm", "repro-3490"]
        + (["tblite"] if USE_TBLITE else []),
        agent=HARBOR_AGENT,
        n_concurrent=HARBOR_N_CONCURRENT,
        env=ENV_TYPE,
        agent_kwargs=HARBOR_AGENT_KWARGS,
        task_names=HARBOR_TASK_NAMES,
    ).with_output_path(OUTPUT_DIR)

    executor_main(steps=[step])
