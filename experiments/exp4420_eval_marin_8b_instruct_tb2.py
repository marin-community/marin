# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Eval Marin-8B Instruct on Harbor benchmarks (TB2, TBLite).

Model: marin-community/marin-8b-instruct (Llama 3 8B architecture, SFT on 5.3B tokens)
Agent: Terminus 2
Benchmarks: TB2, TBLite (controlled by HARBOR_BENCHMARK env var)

Expecting ~0% on TB2 since this model was not trained on terminal agent data.

Tracked in: https://github.com/marin-community/marin/issues/4420

Usage (Iris with Daytona):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 \
        --extra harbor --extra vllm --extra tpu \
        -e HARBOR_BENCHMARK tb2 \
        -e MARIN_PREFIX gs://marin-us-central1 \
        -e DAYTONA_API_KEY ${DAYTONA_API_KEY} \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT harbor \
        -e HF_TOKEN ${HF_TOKEN} \
        -e ENV_TYPE daytona \
        -e MARIN_VLLM_MODE native \
        -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 \
        --no-wait \
        -- python experiments/exp4420_eval_marin_8b_instruct_tb2.py

    For sharded eval (4-way):
        Add -e HARBOR_TASK_NAMES_JSON '<json list of task names>'

Environment variables:
    - HARBOR_BENCHMARK: "tblite" | "tb2" (default: "tb2")
    - ENV_TYPE: "local" | "daytona" (default: "daytona")
    - HARBOR_N_CONCURRENT: number of parallel trials (default: 25)
    - HARBOR_TASK_NAMES_JSON: JSON list of task names (for sharding)
    - HARBOR_AGENT_KWARGS_JSON: JSON dict forwarded to Harbor AgentConfig.kwargs
"""

import json
import logging
import os

from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


BENCHMARK = os.environ.get("HARBOR_BENCHMARK", "tb2").strip().lower()

if BENCHMARK == "tblite":
    DATASET = "open-thoughts/OpenThoughts-TBLite"
    VERSION = "hf"
elif BENCHMARK == "tb2":
    DATASET = "terminal-bench"
    VERSION = "2.0"
else:
    raise ValueError(f"Unknown benchmark: {BENCHMARK}")

ENV_TYPE = os.environ.get("ENV_TYPE", "daytona")
HARBOR_AGENT = "terminus-2"

MODEL_NAME = "marin-8b-instruct"
MODEL_PATH = "marin-community/marin-8b-instruct"

HARBOR_N_CONCURRENT = _optional_int_from_env("HARBOR_N_CONCURRENT", default=25) or 1
HARBOR_AGENT_KWARGS = _optional_json_dict_from_env("HARBOR_AGENT_KWARGS_JSON")
HARBOR_TASK_NAMES: list[str] | None = None
_task_names_raw = os.environ.get("HARBOR_TASK_NAMES_JSON", "").strip()
if _task_names_raw:
    HARBOR_TASK_NAMES = json.loads(_task_names_raw)
    if not isinstance(HARBOR_TASK_NAMES, list):
        raise TypeError(f"HARBOR_TASK_NAMES_JSON must be a JSON list, got {type(HARBOR_TASK_NAMES)}")

# Use temperature=0.6 to match the Nemotron-Terminal evaluation protocol.
HARBOR_AGENT_KWARGS.setdefault("temperature", 0.6)

DATASET_PATH_SEGMENT = DATASET.replace("/", "__") if "/" in DATASET else DATASET
OUTPUT_DIR = os.path.join("evaluation", "harbor", DATASET_PATH_SEGMENT, MODEL_NAME, HARBOR_AGENT)
if HARBOR_TASK_NAMES:
    import hashlib as _hl

    _shard_hash = _hl.sha256("|".join(sorted(HARBOR_TASK_NAMES)).encode()).hexdigest()[:8]
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"shard_{_shard_hash}")

if ENV_TYPE == "daytona" and not os.environ.get("DAYTONA_API_KEY"):
    logger.warning("DAYTONA_API_KEY not set but ENV_TYPE=daytona!")

logger.info("=" * 80)
logger.info("Eval Marin-8B Instruct on %s (exp4420)", DATASET)
logger.info("=" * 80)
logger.info("Benchmark: %s, Dataset: %s@%s", BENCHMARK, DATASET, VERSION)
logger.info("Model: %s (%s)", MODEL_NAME, MODEL_PATH)
logger.info("Agent: %s, Env: %s, Concurrent: %s", HARBOR_AGENT, ENV_TYPE, HARBOR_N_CONCURRENT)
logger.info("Shard: %s", f"{len(HARBOR_TASK_NAMES)} tasks" if HARBOR_TASK_NAMES else "all")
logger.info("Output: %s", OUTPUT_DIR)
logger.info("=" * 80)


if __name__ == "__main__":
    step = evaluate_harbor(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        dataset=DATASET,
        version=VERSION,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        wandb_tags=["harbor", DATASET, "terminus-2", ENV_TYPE, MODEL_NAME, "vllm", "exp4420", BENCHMARK],
        agent=HARBOR_AGENT,
        n_concurrent=HARBOR_N_CONCURRENT,
        env=ENV_TYPE,
        agent_kwargs=HARBOR_AGENT_KWARGS,
        task_names=HARBOR_TASK_NAMES,
    ).with_output_path(OUTPUT_DIR)

    executor_main(steps=[step])
