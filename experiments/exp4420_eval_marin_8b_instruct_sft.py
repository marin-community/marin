# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Eval Marin-8B Instruct Terminal-Corpus SFT checkpoint on Harbor benchmarks.

Model: exp4420 SFT (Marin-8B Instruct, 32K context, 366K examples, 2 epochs)
Agent: Terminus 2
Benchmarks: TBLite, TB2 (controlled by HARBOR_BENCHMARK env var)

Tracked in: https://github.com/marin-community/marin/issues/4420

Environment variables:
    - HARBOR_BENCHMARK: "tblite" | "tb2" (default: "tblite")
    - HARBOR_MODEL_PATH: override the SFT checkpoint path (default: auto-derived)
    - ENV_TYPE: "local" | "daytona" (default: "daytona")
    - HARBOR_N_CONCURRENT: number of parallel trials (default: 25)
    - HARBOR_TASK_NAMES_JSON: JSON list of task names (for sharding)
    - HARBOR_AGENT_KWARGS_JSON: JSON dict forwarded to Harbor AgentConfig.kwargs

Usage (Iris with Daytona):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 \
        --extra harbor --extra vllm --extra tpu \
        -e HARBOR_BENCHMARK tblite \
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
        -- python experiments/exp4420_eval_marin_8b_instruct_sft.py
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


BENCHMARK = os.environ.get("HARBOR_BENCHMARK", "tblite").strip().lower()

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

MODEL_NAME = "exp4420-marin-8b-instruct-sft"
# Override via HARBOR_MODEL_PATH once SFT training completes and the checkpoint path is known.
MODEL_PATH = os.environ.get("HARBOR_MODEL_PATH")
if not MODEL_PATH:
    raise ValueError(
        "HARBOR_MODEL_PATH must be set to the GCS path of the exp4420 SFT checkpoint "
        "(e.g., gs://marin-us-east5/checkpoints/exp4420_.../hf/step-NNNN/)"
    )

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
logger.info("Eval Marin-8B Instruct SFT checkpoint: %s on %s (exp4420)", MODEL_NAME, DATASET)
logger.info("=" * 80)
logger.info("Benchmark: %s, Dataset: %s@%s", BENCHMARK, DATASET, VERSION)
logger.info("Model path: %s", MODEL_PATH)
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
        wandb_tags=["harbor", DATASET, "terminus-2", ENV_TYPE, MODEL_NAME, "vllm", "exp4420", BENCHMARK, "marin-sft"],
        agent=HARBOR_AGENT,
        n_concurrent=HARBOR_N_CONCURRENT,
        env=ENV_TYPE,
        agent_kwargs=HARBOR_AGENT_KWARGS,
        task_names=HARBOR_TASK_NAMES,
    ).with_output_path(OUTPUT_DIR)

    executor_main(steps=[step])
