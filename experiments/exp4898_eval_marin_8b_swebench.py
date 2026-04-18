# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Baseline eval: Marin-8B base on SWE-bench Verified (100-instance subset).

Evaluates the untrained Marin-8B base model on 100 random SWE-bench Verified
instances (DCAgent2/swebench-verified-random-100-folders) using the mini-swe-agent
Harbor agent. Expected result: ~0% resolve rate (untrained base model).

This establishes the baseline for the SWE-ZERO quality validation (#4898).
The same eval will be run on each SFT checkpoint (10K, 50K, 100K) to measure
improvement from SWE-ZERO trajectory training.

Usage:
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --cpu 0.5 --memory 4GB --disk 10GB \
        -e DAYTONA_API_KEY ${DAYTONA_API_KEY} \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT harbor \
        -e HF_TOKEN ${HF_TOKEN} \
        -e MARIN_PREFIX gs://marin-us-central2 \
        -e ENV_TYPE daytona \
        --no-wait \
        -- python experiments/exp4898_eval_marin_8b_swebench.py

Environment variables:
    - MODEL_PATH: HF model path (default: marin-community/marin-8b-base)
    - MODEL_NAME: Model alias for logging (default: marin-8b-base)
    - ENV_TYPE: "local" | "daytona" (default: daytona)
    - HARBOR_N_CONCURRENT: parallel trials (default: 4)
"""

import json
import logging
import os

from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 100 random SWE-bench Verified instances from DCAgent2/swebench-verified-random-100-folders
# Override with HARBOR_TASK_NAMES_JSON env var for sharded evaluation.
ALL_TASK_NAMES = [
    "astropy__astropy-13236",
    "astropy__astropy-14369",
    "astropy__astropy-14508",
    "astropy__astropy-14598",
    "astropy__astropy-14995",
    "astropy__astropy-8872",
    "django__django-10097",
    "django__django-11138",
    "django__django-11141",
    "django__django-11206",
    "django__django-11276",
    "django__django-11333",
    "django__django-11433",
    "django__django-11477",
    "django__django-11490",
    "django__django-11728",
    "django__django-11820",
    "django__django-12050",
    "django__django-12276",
    "django__django-12308",
    "django__django-12406",
    "django__django-13109",
    "django__django-13128",
    "django__django-13315",
    "django__django-13346",
    "django__django-13363",
    "django__django-13401",
    "django__django-13410",
    "django__django-13449",
    "django__django-13516",
    "django__django-13670",
    "django__django-13925",
    "django__django-13933",
    "django__django-14017",
    "django__django-14053",
    "django__django-14238",
    "django__django-14315",
    "django__django-14855",
    "django__django-14999",
    "django__django-15037",
    "django__django-15128",
    "django__django-15252",
    "django__django-15277",
    "django__django-15278",
    "django__django-15368",
    "django__django-15467",
    "django__django-15499",
    "django__django-15987",
    "django__django-16082",
    "django__django-16485",
    "django__django-16527",
    "django__django-16595",
    "matplotlib__matplotlib-20826",
    "matplotlib__matplotlib-24870",
    "matplotlib__matplotlib-25332",
    "matplotlib__matplotlib-25960",
    "matplotlib__matplotlib-26466",
    "psf__requests-2317",
    "pydata__xarray-3151",
    "pydata__xarray-3305",
    "pydata__xarray-4629",
    "pydata__xarray-4687",
    "pydata__xarray-6938",
    "pylint-dev__pylint-4551",
    "pylint-dev__pylint-6386",
    "pylint-dev__pylint-6903",
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-10081",
    "pytest-dev__pytest-5840",
    "pytest-dev__pytest-7324",
    "pytest-dev__pytest-7521",
    "pytest-dev__pytest-8399",
    "scikit-learn__scikit-learn-12973",
    "scikit-learn__scikit-learn-13135",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-14087",
    "scikit-learn__scikit-learn-15100",
    "scikit-learn__scikit-learn-25931",
    "scikit-learn__scikit-learn-26194",
    "sphinx-doc__sphinx-11445",
    "sphinx-doc__sphinx-7440",
    "sphinx-doc__sphinx-7757",
    "sphinx-doc__sphinx-8721",
    "sphinx-doc__sphinx-9229",
    "sphinx-doc__sphinx-9230",
    "sphinx-doc__sphinx-9698",
    "sympy__sympy-13372",
    "sympy__sympy-13480",
    "sympy__sympy-14531",
    "sympy__sympy-15017",
    "sympy__sympy-16450",
    "sympy__sympy-16597",
    "sympy__sympy-18199",
    "sympy__sympy-19495",
    "sympy__sympy-20801",
    "sympy__sympy-21379",
    "sympy__sympy-21847",
    "sympy__sympy-22456",
    "sympy__sympy-23413",
    "sympy__sympy-24443",
]

# Support sharded evaluation via HARBOR_TASK_NAMES_JSON env var
_task_names_json = os.environ.get("HARBOR_TASK_NAMES_JSON")
TASK_NAMES = json.loads(_task_names_json) if _task_names_json else ALL_TASK_NAMES

MODEL_PATH = os.environ.get("MODEL_PATH", "marin-community/marin-8b-base")
MODEL_NAME = os.environ.get("MODEL_NAME", "marin-8b-base")
ENV_TYPE = os.environ.get("ENV_TYPE", "daytona")
N_CONCURRENT = int(os.environ.get("HARBOR_N_CONCURRENT", "4"))

_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v6e-4")
RESOURCES = ResourceConfig.with_tpu(_TPU_VARIANT)

VLLM_ENGINE_KWARGS = {
    "max_model_len": 32768,
    "max_num_seqs": N_CONCURRENT,
    "tensor_parallel_size": 4,
}

os.environ.setdefault("MSWEA_API_KEY", "EMPTY")  # vLLM doesn't need auth
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")  # Fallback for hosted_vllm provider

AGENT_KWARGS = {
    "version": "v1",  # Pin to mini-swe-agent v1 (bash-only, matches SWE-ZERO training format)
    "temperature": 1.0,
    "model_info": {
        "max_input_tokens": 32768,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
}

OUTPUT_DIR = os.path.join("evaluation", "harbor", "swebench-verified", MODEL_NAME, "mini-swe-agent", "baseline-100")

logger.info("=" * 60)
logger.info("exp4898 baseline: %s on SWE-bench Verified (100 tasks)", MODEL_NAME)
logger.info("Model: %s", MODEL_PATH)
logger.info("Agent: mini-swe-agent, Env: %s, Concurrent: %d", ENV_TYPE, N_CONCURRENT)
logger.info("=" * 60)

if __name__ == "__main__":
    step = evaluate_harbor(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        dataset="swebench-verified",
        version="1.0",
        task_names=TASK_NAMES,
        resource_config=RESOURCES,
        engine_kwargs=VLLM_ENGINE_KWARGS,
        wandb_tags=[
            "harbor",
            "swebench-verified",
            "mini-swe-agent",
            ENV_TYPE,
            MODEL_NAME,
            "baseline",
            "exp4898",
            "swe-zero-validation",
        ],
        agent="mini-swe-agent",
        n_concurrent=N_CONCURRENT,
        env=ENV_TYPE,
        agent_kwargs=AGENT_KWARGS,
    ).with_output_path(OUTPUT_DIR)

    executor_main(steps=[step])
