# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Reproduce ricdomolm/mini-coder-1.7b on SWE-bench Verified (100-instance subset).

Evaluates the published mini-coder-1.7b checkpoint on the same 100 random
SWE-bench Verified instances used in exp4898 (DCAgent2/swebench-verified-random-100-folders),
configured to match the mini-coder author's eval recipe as closely as the
Harbor + mini-swe-agent-v1 stack allows:

  - Agent template: mini-swe-agent v1 bundled `extra/swebench.yaml` (THOUGHT +
    single bash code block; matches the format the model was trained on).
  - step_limit=250 (bundled swebench.yaml default; Harbor's old default was 15).
  - temperature=0.0 (mini-extra default).
  - max_input_tokens=40960, max_output_tokens=4096 (Qwen3-1.7B context cap;
    talkie-coder's eval used 4096 output tokens).
  - vLLM: tensor_parallel_size=1, max_model_len=40960 (1.7B fits a single chip).
  - VLLM_TOKENIZER NOT overridden — Qwen3 default chat_template applies.

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
        -- python experiments/exp_repro_mini_coder_1_7b.py

Environment variables (with defaults):
    MODEL_PATH=ricdomolm/mini-coder-1.7b
    MODEL_NAME=mini-coder-1.7b
    HARBOR_RUN_ID=repro-100
    ENV_TYPE=daytona
    HARBOR_N_CONCURRENT=4
    TPU_VARIANT=v6e-1
"""

import json
import logging
import os
from pathlib import Path

import yaml

from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Same 100 SWE-bench Verified instances as exp4898 baseline so results are
# directly comparable across models trained/evaluated on this subset.
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

_task_names_json = os.environ.get("HARBOR_TASK_NAMES_JSON")
TASK_NAMES = json.loads(_task_names_json) if _task_names_json else ALL_TASK_NAMES

MODEL_PATH = os.environ.get("MODEL_PATH", "ricdomolm/mini-coder-1.7b")
MODEL_NAME = os.environ.get("MODEL_NAME", "mini-coder-1.7b")
ENV_TYPE = os.environ.get("ENV_TYPE", "daytona")
N_CONCURRENT = int(os.environ.get("HARBOR_N_CONCURRENT", "4"))
RUN_ID = os.environ.get("HARBOR_RUN_ID", "repro-100")

# Qwen3-1.7B fits a single chip; tp=1, n=4 fits in v6e-1 HBM with room to spare.
_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v6e-1")
RESOURCES = ResourceConfig.with_tpu(_TPU_VARIANT)

# vLLM context: 40960 matches mini-coder's litellm registry max_tokens cap.
MAX_MODEL_LEN = 40960
MAX_INPUT_TOKENS = 40960
MAX_OUTPUT_TOKENS = 4096

VLLM_ENGINE_KWARGS = {
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": N_CONCURRENT,
    "tensor_parallel_size": 1,
}

# Mini-coder-1.7b is a Qwen3 finetune; use its native chat template (do NOT
# override VLLM_TOKENIZER — exp4898 sets marin-tokenizer for marin-8b only).
os.environ.pop("VLLM_TOKENIZER", None)
os.environ.setdefault("MSWEA_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

# Load the mini-swe-agent v1 bundled swebench.yaml templates. This is what
# `mini-extra swebench` (the command in mini-coder-1.7b's README) uses, and
# what the model was trained against. The Harbor adapter applies these via
# `agent_config_overrides`.
_SWEBENCH_YAML = Path(__file__).parent / "configs" / "miniswea_v1_swebench.yaml"
_swebench_cfg = yaml.safe_load(_SWEBENCH_YAML.read_text())
AGENT_CONFIG_OVERRIDES = _swebench_cfg["agent"]
# Match mini-extra default. step_limit is also enforced via the explicit kwarg
# below, but include it here so the loaded config is self-consistent.
AGENT_CONFIG_OVERRIDES["step_limit"] = 250
AGENT_CONFIG_OVERRIDES["cost_limit"] = 3.0

AGENT_KWARGS = {
    "temperature": 0.0,
    "step_limit": 250,
    "agent_config_overrides": AGENT_CONFIG_OVERRIDES,
    "model_info": {
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
}

OUTPUT_DIR = os.path.join("evaluation", "harbor", "swebench-verified", MODEL_NAME, "mini-swe-agent-v1", RUN_ID)

logger.info("=" * 60)
logger.info("Reproduce mini-coder-1.7b on SWE-bench Verified (100 tasks)")
logger.info("Model: %s  Tasks: %d  Concurrent: %d  Env: %s", MODEL_PATH, len(TASK_NAMES), N_CONCURRENT, ENV_TYPE)
logger.info("step_limit=250  temperature=0.0  max_model_len=%d  tp=1  variant=%s", MAX_MODEL_LEN, _TPU_VARIANT)
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
            "mini-swe-agent-v1",
            ENV_TYPE,
            MODEL_NAME,
            "repro",
            "mini-coder",
        ],
        agent="mini-swe-agent-v1",
        n_concurrent=N_CONCURRENT,
        env=ENV_TYPE,
        agent_kwargs=AGENT_KWARGS,
    ).with_output_path(OUTPUT_DIR)

    executor_main(steps=[step])
