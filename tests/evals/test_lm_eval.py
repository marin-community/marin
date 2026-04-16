# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier C: full-stack lm-eval against a real vLLM server on TPU CI."""

import time

import pytest

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig

from marin.evaluation.evaluation_config import EvalTaskConfig


@pytest.fixture
def current_date_time() -> str:
    return time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())


@pytest.mark.tpu_ci
def test_lm_eval_harness(current_date_time):
    """End-to-end: vLLM on TPU + lm-eval's local-completions on gsm8k_cot."""
    model_name = "test-llama-200m"
    step = evaluate_lm_evaluation_harness(
        model_name=model_name,
        model_path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        evals=[EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)],
        max_eval_instances=1,
        deployment_kwargs={"enforce_eager": True, "max_model_len": 1024},
        resource_config=ResourceConfig.with_cpu(cpu=1),
        discover_latest_checkpoint=False,
    )
    # Execute the remote-wrapped step function directly for CI simplicity;
    # the test already runs on a TPU-CI worker and owns the model lifecycle.
    step.fn(step.config)
