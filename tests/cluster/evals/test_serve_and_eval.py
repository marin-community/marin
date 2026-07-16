# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU smoke for the serve->eval orchestration, on the standing Marin cluster.

Validates the two-job evalchemy path end-to-end (issue #7267): a parent orchestrator job serves
Qwen3-0.6B with marin-serve (vLLM on a TPU slice), the evalchemy child evaluates a generation task
(gsm8k) and a multiple-choice task (arc_easy) against the served OpenAI URL, and the server is torn
down. It exercises the same ``serve_and_eval`` entrypoint the composable ``eval_step`` runs through.

It drives live Iris TPU jobs, so it is marked ``cluster`` and deselected by default (see
``pyproject.toml`` addopts); the ``marin-cluster-smoke`` workflow runs it. Run it on demand once you
have cluster credentials and HF_TOKEN set:

    uv run pytest tests/cluster/evals/test_serve_and_eval.py \
      -m cluster -o addopts= --import-mode=importlib --timeout=0 -vv -s
"""

from __future__ import annotations

import pytest
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec, is_job_finished
from marin.evaluation.eval_result import EvalchemyResult
from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.evals.evalchemy.serve_and_eval import EvalchemyEvalConfig, ServeSpec, serve_and_eval

pytestmark = pytest.mark.cluster

# One generation task and one multiple-choice task, both capped to a few instances: the minimal check
# that logprob (arc_easy) and free-generation (gsm8k) both flow through the served OpenAI endpoint.
SMOKE_TASKS = (EvalTaskConfig("arc_easy", 0), EvalTaskConfig("gsm8k", 5))

# Serve boot (model fetch + vLLM compile) dominates; the eval itself is a few instances per task.
_SERVE_AND_EVAL_TIMEOUT_SECONDS = 3000.0


def test_serve_and_eval_smoke(iris_client: IrisClient, smoke_region: str) -> None:
    # smoke_region pins the slice and binds the storage root to the same gs://marin-<region>, so the
    # serve child, eval child, and outputs all colocate -- no cross-region I/O.
    out_path = f"gs://marin-{smoke_region}/tmp/eval7267-serve-and-eval-smoke/qwen3-0p6b"
    config = EvalchemyEvalConfig(
        model="Qwen/Qwen3-0.6B",
        tasks=SMOKE_TASKS,
        out_path=out_path,
        serve=ServeSpec(backend="vllm", tpu_type="v6e-4", region=smoke_region),
        max_eval_instances=3,
    )

    # serve_and_eval submits its own serve + eval children, so run it as a plain CPU job rather than
    # through StepRunner. wait(raise_on_failure default) raises if the parent (or either child) fails.
    job = iris_client.submit(
        entrypoint=Entrypoint.from_callable(serve_and_eval, config),
        name="eval7267-serve-and-eval-smoke",
        resources=ResourceSpec(cpu=1, memory="4g", disk="16g"),
        max_retries_failure=0,
    )
    try:
        job.wait(timeout=_SERVE_AND_EVAL_TIMEOUT_SECONDS, stream_logs=True)
    finally:
        if not is_job_finished(job.state):
            job.terminate()

    metrics = EvalchemyResult.raw_load(out_path).task_metrics()
    assert set(metrics) >= {"arc_easy", "gsm8k"}, metrics
    assert metrics["arc_easy"], "arc_easy produced no numeric metrics"
    assert metrics["gsm8k"], "gsm8k produced no numeric metrics"
