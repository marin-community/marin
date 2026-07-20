# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU smoke for the :evalchemy-tpu eval container, on the standing Marin cluster.

Validates the newly-built ``:evalchemy-tpu`` image end-to-end on a preemptible TPU slice: the
container pulls + runs, vllm-tpu 0.20.0 serves a tiny model, the evalchemy fork runs MATH500 seed42,
and it emits a score. Tiny public Qwen3-0.6B model, single seed, one task.

It drives a live Iris TPU job with a custom container, so it is marked ``cluster`` and deselected by
default (see ``pyproject.toml`` addopts); the ``marin-cluster-smoke`` workflow runs it. The
``iris_client`` fixture binds the ``marin`` controller as the current Fray client, so ``StepRunner``
submits there. Run it on demand once you have cluster credentials and HF_TOKEN set:

    uv run pytest tests/cluster/evals/test_evalchemy_tpu.py \
      -m cluster -o addopts= --import-mode=importlib --timeout=0 -vv -s
"""
from __future__ import annotations

import dataclasses

import pytest
from iris.client import IrisClient
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

from experiments.evals.evalchemy.marin_evalchemy_tpu import SUITE_TO_TASKS, EvalSpec, evalchemy_tpu_step

pytestmark = pytest.mark.cluster

# smoke suite: MATH500 only (seed 42), the minimal end-to-end check
SUITE_TO_TASKS["smoke_math500"] = ["MATH500"]

SMOKE_SPEC = EvalSpec(
    run_name="smoke-evalchemy-tpu-qwen3-0p6b",
    model="Qwen/Qwen3-0.6B",  # tiny public Qwen3 (matches the delphi target arch)
    suite="smoke_math500",
    stage="sft",
    seeds=(42,),
    max_model_len=4096,
    max_gen_toks=2048,
    tpu_type="v6e-4",
)


def test_evalchemy_tpu_smoke(iris_client: IrisClient, smoke_region: str) -> None:
    # iris_client binds the marin controller as the current Fray client, so StepRunner submits there.
    # smoke_region pins the slice to a region and binds the storage root to the same region, so the
    # job reads and writes region-locally -- no cross-region I/O.
    spec = dataclasses.replace(SMOKE_SPEC, region=smoke_region)
    StepRunner().run([lower(evalchemy_tpu_step(spec))])
