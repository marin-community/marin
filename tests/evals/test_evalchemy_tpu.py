# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU smoke for the :evalchemy-tpu eval container. MANUAL — not run by pytest CI.

Validates the newly-built ``:evalchemy-tpu`` image end-to-end on a preemptible TPU slice: the
container pulls + runs, vllm-tpu 0.20.0 serves a tiny model, the evalchemy fork runs MATH500 seed42,
and it emits a score. Tiny public Qwen3-0.6B model, single seed, one task.

It drives a live Iris TPU job with a custom container, so it cannot run in unit CI — it is marked
``manual`` and deselected by default (see ``pyproject.toml`` addopts). Run it on demand once you have
cluster credentials and MARIN_PREFIX/HF_TOKEN set:

    uv run pytest tests/evals/test_evalchemy_tpu.py -m manual -o addopts= -vv -s
"""
from __future__ import annotations

import pytest
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

from experiments.evals.evalchemy.marin_evalchemy_tpu import SUITE_TO_TASKS, EvalSpec, evalchemy_tpu_step

pytestmark = pytest.mark.manual

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


def test_evalchemy_tpu_smoke() -> None:
    StepRunner().run([lower(evalchemy_tpu_step(SMOKE_SPEC))])
