# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU SMOKE for the :evalchemy-tpu eval container (NOT a committed artifact).

Validates the newly-built ``:evalchemy-tpu`` image end-to-end on a preemptible TPU slice:
the container pulls + runs, vllm-tpu 0.20.0 serves a tiny model, the evalchemy fork runs
MATH500 seed42, and it emits a score. Tiny public Qwen3-0.6B model, single seed, one task.
"""
from __future__ import annotations

from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

from experiments.evalchemy_tpu.marin_evalchemy_tpu import SUITE_TO_TASKS, EvalSpec, evalchemy_tpu_step

# smoke suite: MATH500 only (seed 42), the minimal end-to-end check
SUITE_TO_TASKS["smoke_math500"] = ["MATH500"]

SPEC = EvalSpec(
    run_name="smoke-evalchemy-tpu-qwen3-0p6b",
    model="Qwen/Qwen3-0.6B",  # tiny public Qwen3 (matches the delphi target arch)
    suite="smoke_math500",
    stage="sft",
    seeds=(42,),
    max_model_len=4096,
    max_gen_toks=2048,
    tpu_type="v6e-4",
)


if __name__ == "__main__":
    StepRunner().run([lower(evalchemy_tpu_step(SPEC))])
