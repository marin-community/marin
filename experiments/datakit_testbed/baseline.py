# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed baseline — the control arm of the ranking protocol.

The testbed's ranking protocol runs every experiment alongside a baseline
where the pipeline's middle stages are deliberate no-ops:

* **no-op dedup** — every sampled doc survives (no fuzzy/exact cut)
* **constant-quality filter** — all docs tagged equal quality
* **bucket by provenance** — the sample output IS already the bucket
  (one shard set per source), so bucketing is an identity op

With all three as no-ops, the sampled parquet that :func:`build_testbed_steps`
produces is also the bucket. :func:`baseline` wires one tokenize ExecutorStep
per sample output, builds the proportional mixture over them, and hands the
result to :func:`run_testbed_config` to assemble the full Grug-MoE training
ExecutorStep.
"""

from __future__ import annotations

import logging
import os

from rigging.log_setup import configure_logging

from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

from experiments.datakit_testbed.sampler import build_testbed_steps
from experiments.datakit_testbed.settings import TESTBED_TOKENIZER
from experiments.datakit_testbed.train import run_testbed_config

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
RUN_ID = "baseline"
TARGET_TOTAL_TOKENS_B = 10.0

_SAMPLE_STEP_PREFIX = "datakit-testbed/sample/"


def baseline(
    steps: list[StepSpec],
    *,
    name: str = "baseline",
    tokenizer: str = TESTBED_TOKENIZER,
    **run_config_kwargs,
) -> ExecutorStep:
    """Assemble the baseline training step off a testbed DAG.

    Args:
        steps: Return value of
            :func:`experiments.datakit_testbed.sampler.build_testbed_steps`.
        name: Config name — forms the executor step name and wandb run id.
        tokenizer: Tokenizer to use across every component; must match
            the training model's tokenizer.
        **run_config_kwargs: Forwarded to
            :func:`experiments.datakit_testbed.train.run_testbed_config`
            (compute_budget_flops, hidden_dim, target_steps, weights,
            tpu, wandb_group, etc.).

    Returns:
        An ``ExecutorStep`` whose ``fn`` is ``run_grug_moe_trial``. Pass
        to ``executor_main`` to actually train.
    """
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the DAG (expected names under 'datakit-testbed/sample/...')")
    return run_testbed_config(
        name=name,
        sampled_by_source=sampled_by_source,
        tokenizer=tokenizer,
        **run_config_kwargs,
    )


def main() -> None:
    """Entry-point: materialize the ferry, then launch baseline training.

    Runs against whichever sources already have a cached normalize (so no
    extra normalize work happens here), targets
    :data:`TARGET_TOTAL_TOKENS_B` billion tokens via proportional
    sampling, then hands the resulting ferry off to the baseline training
    config via :func:`executor_main`.
    """
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    ferry_steps = build_testbed_steps(RUN_ID, target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    logger.info("Materializing %d ferry StepSpecs under %s", len(ferry_steps), STAGING_PREFIX)
    StepRunner().run(ferry_steps)

    training_step = baseline(ferry_steps, name=RUN_ID)
    executor_main(ExecutorMainConfig(), [training_step])


if __name__ == "__main__":
    configure_logging()
    main()
