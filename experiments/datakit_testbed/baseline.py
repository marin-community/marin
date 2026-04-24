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
from experiments.datakit_testbed.train import build_testbed_tokenize_steps, run_testbed_config

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
RUN_ID = "baseline"
TARGET_TOTAL_TOKENS_B = 10.0

_SAMPLE_STEP_PREFIX = "datakit-testbed/"


def baseline(
    steps: list[StepSpec],
    *,
    name: str = "baseline",
    tokenizer: str = TESTBED_TOKENIZER,
    **run_config_kwargs,
) -> ExecutorStep:
    """Assemble the baseline training step off a testbed DAG.

    Bucketing is "one tokenize per source" — the trivial control-arm
    strategy. Other configurations build their own ``tokenized_by_source``
    (e.g. by quality tier) and pass it to :func:`run_testbed_config`
    directly.
    """
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the DAG (expected names under 'datakit-testbed/...')")
    tokenized_by_source = build_testbed_tokenize_steps(sampled_by_source, tokenizer=tokenizer)
    return run_testbed_config(
        name=name,
        tokenized_by_source=tokenized_by_source,
        tokenizer=tokenizer,
        **run_config_kwargs,
    )


def main() -> None:
    """Entry-point: materialize the ferry, then launch baseline training"""
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    testbed_steps = build_testbed_steps(RUN_ID, target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    logger.info("Materializing %d ferry StepSpecs under %s", len(testbed_steps), STAGING_PREFIX)
    StepRunner().run(testbed_steps)

    training_step = baseline(testbed_steps, name=RUN_ID)
    executor_main(ExecutorMainConfig(), [training_step])


if __name__ == "__main__":
    configure_logging()
    main()
