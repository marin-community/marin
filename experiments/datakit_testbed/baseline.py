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
produces is also the bucket. ``main`` wires one tokenize ExecutorStep per
sample output, computes mixture weights at runtime via
:func:`tokenized_bucket_weights_step`, and hands the result to
:func:`run_testbed_config` which assembles the full Grug-MoE training step.

The whole pipeline (ferry → tokenize → weights → train) lives in one
executor DAG so ``--dry_run`` validates structure without touching GCS.
"""

from __future__ import annotations

import dataclasses
import logging
import os

import draccus
from marin.execution.executor import ExecutorMainConfig, executor_main
from rigging.log_setup import configure_logging

from experiments.datakit_testbed.mixture import tokenized_bucket_weights_step
from experiments.datakit_testbed.sampler import build_testbed_steps
from experiments.datakit_testbed.settings import TESTBED_TOKENIZER
from experiments.datakit_testbed.train import run_testbed_config, testbed_tokenize

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
TARGET_TOTAL_TOKENS_B = 1000.0
MAX_STEP_CONCURRENCY = 20

_SAMPLE_STEP_PREFIX = "data/datakit/normalized/"


def main() -> None:
    """Build the baseline DAG and hand it to ``executor_main``."""
    config = draccus.parse(ExecutorMainConfig)
    if config.prefix is None:
        config = dataclasses.replace(config, prefix=STAGING_PREFIX)
    os.environ.setdefault("MARIN_PREFIX", config.prefix)

    tokenizer = TESTBED_TOKENIZER
    run_id = "baseline"

    testbed_steps = build_testbed_steps(target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in testbed_steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the testbed DAG")

    tokenized_buckets = {name: testbed_tokenize(name, sampled, tokenizer) for name, sampled in sampled_by_source.items()}
    weights_step = tokenized_bucket_weights_step(run_id, tokenized_buckets)
    training_step = run_testbed_config(
        name=run_id,
        tokenized_buckets=tokenized_buckets,
        weights_step=weights_step,
        tokenizer=tokenizer,
    )

    logger.info("Baseline DAG: %d sources → tokenize → weights → train", len(sampled_by_source))
    executor_main(config, [training_step], max_concurrent=MAX_STEP_CONCURRENCY)


if __name__ == "__main__":
    configure_logging()
    main()
