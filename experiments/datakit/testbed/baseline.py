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
produces is also the bucket. ``main`` wires one tokenize step per sample output,
computes mixture weights at runtime via :func:`tokenized_bucket_weights_step`, and
hands the result to :func:`run_testbed_config` which assembles the full Grug-MoE
training step. The whole pipeline (ferry → tokenize → weights → train) lives in one
``StepSpec`` graph that :class:`StepRunner` walks, scheduling each step once its
dependencies are satisfied.
"""

import logging
import os

from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging

from experiments.datakit.testbed.mixture import tokenized_bucket_weights_step
from experiments.datakit.testbed.sampler import build_testbed_steps
from experiments.datakit.testbed.settings import TESTBED_TOKENIZER
from experiments.datakit.testbed.train import run_testbed_config, testbed_tokenize
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
TARGET_TOTAL_TOKENS_B = 1000.0
MAX_STEP_CONCURRENCY = 20

_SAMPLE_STEP_PREFIX = "data/datakit/normalized/"


def main() -> None:
    """Build the baseline DAG and run it."""
    os.environ.setdefault("MARIN_PREFIX", STAGING_PREFIX)

    tokenizer = TESTBED_TOKENIZER
    run_id = "baseline"
    validation = [*paloma_datasets(tokenizer=tokenizer).values(), *uncheatable_datasets(tokenizer=tokenizer).values()]

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
        validation=validation,
        tokenizer=tokenizer,
    )

    logger.info("Baseline DAG: %d sources → tokenize → weights → train", len(sampled_by_source))
    StepRunner().run([training_step], max_concurrent=MAX_STEP_CONCURRENCY)


if __name__ == "__main__":
    configure_logging()
    main()
