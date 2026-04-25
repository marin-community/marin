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

from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging

from marin.execution.executor import Executor, ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

from experiments.datakit_testbed.mixture import weights_from_tokenized_bucket_stats
from experiments.datakit_testbed.sampler import build_testbed_steps
from experiments.datakit_testbed.settings import TESTBED_TOKENIZER
from experiments.datakit_testbed.train import run_testbed_config, testbed_tokenize

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
TARGET_TOTAL_TOKENS_B = 1000.0

_SAMPLE_STEP_PREFIX = "data/datakit/"


def baseline(
    steps: list[StepSpec],
    *,
    name: str,
    tokenizer: str,
) -> ExecutorStep:
    """Assemble the baseline training step off a testbed DAG"""

    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the DAG (expected names under 'data/datakit/...')")

    tokenized_buckets = {name: testbed_tokenize(name, sampled, tokenizer) for name, sampled in sampled_by_source.items()}
    # Run tokenize steps through an Executor we keep a handle on so we can read
    # each step's *resolved* output_path afterwards. ``executor_main`` discards
    # its Executor, and ``output_path_of`` outside an executor context returns
    # a lazy ``InputName`` — not a concrete GCS path.
    prefix = marin_prefix()
    tokenize_executor = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
    )
    tokenize_executor.run(list(tokenized_buckets.values()))

    resolved_output_paths = {
        bucket_name: tokenize_executor.output_paths[step] for bucket_name, step in tokenized_buckets.items()
    }
    weights = weights_from_tokenized_bucket_stats(resolved_output_paths)
    return run_testbed_config(
        name=name,
        tokenized_buckets=tokenized_buckets,
        weights=weights,
        tokenizer=tokenizer,
    )


def main() -> None:
    """Entry-point: materialize ferry + tokenize, then launch baseline training"""
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    tokenizer = TESTBED_TOKENIZER
    run_id = "baseline"

    testbed_steps = build_testbed_steps(target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    logger.info("Materializing %d ferry StepSpecs under %s", len(testbed_steps), STAGING_PREFIX)
    StepRunner().run(testbed_steps)

    training_step = baseline(testbed_steps, name=run_id, tokenizer=tokenizer)
    executor_main(ExecutorMainConfig(max_concurrent=42), [training_step])


if __name__ == "__main__":
    configure_logging()
    main()
