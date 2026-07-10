# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontamination arm for the Datakit Testbed 1T sample.

Wires a per-source ``decon_step`` over the testbed's sampled normalized
shards, consuming a single combined bloom built with the eval-task
exclusion applied at read time (``exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS``,
see marin#6852 / #7007). Used to measure the fixed decon's flag rates on the
~1T-token by-provenance sample.

Prereq: the eval corpus must be staged in the run region
(``{marin_prefix()}/datakit/decontam/evals``). It is ~240 MiB; stage once:

    gsutil -m rsync -r gs://marin-eu-west4/datakit/decontam/evals \\
        gs://marin-us-central1/datakit/decontam/evals

Submit on iris (us-central1, same region as the sample — no egress):

    uv run iris --cluster=marin job run --region us-central1 \\
        --extra=cpu --priority interactive \\
        --memory 8GB --cpu 2 --enable-extra-resources \\
        -- python experiments/datakit/testbed/decon_arm.py [--only <source> ...]
"""

import argparse
import logging
import os

from fray.types import ResourceConfig
from marin.datakit.decon import build_eval_bloom_step, decon_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS
from experiments.datakit.testbed.sampler import build_testbed_steps
from experiments.datakit.testbed.settings import RAW_TARGET_TOTAL_TOKENS_B

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
_SAMPLE_STEP_PREFIX = "data/datakit/normalized/"

# Bloom sizing mirrors experiments/datakit/decontam/all_sources_decon.py.
ESTIMATED_DOC_COUNT = 50_000_000
FALSE_POSITIVE_RATE = 1e-9
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")


def build_testbed_decon_steps(
    target_total_tokens_b: float = RAW_TARGET_TOTAL_TOKENS_B,
    only_sources: list[str] | None = None,
) -> list[StepSpec]:
    """Bloom (fixed) + one decon step per sampled source.

    ``only_sources`` restricts the decon fan-out to named sources (for smokes);
    StepRunner still walks each decon step's deps, so the matching sample +
    normalize chains run on demand. Sample fractions are always the full 1T
    proportional fractions, so a restricted run still decons a source's true
    1T-share sample.
    """
    testbed_steps = build_testbed_steps(target_total_tokens_b=target_total_tokens_b)
    sampled = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in testbed_steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if only_sources:
        missing = [s for s in only_sources if s not in sampled]
        if missing:
            raise ValueError(f"unknown sources: {missing}; have {sorted(sampled)[:5]}…")
        sampled = {k: v for k, v in sampled.items() if k in only_sources}

    bloom = build_eval_bloom_step(
        name="datakit/bloom/_combined_fixed",
        eval_data_sources=[f"{marin_prefix()}/datakit/decontam/evals"],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )

    steps: list[StepSpec] = [bloom]
    for name, sample_step in sampled.items():
        steps.append(
            decon_step(
                name=f"datakit/testbed_decon/{name}",
                normalized=sample_step,
                prebuilt_bloom=bloom,
                ngram_length=NGRAM_LENGTH,
                overlap_threshold=OVERLAP_THRESHOLD,
                estimated_doc_count=ESTIMATED_DOC_COUNT,
                false_positive_rate=FALSE_POSITIVE_RATE,
                worker_resources=WORKER_RESOURCES,
            )
        )
    logger.info(
        "testbed decon: %d sources, %d steps (bloom + decon; deps pull sample/normalize)", len(sampled), len(steps)
    )
    return steps


def main() -> None:
    os.environ.setdefault("MARIN_PREFIX", STAGING_PREFIX)
    configure_logging(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="restrict decon to these source names (smoke)")
    ap.add_argument("--target-tokens-b", type=float, default=RAW_TARGET_TOTAL_TOKENS_B)
    args = ap.parse_args()
    StepRunner().run(build_testbed_decon_steps(target_total_tokens_b=args.target_tokens_b, only_sources=args.only))


if __name__ == "__main__":
    main()
