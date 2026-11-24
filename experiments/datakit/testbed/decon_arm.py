# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontamination arm for the Datakit Testbed 1T sample.

Runs the reference pipeline's decontamination stages
(:func:`experiments.datakit.reference_pipeline.decontamination_steps`: item table, keys, one mark per source)
over the testbed's sampled sources, to measure the flag rates on the ~1T-token by-provenance sample.

Prereq: the prepared item table and the decontamination models must be staged in the run region
(``{marin_prefix()}/datakit/decontam/items/<ITEMS_VERSION>`` and ``{marin_prefix()}/datakit/decontam/models``).
The marks score on CPU workers (``CPU_SCORE_WORKERS``) because the GCP cluster has no GPUs; CPU throughput is
not measured.

Submit on iris (us-central1, same region as the sample — no egress):

    uv run iris --cluster=marin job run --region us-central1 \\
        --extra=cpu --priority interactive \\
        --memory 8GB --cpu 2 --enable-extra-resources \\
        -- python experiments/datakit/testbed/decon_arm.py [--only <source> ...]
"""

import argparse
import dataclasses
import logging
import os

from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import check_path_in_region, marin_prefix
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.config import CPU_SCORE_WORKERS
from experiments.datakit.reference_pipeline import (
    DEFAULT_SCALE,
    DecontaminationSteps,
    decontamination_steps,
    require_prebuilt_inputs,
    sample_sources,
)
from experiments.datakit.testbed.sampler import build_testbed_steps
from experiments.datakit.testbed.settings import (
    RAW_TARGET_TOTAL_TOKENS_B,
    TESTBED_STAGING_PREFIX,
    TESTBED_STAGING_REGION,
)

logger = logging.getLogger(__name__)

_SAMPLE_STEP_PREFIX = "data/datakit/normalized/"


def testbed_sources(
    target_total_tokens_b: float = RAW_TARGET_TOTAL_TOKENS_B,
    exclude_sources: frozenset[str] = frozenset(),
    sample_root: str | None = None,
) -> dict[str, StepSpec]:
    """The testbed's sampled sources as normalized steps.

    ``exclude_sources`` drops sources from the sample *construction* (the proportional fractions are recomputed
    over the kept set), so it must match the exclusion used to build the samples being deconned. ``sample_root``
    uses a **pre-materialized** sample root under MARIN_PREFIX (e.g. ``datakit/sample_1t_733c8c5c``) instead of
    re-running the sampler.
    """
    if sample_root:
        found = sample_sources(f"{marin_prefix()}/{sample_root}")
        return {name: step for name, step in found.items() if name not in exclude_sources}
    sources = [s for n, s in all_sources().items() if n not in exclude_sources] if exclude_sources else None
    steps = build_testbed_steps(target_total_tokens_b=target_total_tokens_b, sources=sources)
    return {s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)}


def build_testbed_decon_steps(
    target_total_tokens_b: float = RAW_TARGET_TOTAL_TOKENS_B,
    only_sources: list[str] | None = None,
    exclude_sources: frozenset[str] = frozenset(),
    sample_root: str | None = None,
) -> DecontaminationSteps:
    """The decontamination stages over the testbed sample; ``only_sources`` restricts the marks (smokes) while the
    keys still use every sampled source, so a subset mark keeps its full-run identity."""
    sources = testbed_sources(target_total_tokens_b, exclude_sources, sample_root)
    if only_sources:
        missing = [s for s in only_sources if s not in sources]
        if missing:
            raise ValueError(f"unknown sources: {missing}; have {sorted(sources)[:5]}…")
    scale = dataclasses.replace(DEFAULT_SCALE, decon_score=CPU_SCORE_WORKERS)
    steps = decontamination_steps(sources, scale=scale, mark_source_names=only_sources)
    logger.info("testbed decon: %d sources, %d marks", len(sources), len(steps.marks))
    return steps


def main() -> None:
    configure_logging(logging.INFO)
    os.environ.setdefault("MARIN_PREFIX", TESTBED_STAGING_PREFIX)
    check_path_in_region("MARIN_PREFIX", marin_prefix(), TESTBED_STAGING_REGION)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="restrict decon to these source names (smoke)")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="drop these sources from the sample construction (recomputes fractions; "
        "match the samples being deconned, e.g. finetranslations ghalogs/public for the CoreWeave testbed)",
    )
    ap.add_argument("--target-tokens-b", type=float, default=RAW_TARGET_TOTAL_TOKENS_B)
    ap.add_argument(
        "--sample-root",
        default=None,
        help="decon a pre-materialized sample root under MARIN_PREFIX directly "
        "(e.g. datakit/sample_1t_733c8c5c); marks read <root>/<source>/outputs/main",
    )
    args = ap.parse_args()
    steps = build_testbed_decon_steps(
        target_total_tokens_b=args.target_tokens_b,
        sample_root=args.sample_root,
        only_sources=args.only,
        exclude_sources=frozenset(args.exclude or ()),
    )
    require_prebuilt_inputs([steps.items])
    StepRunner().run([steps.keys, *steps.scans.values(), *steps.scores.values(), *steps.marks.values()])


if __name__ == "__main__":
    main()
