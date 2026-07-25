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
from marin.datakit.decon import all_source_drop_sets_step, build_eval_bloom_step, decon_step
from marin.datakit.sources import all_sources
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

# Bloom sizing mirrors experiments/datakit/reference_pipeline.py.
ESTIMATED_DOC_COUNT = 50_000_000
FALSE_POSITIVE_RATE = 1e-9
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
# True-paragraph split (blank-line-delimited) rather than per-line: dilutes
# isolated-line coincidences (fewer FPs) and lets short-line / inline-embedded
# eval text be matched (higher recall). See marin#6852.
PARAGRAPH_DELIMITER = "\n\n"
# Per-source common-ngram filter (marin#6852): drop eval ngrams ubiquitous within
# a source (legal enacting clauses, license headers) from that source's overlap.
DF_SAMPLE_DOCS = 5000  # docs/source sampled to estimate per-source ngram DF
DF_COMMON_FRAC = 0.005  # ngram is "common" if present in >= this fraction of them
DF_COMMON_MIN_ABS = 5  # and in >= this many (small-source floor)
# DF is a source property, so estimate it from the *largest* materialized sample
# (the pre-built 1T per-source root) regardless of the decon target — a 100M mark
# reuses a drop-set estimated over thousands of docs. Layout: <root>/<source>/outputs/main.
SAMPLE_1T_ROOT = "datakit/sample_1t_733c8c5c"
# Reservoir-sample this many flagged docs/source into a `_flagged` sidecar at mark
# time so the viewer scales — it reads the sidecar instead of rescanning the corpus.
FLAGGED_SAMPLE_SIZE = 60
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")


def build_testbed_decon_steps(
    target_total_tokens_b: float = RAW_TARGET_TOTAL_TOKENS_B,
    only_sources: list[str] | None = None,
    exclude_sources: frozenset[str] = frozenset(),
    sample_root: str | None = None,
) -> list[StepSpec]:
    """Bloom (fixed) + one decon step per sampled source.

    ``only_sources`` restricts the decon fan-out to named sources (for smokes);
    StepRunner still walks each decon step's deps, so the matching sample +
    normalize chains run on demand.

    ``exclude_sources`` drops sources from the sample *construction* — the
    proportional fractions are recomputed over the kept set, so this must match
    the exclusion used to build the samples being deconned (e.g. the
    finetranslations + ghalogs/public exclusion behind the CoreWeave testbed's
    115-source ``sample_{100b,500b,1t}_*`` roots). Leave empty for the full
    117-source fractions. Mutually distinct from ``only_sources``, which only
    narrows the fan-out without changing fractions.

    ``sample_root`` decons a **pre-materialized** sample root under MARIN_PREFIX
    (e.g. ``datakit/sample_1t_733c8c5c``) directly — the marks read
    ``<sample_root>/<source>/outputs/main`` instead of re-running the sampler
    (whose step identity has drifted from those roots). ``target_total_tokens_b``
    is then only used to enumerate the source set.
    """
    names = [n for n in all_sources() if n not in exclude_sources]
    if sample_root:
        sampled: dict[str, StepSpec | None] = dict.fromkeys(names)
    else:
        sources = [s for n, s in all_sources().items() if n not in exclude_sources] if exclude_sources else None
        testbed_steps = build_testbed_steps(target_total_tokens_b=target_total_tokens_b, sources=sources)
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
        paragraph_delimiter=PARAGRAPH_DELIMITER,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )

    # One distributed drop-set step for all deconned sources (zephyr shard/source),
    # sourcing DF from the large 1T per-source sample; each decon reads its subdir.
    drop_sets = all_source_drop_sets_step(
        name="datakit/decon_drop/_combined",
        sources=[(name, f"{marin_prefix()}/{SAMPLE_1T_ROOT}/{name}/outputs/main") for name in sampled],
        prebuilt_bloom=bloom,
        ngram_length=NGRAM_LENGTH,
        paragraph_delimiter=PARAGRAPH_DELIMITER,
        sample_docs=DF_SAMPLE_DOCS,
        common_frac=DF_COMMON_FRAC,
        common_min_abs=DF_COMMON_MIN_ABS,
        worker_resources=WORKER_RESOURCES,
    )

    steps: list[StepSpec] = [bloom, drop_sets]
    for name, sample_step in sampled.items():
        input_dir = f"{marin_prefix()}/{sample_root}/{name}/outputs/main" if sample_root else None
        steps.append(
            decon_step(
                name=f"datakit/testbed_decon/{name}",
                normalized=sample_step,
                input_dir=input_dir,
                prebuilt_bloom=bloom,
                drop_sets=drop_sets,
                drop_set_source=name,
                ngram_length=NGRAM_LENGTH,
                overlap_threshold=OVERLAP_THRESHOLD,
                paragraph_delimiter=PARAGRAPH_DELIMITER,
                flagged_sample_size=FLAGGED_SAMPLE_SIZE,
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
    StepRunner().run(
        build_testbed_decon_steps(
            target_total_tokens_b=args.target_tokens_b,
            sample_root=args.sample_root,
            only_sources=args.only,
            exclude_sources=frozenset(args.exclude or ()),
        )
    )


if __name__ == "__main__":
    main()
