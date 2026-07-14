# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global decontamination across every Datakit source.

DAG shape:

    EVAL_ROOT (gs://.../datakit/decontam/evals/, parquet tree)
        │
        ▼ build_eval_bloom_step (single combined bloom over the whole tree)
        │     (datakit/bloom/_combined)
        │
        ▼ one decon_step per Datakit source, consuming the combined bloom
              (datakit/decontam/<source>)

We deliberately build a single combined bloom rather than per-eval blooms +
merge. The eval corpus contains ~850 leaves (8 AA + ~850 lm-eval-harness
group-expanded leaves under ``prepare_eval_corpus.py``); per-eval would
mean ~850 bloom build steps, all sharing identical sizing, which is pure
DAG overhead without real cache-reuse value. Adding or swapping one eval
invalidates the single bloom plus all 104 corpus marks — acceptable
for the cadence at which the eval set changes.

Decon outputs land at ``gs://marin-eu-west4/datakit/decontam/<source>_<hash>/``
(no ``output_path_prefix`` override — the iris worker's ``MARIN_PREFIX``
is ``gs://marin-eu-west4`` and the step name carries the rest of the
path). Persistent (no TTL).

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``):

    uv run iris --cluster=marin job run --region europe-west4 \\
        --extra=cpu --extra=eval --priority interactive \\
        --memory 8GB --cpu 2 --enable-extra-resources \\
        -- python experiments/datakit/decontam/all_sources_decon.py
"""

import logging

from fray.types import ResourceConfig
from marin.datakit.decon import build_eval_bloom_step, decon_step
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS

logger = logging.getLogger(__name__)


# Combined eval corpus written by ``prepare_eval_corpus.py``. Resolves to the
# in-region bucket via ``marin_prefix()`` (auto-detected from GCS metadata on
# iris workers), so a worker in eu-west4 reads from gs://marin-eu-west4/... and
# a worker in us-central2 reads from gs://marin-us-central2/.... The corpus
# must be staged in the region you're running -- copy once with
# ``gsutil -m cp -r`` if it's not there yet. Two top-level subtrees:
#
#   aa/<eval>/<split>.parquet     (8 evals, AA Intelligence Index v4.0)
#   lmh/<task>/eval.parquet       (lm-eval-harness leaf tasks, ~850)
EVAL_ROOT = f"{marin_prefix()}/datakit/decontam/evals"

# Bloom capacity — unique ngram hashes the filter must hold. Sized from
# ``count_docs.py`` against the parquet eval corpus (2026-05-17 run):
# 1.8M eval records, 98M total ngram inserts, **21.78M unique ngram
# hashes** across the AA + LMH corpus. 2.3x headroom for stability.
# At FPR=1e-9 this gives a ~270 MB bloom filter.
ESTIMATED_DOC_COUNT = 50_000_000
FALSE_POSITIVE_RATE = 1e-9
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")


def build_decon_steps() -> list[StepSpec]:
    # One combined bloom from the parquet tree under EVAL_ROOT. Decon
    # workers consume this directly — no merge step. ``exclude_eval_dirs``
    # drops the answerless corpus-material eval tasks at read time, so an
    # already-materialized eval corpus that still contains those dirs is
    # excluded without regenerating it (see marin#6852). Because the set is
    # folded into the step hash, the bloom rebuilds at a fresh path.
    combined_bloom = build_eval_bloom_step(
        name="datakit/bloom/_combined",
        eval_data_sources=[EVAL_ROOT],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )

    # Per-corpus decon steps, all consuming the same combined bloom.
    # Step name is the routing path (under marin_prefix on the iris worker),
    # so outputs land at gs://marin-eu-west4/datakit/decontam/<source>_<hash>/.
    return [
        decon_step(
            name=f"datakit/decontam/{name}",
            normalized=src.normalized,
            prebuilt_bloom=combined_bloom,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
            worker_resources=WORKER_RESOURCES,
        )
        for name, src in all_sources().items()
    ]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_decon_steps())
