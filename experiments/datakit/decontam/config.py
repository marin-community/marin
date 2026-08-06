# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared decontamination policy for the reference pipeline, testbed, and PDF pipeline."""

from rigging.filesystem import marin_prefix

# The shared eval bloom. Every consumer must build it with exactly this step name and these
# parameters: step identity is name + params, so keeping them identical is what makes the ~270 MB
# bloom already built under a prefix a cache hit rather than a rebuild.
BLOOM_STEP_NAME = "datakit/bloom/_combined_fixed"
# The combined eval corpus written by decontam/prepare_eval_corpus.py, staged per-region
# (``aa/<eval>/<split>.parquet`` + ``lmh/<task>/eval.parquet``).
EVAL_ROOT = f"{marin_prefix()}/datakit/decontam/evals"
# Bloom capacity -- unique ngram hashes the filter must hold: ~21.78M unique hashes across the
# AA + LMH corpus, with 2.3x headroom. At FPR=1e-9 this is a ~270 MB filter.
ESTIMATED_DOC_COUNT = 50_000_000
FALSE_POSITIVE_RATE = 1e-9
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
# Contaminated docs reservoir-sampled per shard into the flagged side output the decontam stage
# report reads.
FLAGGED_SAMPLE_SIZE = 8

# Source-local DF needs only enough documents to find high-density boilerplate
# such as legal enacting clauses and license headers.
SOURCE_DF_SAMPLE_DOCS = 5_000
SOURCE_DF_COMMON_FRAC = 0.005
SOURCE_DF_COMMON_MIN_ABS = 5

# Cross-source boilerplate is sparse within each source. Scan a larger prefix,
# then require both corpus document frequency and recurrence across sources.
GLOBAL_DF_SAMPLE_DOCS = 1_000_000
GLOBAL_DF_COMMON_MIN_ABS = 50
GLOBAL_DF_COMMON_MIN_SOURCES = 3
