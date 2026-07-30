# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared decontamination policy for the reference pipeline and testbed."""

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
