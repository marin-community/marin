# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared decontamination policy for the reference pipeline and testbed."""

from marin.datakit.decon import ItemMatchConfig

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

# Item rule (marin#6852): also mark a document holding at least this share of one
# core or Uncheatable item's indexed 13-grams, not counting drop-set n-grams. GSM8K
# and HumanEval copies often lack the worked answer or the tests, so their bar is
# lower. Other evals are left to the paragraph rule; the rule has not been tuned
# on them. ``mmlu_`` covers MMLU subjects and the MMLU-Pro categories.
ITEM_MATCH = ItemMatchConfig(
    thresholds=(
        ("gsm8k", 0.3),
        ("humaneval", 0.3),
        ("mmlu_", 0.5),
        ("leaderboard_mmlu_pro-", 0.5),
        ("uncheatable-", 0.5),
    ),
    min_matched_features=1,
)
