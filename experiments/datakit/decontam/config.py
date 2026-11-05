# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared decontamination policy for the reference pipeline and testbed."""

from marin.datakit.decon import ItemMatchConfig

# Word n-gram length for eval features and document matching.
NGRAM_LENGTH = 13

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

# Item rule: also mark a document holding at least this share of one listed item's
# indexed 13-grams, not counting drop-set n-grams. GSM8K copies often lack the worked
# answer, and HumanEval copies keep the prompt but rewrite the solution, so their bars
# are lower. ``mmlu_`` covers MMLU subjects and the MMLU-Pro categories. A task's view
# records (``-instruction``, ``-solution``, ``-tests`` ...) are scored on their own, so
# a document holding most of one view is marked. The longest matching prefix wins, so
# ``humanevalplus-`` overrides ``humaneval``; ``piqa-``, ``winogrande-`` and
# ``triviaqa-`` also cover the lm-eval records of those tasks. TriviaQA copies are quiz
# pages that list the question without its answer, and a 13-word question followed by
# a 3-word answer has only 1 of its 4 n-grams in the question, so its bar is 0.2. BoolQ (a
# short question once its Wikipedia passage is left out) and the AA v4.1.1 evals are
# left to the paragraph rule.
ITEM_BAR = 0.5
ITEM_RULE_EVALS: tuple[str, ...] = (
    # AA Intelligence Index v5.0, public parts.
    "gdp_pdf",
    "terminal_bench_4",
    "terminal_bench_science",
    "aa_briefcase_lite",
    "automationbench",
    "aa_omniscience_v1_1",
    "hle_diamond",
    # Post-training eval policy.
    "math500",
    "humanevalplus",
    "mbppplus",
    "olympiadbench",
    "piqa",
    "winogrande",
    "truthfulqa",
    "aime24",
    "cruxeval",
    "ifbench",
    "mrcr",
    "financebench",
    "swebench_verified",
    "ot_tblite",
    "ds1000",
    "bfcl_parity",
    "bixbench",
    "tau3",
    "sotopia_hard",
    "deepswe",
)
ITEM_MATCH = ItemMatchConfig(
    thresholds=(
        ("gsm8k", 0.3),
        ("humaneval", 0.3),
        ("mmlu_", 0.5),
        ("leaderboard_mmlu_pro-", 0.5),
        ("uncheatable-", 0.5),
        ("triviaqa-", 0.2),
        *((f"{name}-", ITEM_BAR) for name in ITEM_RULE_EVALS),
    ),
    min_matched_features=1,
)
