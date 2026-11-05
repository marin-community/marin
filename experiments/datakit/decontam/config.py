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
# ``triviaqa-`` also cover the lm-eval records of those tasks. TriviaQA keeps the 0.5 bar
# although copies that list the question without its answer fall under it: its questions
# often quote a famous line or definition, and a lower bar marks every page quoting it.
# TruthfulQA copies put lettered options between a long question and its short best
# answer, so only the question's n-grams match, and its bar is 0.4. BoolQ, whose records are mostly a Wikipedia
# passage, and the AA v4.1.1 evals are left to the paragraph rule.
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
    "triviaqa",
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
        ("truthfulqa-", 0.4),
        *((f"{name}-", ITEM_BAR) for name in ITEM_RULE_EVALS),
    ),
    min_matched_features=1,
)
