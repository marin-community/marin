# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-specific training exclusions and retained MarinSkyRL admission exceptions."""

CROSSCODE_EVAL_SOURCE_IDS: frozenset[str] = frozenset(
    {
        "TaskTrove:laion__exp_rpt_crosscodeeval-csharp-v4",
        "TaskTrove:laion__exp_rpt_crosscodeeval-java-v3",
        "TaskTrove:laion__exp_rpt_crosscodeeval-python-v2",
        "TaskTrove:laion__exp_rpt_crosscodeeval-typescript-v2",
    }
)
CROSSCODE_EVAL_REASON = (
    "Evaluation only for CrossCodeEval-derived sources: Python's native tests contain the Terminal-Bench canary. "
    "An inspected Python tests/test_state.py marker is preserved at private pin "
    "a953f93bc830704525ff2cce0b296dd262009662, file SHA-256 "
    "335184f930fd957094b86f8f5402465766c40674de9044e2feb6faf21d904789. "
    "Within-source dedup applies; lexical contamination is measured only."
)

# Ruling 2026-11-23: a source superseded by a replacement version is Deprecated
# with a typed link to its replacement, never Excluded. The replacements are
# declared sources in the Atlas; the audit refuses a link that does not resolve.
SUPERSEDED_REPLACEMENTS: dict[str, str] = {
    "TaskTrove:laion__swesmith-oracle-filtered-v2": "TaskTrove:swesmith-verifier-repaired-v5",
    "TaskTrove:smartie-swesmith-admitted": "TaskTrove:swesmith-verifier-repaired-v5",
    # Ruling 2026-11-24: the shared-v2 Harbor cohort's data is already served as
    # the rebuilt TaskSpec variant; the Harbor population is superseded, not
    # re-converted.
    "TaskTrove:smartie-calibforge-shared-v2": "TaskTrove:smartie-calibforge-rebuilt-v1",
}
