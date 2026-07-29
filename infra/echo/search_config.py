# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search settings shared by Echo's schema and query statements."""

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TEXT_SEARCH_CONFIG = "english"
TS_RANK_NORMALIZATION = 32
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
MAX_SEMANTIC_DISTANCE = 0.45
RRF_K = 60
LEXICAL_WEIGHT = 2.0
MIN_CANDIDATES = 40
CANDIDATE_MULTIPLIER = 4


def candidate_limit(limit: int) -> int:
    return max(MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER)
