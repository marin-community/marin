# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search settings shared by Echo's schema and query statements."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

PUBLIC_URL = "https://echo.oa.dev"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INDEXED_REPOSITORY = "marin-community/marin"
INDEXED_BRANCH = "main"
DISPLAY_SHA_CHARACTERS = 12
FEDERATED_SUMMARY_CHARACTERS = 240
SearchDomain = Literal["wiki", "file", "discord", "pr", "issue"]
SEARCH_DOMAINS: tuple[SearchDomain, ...] = ("wiki", "file", "discord", "pr", "issue")
DEFAULT_SEARCH_DOMAINS: tuple[SearchDomain, ...] = ("wiki", "file", "pr", "issue")
SEARCH_DOMAIN_LABELS: Mapping[SearchDomain, str] = MappingProxyType(
    {
        "wiki": "Wiki",
        "file": "Files",
        "discord": "Discord",
        "pr": "Pull requests",
        "issue": "Issues",
    }
)
TEXT_SEARCH_CONFIG = "english"
TS_RANK_NORMALIZATION = 32
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
MAX_SEMANTIC_DISTANCE = 0.45
RRF_K = 60
MIN_CANDIDATES = 40
CANDIDATE_MULTIPLIER = 4
FILE_CHUNK_CANDIDATE_MULTIPLIER = 4
FILE_ADDITIONAL_HIT_WEIGHT = 0.2
FILE_ADDITIONAL_HIT_MAX_FRACTION = 0.3
QUERY_PROSE_FILE_SCORE_MULTIPLIER = 1.15
QUERY_TEST_FILE_SCORE_MULTIPLIER = 0.85
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2-int8"
RERANK_MODEL_SOURCE = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_MODEL_FILE = "onnx/model_int8.onnx"
RERANK_MIN_RESULTS_PER_DOMAIN = 20
RERANK_MAX_CANDIDATES = 20
RERANK_BASE_WEIGHT = 0.2
RERANK_MODEL_WEIGHT = 0.8
MIN_RERANK_SCORE = -2.0
PROSE_FILE_SUFFIXES = (".md", ".rst")
IDENTIFIER_QUERY_PATTERN = re.compile(r"[/_.:]|(?:[a-z][A-Z])|(?:^|\s)--?[a-z0-9]")
QUERY_WORD_PATTERN = re.compile(r"[a-z0-9]+")
LOG_QUERY_TERMS = frozenset({"log", "logging", "logs"})
LOG_QUERY_CONTEXT_TERMS = frozenset({"job", "jobs", "query", "run", "runs", "train", "training"})
LOG_QUERY_EXPANSION = "Relevant Marin terms: finelog iris job logs"
KV_QUERY_TERMS = frozenset({"kv"})
KV_CACHE_TERMS = frozenset({"cache", "caches", "caching"})
KV_QUERY_EXPANSION = "Levanter inference KvPageCache kv_cache"
PROSE_QUERY_MIN_WORDS = 3


@dataclass(frozen=True)
class SearchWeights:
    semantic: float
    lexical: float


QUERY_SEARCH_WEIGHTS = SearchWeights(semantic=2.0, lexical=1.0)
IDENTIFIER_SEARCH_WEIGHTS = SearchWeights(semantic=1.0, lexical=2.0)


def candidate_limit(limit: int) -> int:
    return max(MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER)


def is_identifier_query(query: str) -> bool:
    return IDENTIFIER_QUERY_PATTERN.search(query) is not None


def search_weights(query: str) -> SearchWeights:
    """Prefer semantic rank for prose and lexical rank for identifiers or terse keywords."""
    if not is_identifier_query(query) and len(query.split()) >= PROSE_QUERY_MIN_WORDS:
        return QUERY_SEARCH_WEIGHTS
    return IDENTIFIER_SEARCH_WEIGHTS


def expanded_query(query: str) -> str:
    """Add Marin vocabulary when a prose question names a known subsystem indirectly."""
    if is_identifier_query(query):
        return query
    terms = set(QUERY_WORD_PATTERN.findall(query.casefold()))
    if terms & KV_QUERY_TERMS and terms & KV_CACHE_TERMS:
        return f"{KV_QUERY_EXPANSION}\nUser query: {query}"
    if terms & LOG_QUERY_TERMS and terms & LOG_QUERY_CONTEXT_TERMS:
        return f"{query}\n{LOG_QUERY_EXPANSION}"
    return query
