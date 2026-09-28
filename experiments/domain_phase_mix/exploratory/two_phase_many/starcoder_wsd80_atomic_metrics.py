# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic BPB metrics shared by the StarCoder WSD80 surface explorers."""

from __future__ import annotations

import json

ATOMIC_METRICS = (
    ("eval/paloma/dolma_100_programing_languages-llama3/bpb", "Paloma · Programming Languages"),
    ("eval/uncheatable_eval/github_cpp-llama3/bpb", "Uncheatable · GitHub C++"),
    ("eval/uncheatable_eval/github_python-llama3/bpb", "Uncheatable · GitHub Python"),
    ("eval/uncheatable_eval/arxiv_computer_science-llama3/bpb", "Uncheatable · ArXiv Computer Science"),
    ("eval/uncheatable_eval/arxiv_physics-llama3/bpb", "Uncheatable · ArXiv Physics"),
    ("eval/uncheatable_eval/wikipedia_english-llama3/bpb", "Uncheatable · Wikipedia English"),
    ("eval/uncheatable_eval/bbc_news-llama3/bpb", "Uncheatable · BBC News"),
    ("eval/uncheatable_eval/ao3_english-llama3/bpb", "Uncheatable · AO3 English"),
    ("eval/paloma/4chan-llama3/bpb", "Paloma · 4chan"),
    ("eval/paloma/c4_100_domains-llama3/bpb", "Paloma · C4 100 Domains"),
    ("eval/paloma/c4_en-llama3/bpb", "Paloma · C4 English"),
    ("eval/paloma/dolma-v1_5-llama3/bpb", "Paloma · Dolma 1.5"),
    ("eval/paloma/dolma_100_subreddits-llama3/bpb", "Paloma · Dolma 100 Subreddits"),
    ("eval/paloma/falcon-refinedweb-llama3/bpb", "Paloma · Falcon RefinedWeb"),
    ("eval/paloma/gab-llama3/bpb", "Paloma · Gab"),
    ("eval/paloma/m2d2_s2orc_unsplit-llama3/bpb", "Paloma · M2D2 S2ORC"),
    ("eval/paloma/m2d2_wikipedia_unsplit-llama3/bpb", "Paloma · M2D2 Wikipedia"),
    ("eval/paloma/manosphere_meta_sep-llama3/bpb", "Paloma · Manosphere"),
    ("eval/paloma/mc4-llama3/bpb", "Paloma · mC4"),
    ("eval/paloma/ptb-llama3/bpb", "Paloma · Penn Treebank"),
    ("eval/paloma/redpajama-llama3/bpb", "Paloma · RedPajama"),
    ("eval/paloma/twitterAAE_HELM_fixed-llama3/bpb", "Paloma · TwitterAAE"),
    ("eval/paloma/wikitext_103-llama3/bpb", "Paloma · WikiText-103"),
)
METRIC_KEYS = tuple(key for key, _label in ATOMIC_METRICS)
DEFAULT_METRIC_KEY = METRIC_KEYS[0]


def final_atomic_metrics(text: str, *, source: str) -> dict[str, float]:
    """Extract the final complete atomic-metric row from an eval JSONL file."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No evaluation rows in {source}")
    payload = json.loads(lines[-1])
    missing = [key for key in METRIC_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{source} is missing atomic metrics: {missing}")
    return {key: float(payload[key]) for key in METRIC_KEYS}
