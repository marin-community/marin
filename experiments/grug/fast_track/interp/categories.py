# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Next-token categories for the attribution study, from the token strings and the context (host-side numpy).

Every position ``t`` is labelled by its target ``y = tokens[t + 1]`` (and, for ``copy``, its context): the categories
overlap, and ``all`` covers every scored position.
"""

import string

import numpy as np

CATEGORIES = (
    "all",
    "function_word",
    "word_start",
    "continuation",
    "digit",
    "punctuation",
    "whitespace",
    "code",
    "rare",
    "copy",
)
FUNCTION_WORDS = frozenset(
    """the of and to a in is that for it as with was on be by at this are from or an have not but which
    they his her he she you we i their there been were has had its will would can all one so if more no do
    what who when also about into than them then out up only other some these could my your our him me us""".split()
)
RARE_RANK = 4000
CODE_SOURCE = "dolma_100_programing_languages"
_PUNCT = set(string.punctuation) | set("\u201c\u201d\u2018\u2019\u2014\u2013\u2026\u00ab\u00bb")


def token_features(vocab: list[str]) -> dict[str, np.ndarray]:
    """Per token id booleans: ``function_word``, ``word_start``, ``alpha_cont``, ``digit``, ``punctuation``,
    ``whitespace``, and ``ends_alpha`` (the token's last character is a letter)."""
    n = len(vocab)
    feats = {
        k: np.zeros(n, bool)
        for k in ("function_word", "word_start", "alpha_cont", "digit", "punctuation", "whitespace", "ends_alpha")
    }
    for i, s in enumerate(vocab):
        stripped = s.strip()
        if not s:
            continue
        feats["whitespace"][i] = stripped == ""
        feats["digit"][i] = any(c.isdigit() for c in s)
        feats["punctuation"][i] = stripped != "" and all(c in _PUNCT for c in stripped)
        feats["ends_alpha"][i] = s[-1].isalpha()
        if stripped.isalpha():
            if s[0] == " ":
                feats["function_word"][i] = stripped.lower() in FUNCTION_WORDS
                feats["word_start"][i] = stripped.lower() not in FUNCTION_WORDS
            else:
                feats["alpha_cont"][i] = True
    return feats


def copy_targets(tokens: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """``[B, S]`` True where the bigram ``(tokens[t], tokens[t + 1])`` already occurred earlier in the same document."""
    out = np.zeros(tokens.shape, bool)
    for b in range(tokens.shape[0]):
        seen: set[tuple[int, int, int]] = set()
        row, seg = tokens[b], segment_ids[b]
        for t in range(tokens.shape[1] - 1):
            if seg[t + 1] != seg[t]:
                continue
            key = (int(seg[t]), int(row[t]), int(row[t + 1]))
            out[b, t] = key in seen
            seen.add(key)
    return out


def category_masks(
    tokens: np.ndarray,
    segment_ids: np.ndarray,
    loss_weight: np.ndarray,
    source: str,
    feats: dict[str, np.ndarray],
    freq_rank: np.ndarray,
) -> np.ndarray:
    """``[len(CATEGORIES), B, S]`` float masks of scored positions (``loss_weight > 0`` and a same-document target)."""
    target = np.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    same_doc = np.pad(segment_ids[:, 1:] == segment_ids[:, :-1], ((0, 0), (0, 1)))
    scored = (loss_weight > 0) & same_doc
    cont = feats["alpha_cont"][target] & feats["ends_alpha"][tokens]
    masks = {
        "all": np.ones_like(scored),
        "function_word": feats["function_word"][target],
        "word_start": feats["word_start"][target],
        "continuation": cont,
        "digit": feats["digit"][target],
        "punctuation": feats["punctuation"][target],
        "whitespace": feats["whitespace"][target],
        "code": np.full_like(scored, source == CODE_SOURCE),
        "rare": freq_rank[target] >= RARE_RANK,
        "copy": copy_targets(tokens, segment_ids),
    }
    return np.stack([(masks[c] & scored).astype(np.float32) for c in CATEGORIES])
