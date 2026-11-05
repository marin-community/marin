# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A benchmark extracted from a pinned source into decontamination records."""

import dataclasses
import urllib.request
from collections.abc import Callable
from pathlib import Path

from marin.datakit.decon import DEFAULT_PARAGRAPH_DELIMITER

from experiments.datakit.decontam.config import NGRAM_LENGTH

# A record needs as many words with a letter as one n-gram to be matched at all.
MIN_LETTER_WORDS = NGRAM_LENGTH
DOWNLOAD_TIMEOUT = 120


@dataclasses.dataclass(frozen=True)
class PinnedEval:
    """One benchmark of a mandatory suite.

    Args:
        name: Benchmark name in the suite manifest.
        subdir: Artifact directory; also the record-id prefix (``<subdir>-<item>``).
        source: URL or repository of the pinned source.
        revision: Commit, dataset revision or content digest of the source.
        expected_records: Records the extraction must produce.
        build: Returns the ``{id, text}`` records, given a download cache directory.
    """

    name: str
    subdir: str
    source: str
    revision: str
    expected_records: int
    build: Callable[[Path], list[dict[str, str]]]


def letter_words(text: str) -> int:
    return sum(1 for word in text.split() if any(ch.isalpha() for ch in word))


def fetch_cached(url: str, path: Path) -> bytes:
    """Return the bytes at ``url``, downloading them to ``path`` on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def markable(text: str) -> str:
    """Return ``text`` as one paragraph when none of its paragraphs holds two n-grams.

    The paragraph rule needs two matched n-grams in one paragraph, so an exact copy of a
    record made of short paragraphs (a PIQA goal and its two solutions) is never marked.
    Joining them with single line breaks keeps the record's words and lets its n-grams
    span the former paragraph breaks.
    """
    paragraphs = [p for p in text.split(DEFAULT_PARAGRAPH_DELIMITER) if p.strip()]
    if any(len(p.split()) > NGRAM_LENGTH for p in paragraphs):
        return text
    return "\n".join(p.strip() for p in paragraphs)
