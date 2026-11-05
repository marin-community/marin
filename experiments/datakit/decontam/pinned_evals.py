# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A benchmark extracted from a pinned source into decontamination records."""

import dataclasses
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from huggingface_hub import hf_hub_download
from marin.datakit.decon import DEFAULT_PARAGRAPH_DELIMITER

from experiments.datakit.decontam.config import NGRAM_LENGTH

# A record needs as many words with a letter as one n-gram to be matched at all.
MIN_LETTER_WORDS = NGRAM_LENGTH
DOWNLOAD_TIMEOUT = 120
# A part of an item gets its own view record only with this many words with a letter,
# so that one or two shared n-grams cannot carry a view over the item-rule bar.
MIN_VIEW_WORDS = 30
MAX_RECORD_CHARS = 200_000


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


def paragraphs(parts: Sequence[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def item_records(
    name: str, item_id: str, parts: Sequence[str], views: Mapping[str, str] | None = None
) -> list[dict[str, str]]:
    """One item's records: ``parts`` joined as the full record, then each view long enough to stand alone.

    The full record's id is ``<name>-<item_id>`` and a view's ``<name>-<item_id>-<view>``.
    The item is skipped when the full record has fewer than ``MIN_LETTER_WORDS`` words with
    a letter; a view needs ``MIN_VIEW_WORDS`` and must differ from the full record. Every
    record is cut to ``MAX_RECORD_CHARS``.
    """
    record_id = f"{name}-{item_id}"
    full = paragraphs(parts)
    if letter_words(full) < MIN_LETTER_WORDS:
        return []
    rows = [{"id": record_id, "text": full[:MAX_RECORD_CHARS]}]
    for view, text in (views or {}).items():
        text = text.strip()
        if text != full and letter_words(text) >= MIN_VIEW_WORDS:
            rows.append({"id": f"{record_id}-{view}", "text": text[:MAX_RECORD_CHARS]})
    return rows


def fetch_cached(url: str, path: Path) -> bytes:
    """Return the bytes at ``url``, downloading them to ``path`` on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def hf_file(repo_rev: tuple[str, str], filename: str, cache_dir: Path) -> str:
    """Return the local path of ``filename`` in a Hugging Face dataset repo at a pinned revision."""
    repo, revision = repo_rev
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir / "hf"))


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
