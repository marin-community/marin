# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Intake checks for a Data Belay candidate: pinned revision, license, text field and size.

A candidate is a Hugging Face dataset at its head commit. The check pins that commit, reads the license from the
dataset card, and estimates the dataset's tokens under the fast-track tokenizer from a streamed sample, so that the
report can say how many epochs each ladder rung would read at the requested share.
"""

import dataclasses
import fnmatch
import json
import math
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

import pyarrow.parquet as pq
import requests
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi, HfFileSystem, get_token
from levanter.tokenizers import MarinTokenizer

# Licenses in Marin's source registry (marin-community/token-counts) that allow training and redistribution of a
# derived model, as HF card ids. A pattern matches the lowercase card id.
ALLOWED_LICENSE_PATTERNS = (
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-*",
    "isc",
    "cc0-1.0",
    "cc-by-[0-9].[0-9]",
    "cc-by-sa-[0-9].[0-9]",
    "odc-by",
    "pddl",
    "odbl",
    "openmdw",
    "unlicense",
    "public-domain",
)
# Card ids that name a license a person must read: custom terms, or no stated terms.
REVIEW_LICENSES = frozenset({"other", "unknown"})
# Card id fragments that forbid a use Marin needs: non-commercial or no derivatives.
REJECTED_LICENSE_FRAGMENTS = ("-nc", "-nd", "noncommercial")

DATASETS_SERVER_SIZE_URL = "https://datasets-server.huggingface.co/size"
SAMPLE_ROWS = 2000
# The check refuses a share that would read the new data more times than this at the largest rung that runs.
MAX_EPOCHS = 4.0


class LicenseVerdict(StrEnum):
    ALLOWED = "allowed"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


@dataclasses.dataclass(frozen=True)
class DatasetRequest:
    """The dataset a person asks the Data Belay to test."""

    repo: str
    config: str | None
    split: str
    text_field: str
    license: str | None = None
    """The license a person found in the dataset's README or terms, for a card that lists none."""


@dataclasses.dataclass(frozen=True)
class SampleStats:
    rows: int
    text_bytes: int
    tokens: int

    @property
    def tokens_per_row(self) -> float:
        return self.tokens / self.rows


@dataclasses.dataclass(frozen=True)
class RungNeed:
    rung: str
    rung_tokens: int
    new_tokens: int
    epochs: float


@dataclasses.dataclass(frozen=True)
class IntakeReport:
    repo: str
    revision: str
    config: str
    split: str
    text_field: str
    licenses: tuple[str, ...]
    license_from_card: bool
    """False when the card lists no license and ``licenses`` is the one a person gave."""
    license_verdict: LicenseVerdict
    gated: bool
    num_rows: int
    sample: SampleStats
    estimated_tokens: int
    share: float
    needs: tuple[RungNeed, ...]

    @property
    def passed(self) -> bool:
        return self.license_verdict is LicenseVerdict.ALLOWED and all(n.epochs <= MAX_EPOCHS for n in self.needs)

    def to_json(self) -> str:
        return json.dumps({**dataclasses.asdict(self), "passed": self.passed}, indent=2)


def license_verdict(licenses: tuple[str, ...]) -> LicenseVerdict:
    """Classify a dataset card's license ids; every id must be allowed for the dataset to be allowed."""
    if not licenses:
        return LicenseVerdict.NEEDS_REVIEW
    verdicts = {_one_license_verdict(card_id.strip().lower()) for card_id in licenses}
    for verdict in (LicenseVerdict.REJECTED, LicenseVerdict.NEEDS_REVIEW):
        if verdict in verdicts:
            return verdict
    return LicenseVerdict.ALLOWED


def _one_license_verdict(card_id: str) -> LicenseVerdict:
    if any(fragment in card_id for fragment in REJECTED_LICENSE_FRAGMENTS):
        return LicenseVerdict.REJECTED
    if card_id in REVIEW_LICENSES:
        return LicenseVerdict.NEEDS_REVIEW
    if any(fnmatch.fnmatchcase(card_id, pattern) for pattern in ALLOWED_LICENSE_PATTERNS):
        return LicenseVerdict.ALLOWED
    return LicenseVerdict.NEEDS_REVIEW


def rung_needs(estimated_tokens: int, share: float, rung_tokens: dict[str, int]) -> tuple[RungNeed, ...]:
    """New-data tokens and epochs of the dataset at each rung for ``share`` of the rung's tokens."""
    if not 0.0 < share <= 1.0:
        raise ValueError(f"share must be in (0, 1], got {share}")
    needs = []
    for rung, tokens in rung_tokens.items():
        new_tokens = math.ceil(share * tokens)
        needs.append(RungNeed(rung, tokens, new_tokens, new_tokens / estimated_tokens))
    return tuple(needs)


def card_licenses(info) -> tuple[str, ...]:
    card = info.card_data.to_dict() if info.card_data is not None else {}
    card_license = card.get("license")
    if card_license is None:
        return tuple(tag.removeprefix("license:") for tag in info.tags or () if tag.startswith("license:"))
    return tuple(card_license) if isinstance(card_license, list) else (card_license,)


def _num_rows(repo: str, config: str, split: str) -> int:
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(DATASETS_SERVER_SIZE_URL, params={"dataset": repo}, headers=headers, timeout=60)
    response.raise_for_status()
    splits = response.json()["size"]["splits"]
    matches = [s for s in splits if s["split"] == split and s["config"] == config]
    if not matches:
        available = sorted({(s["config"], s["split"]) for s in splits})
        raise ValueError(f"{repo} has no split {split!r} in config {config!r}; available (config, split): {available}")
    return sum(s["num_rows"] for s in matches)


def _first_rows(repo: str, sha: str, config: str, split: str, text_field: str) -> list[dict[str, Any]]:
    """The first ``SAMPLE_ROWS`` rows of the split at commit ``sha``."""
    data_files = load_dataset_builder(repo, config, revision=sha).config.data_files
    if split not in data_files:
        raise ValueError(f"{repo} config {config!r} has no split {split!r}; splits: {sorted(data_files)}")
    files = list(data_files[split])
    if not all(path.endswith(".parquet") for path in files):
        rows = load_dataset(repo, config, split=split, streaming=True, revision=sha)
        return list(rows.take(SAMPLE_ROWS))
    # Read Parquet by row groups, so the sample costs a few range requests.
    fs = HfFileSystem()
    rows: list[dict[str, Any]] = []
    for path in files:
        with fs.open(path) as handle:
            parquet = pq.ParquetFile(handle)
            if text_field not in parquet.schema_arrow.names:
                raise ValueError(f"{path} has no column {text_field!r}; columns: {parquet.schema_arrow.names}")
            for batch in parquet.iter_batches(batch_size=SAMPLE_ROWS - len(rows), columns=[text_field]):
                rows.extend(batch.to_pylist())
                if len(rows) >= SAMPLE_ROWS:
                    return rows[:SAMPLE_ROWS]
    return rows


def sample_stats(rows: Iterable[Mapping[str, Any]], text_field: str, tokenizer: MarinTokenizer) -> SampleStats:
    """Count the UTF-8 bytes and tokens of the text of ``rows``."""
    texts = []
    for index, row in enumerate(rows):
        text = row.get(text_field)
        if not isinstance(text, str):
            string_fields = sorted(name for name, value in row.items() if isinstance(value, str))
            raise ValueError(f"row {index} has no string field {text_field!r}; string fields: {string_fields}")
        texts.append(text)
    if not texts:
        raise ValueError("the dataset split has no rows")
    # BOS and EOS per document, as the training caches store them.
    tokens = sum(len(ids) + 2 for ids in tokenizer.encode_batch(texts))
    return SampleStats(len(texts), sum(len(text.encode("utf-8")) for text in texts), tokens)


def run_intake(
    request: DatasetRequest, share: float, rung_tokens: dict[str, int], tokenizer: MarinTokenizer
) -> IntakeReport:
    """Pin the head commit, classify the license and estimate the size of ``request``.

    The head only, because the row count comes from the datasets-server, which sizes the head.
    """
    info = HfApi().dataset_info(request.repo)
    licenses = card_licenses(info)
    if request.license is not None:
        if licenses:
            raise ValueError(f"{request.repo}'s card lists {licenses}; --license is only for a card that lists none")
        licenses = (request.license,)
    # One config name for the sample, the row count and the cache plan: with no config given, the builder's default.
    # The size endpoint lists every config, so counting rows without a name would sum all of them.
    config = load_dataset_builder(request.repo, request.config, revision=info.sha).config.name
    sample = sample_stats(
        _first_rows(request.repo, info.sha, config, request.split, request.text_field),
        request.text_field,
        tokenizer,
    )
    num_rows = _num_rows(request.repo, config, request.split)
    estimated_tokens = round(sample.tokens_per_row * num_rows)
    return IntakeReport(
        repo=request.repo,
        revision=info.sha,
        config=config,
        split=request.split,
        text_field=request.text_field,
        licenses=licenses,
        license_from_card=request.license is None,
        license_verdict=license_verdict(licenses),
        gated=bool(info.gated),
        num_rows=num_rows,
        sample=sample,
        estimated_tokens=estimated_tokens,
        share=share,
        needs=rung_needs(estimated_tokens, share, rung_tokens),
    )
