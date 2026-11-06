# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize a seeded sample of a candidate dataset's files into a Levanter cache for the mixture runs.

The sample selects files in seeded random order until their estimated tokens reach ``FILE_MARGIN`` times the largest
rung's new-data tokens, or the files run out. Whole files preserve documents; random selection samples across files
ordered by date, source or score. Each tokenize shard stops after ``max_rows`` documents.
"""

import dataclasses
import hashlib
import json
import logging
import math
import random

from datasets import load_dataset_builder
from levanter.data.text.formats import TextLmDatasetFormat
from marin.processing.tokenize._core import FileEntry, drop_sidecars, glob_with_sizes
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.data_belay.intake import MAX_EPOCHS, IntakeReport

logger = logging.getLogger(__name__)

# Candidate caches live under the training cluster's MARIN_PREFIX, beside the Track's base cache.
CACHE_PATH = "data_belay/caches"
FILE_MARGIN = 1.5
SAMPLE_SEED = 0
MANIFEST_NAME = "data_belay_cache.json"


@dataclasses.dataclass(frozen=True)
class CachePlan:
    """Where a candidate's cache goes and which files it holds."""

    cache_path: str
    files: tuple[str, ...]
    max_rows: int
    """Documents per tokenize shard: ``FILE_MARGIN`` times the rows that hold the need, by the intake estimate."""
    estimated_tokens: int
    need_tokens: int

    @property
    def train_cache(self) -> str:
        """The flat cache root that the mixture's new-data component reads."""
        return prefix_join(self.cache_path, "train")


def select_files(files: list[FileEntry], tokens_per_byte: float, need_tokens: int, seed: int) -> list[FileEntry]:
    """Select files until their estimated tokens reach ``FILE_MARGIN * need_tokens`` or the files run out."""
    order = sorted(files, key=lambda f: f.path)
    random.Random(seed).shuffle(order)
    target = FILE_MARGIN * need_tokens
    chosen: list[FileEntry] = []
    tokens = 0.0
    for entry in order:
        if tokens >= target:
            break
        chosen.append(entry)
        tokens += entry.size * tokens_per_byte
    return chosen


def plan_cache(
    report: IntakeReport, need_tokens: int, tokenizer: str, cache_root: str, seed: int = SAMPLE_SEED
) -> CachePlan:
    """Choose the files for ``report``'s dataset and the cache path under ``cache_root`` that names them."""
    data_files = load_dataset_builder(report.repo, report.config, revision=report.revision).config.data_files
    files = drop_sidecars(glob_with_sizes(list(data_files[report.split])))
    total_bytes = sum(f.size for f in files)
    if total_bytes <= 0:
        raise ValueError(f"{report.repo} split {report.split!r} lists no file sizes")
    chosen = select_files(files, report.estimated_tokens / total_bytes, need_tokens, seed)
    paths = tuple(sorted(f.path for f in chosen))
    max_rows = math.ceil(FILE_MARGIN * need_tokens / report.sample.tokens_per_row)
    identity = json.dumps(
        {"files": paths, "max_rows": max_rows, "text_field": report.text_field, "tokenizer": tokenizer},
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    slug = report.repo.replace("/", "--")
    name = f"{report.config or 'default'}-{report.split}-{digest}"
    estimated = round(sum(f.size for f in chosen) * report.estimated_tokens / total_bytes)
    estimated = min(estimated, round(max_rows * report.sample.tokens_per_row))
    path = prefix_join(cache_root, f"{slug}/{report.revision[:12]}/{name}")
    return CachePlan(path, paths, max_rows, estimated, need_tokens)


def cache_epochs(need_tokens: int, cache_tokens: int, dataset_tokens: int) -> float:
    """Epochs of the built cache that the largest rung reads; raises when the cache is too small for the need.

    A dataset that holds the need must give a cache that holds it too, so it is read at most once. A smaller dataset
    repeats, but at most ``MAX_EPOCHS`` times: intake estimates its epochs over the whole dataset, and this checks
    them on the tokens that training actually reads.
    """
    epochs = need_tokens / cache_tokens
    if epochs > MAX_EPOCHS:
        raise ValueError(f"the cache's {cache_tokens:,} tokens repeat {epochs:.2f} times for a need of {need_tokens:,}")
    if dataset_tokens >= need_tokens and epochs > 1:
        raise ValueError(
            f"the cache holds {cache_tokens:,} tokens, less than the need of {need_tokens:,} of a dataset of about "
            f"{dataset_tokens:,} tokens; the intake estimate was too high for the sampled files"
        )
    return epochs


def build_cache(report: IntakeReport, plan: CachePlan, tokenizer: str) -> dict:
    """Tokenize the plan's files (skipped when the cache exists), check its epochs, then write the cache manifest.

    ``train`` reads its share from the manifest, so a cache without one cannot be trained on.
    """
    config = TokenizeConfig(
        train_paths=list(plan.files),
        validation_paths=[],
        cache_path=plan.cache_path,
        tokenizer=tokenizer,
        format=TextLmDatasetFormat(text_key=report.text_field),
        sample_count=plan.max_rows,
    )
    tokenize(config)
    stats = read_tokenized_cache_stats(plan.cache_path, "train")
    epochs = cache_epochs(plan.need_tokens, stats.total_tokens, report.estimated_tokens)
    manifest = {
        "intake": json.loads(report.to_json()),
        "plan": dataclasses.asdict(plan),
        "stats": dataclasses.asdict(stats),
        "epochs_at_need": epochs,
    }
    StoragePath(prefix_join(plan.cache_path, MANIFEST_NAME)).write_text(json.dumps(manifest, indent=2))
    logger.info("cache %s: %s", plan.train_cache, stats)
    return manifest


def cache_share(train_cache: str) -> float:
    """The share that ``train_cache`` was prepared for and its epochs were checked at, from its manifest."""
    cache_path = train_cache.rstrip("/").removesuffix("/train")
    manifest = json.loads(StoragePath(prefix_join(cache_path, MANIFEST_NAME)).read_text())
    return float(manifest["intake"]["share"])
