# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize a seeded sample of a candidate dataset's files into a Levanter cache for the mixture runs.

The sample is a seeded random subset of the split's data files that holds at least ``FILE_MARGIN`` times the new-data
tokens of the largest rung, by the intake estimate. Whole files keep the dataset's documents whole, and a random subset
avoids reading only the head of a dataset whose files are ordered (by date, source or score). Each tokenize shard stops
after ``max_rows`` documents, so that a dataset with a few huge files is not tokenized whole.
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
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.data_belay.intake import IntakeReport

logger = logging.getLogger(__name__)

CACHE_ROOT = "s3://marin-us-east-02a/marin/data_belay/caches"
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
    """A seeded random subset of ``files`` whose estimated tokens reach ``FILE_MARGIN * need_tokens``."""
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


def plan_cache(report: IntakeReport, need_tokens: int, seed: int = SAMPLE_SEED) -> CachePlan:
    """Choose the files for ``report``'s dataset and the cache path that names them."""
    data_files = load_dataset_builder(report.repo, report.config, revision=report.revision).config.data_files
    files = drop_sidecars(glob_with_sizes(list(data_files[report.split])))
    total_bytes = sum(f.size for f in files)
    if total_bytes <= 0:
        raise ValueError(f"{report.repo} split {report.split!r} lists no file sizes")
    chosen = select_files(files, report.estimated_tokens / total_bytes, need_tokens, seed)
    paths = tuple(sorted(f.path for f in chosen))
    max_rows = math.ceil(FILE_MARGIN * need_tokens / report.sample.tokens_per_row)
    identity = json.dumps(
        {"files": paths, "max_rows": max_rows, "text_field": report.text_field, "tokenizer": "hero-bpe-v16384"},
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    slug = report.repo.replace("/", "--")
    name = f"{report.config or 'default'}-{report.split}-{digest}"
    estimated = round(sum(f.size for f in chosen) * report.estimated_tokens / total_bytes)
    estimated = min(estimated, round(max_rows * report.sample.tokens_per_row))
    path = prefix_join(CACHE_ROOT, f"{slug}/{report.revision[:12]}/{name}")
    return CachePlan(path, paths, max_rows, estimated, need_tokens)


def build_cache(report: IntakeReport, plan: CachePlan, tokenizer: str) -> dict:
    """Tokenize the plan's files (skipped when the cache exists) and write the cache manifest."""
    config = TokenizeConfig(
        train_paths=list(plan.files),
        validation_paths=[],
        cache_path=plan.cache_path,
        tokenizer=tokenizer,
        format=TextLmDatasetFormat(text_key=report.text_field),
        sample_count=plan.max_rows,
    )
    tokenize(config)
    stats = json.loads(StoragePath(prefix_join(plan.train_cache, ".stats.json")).read_text())
    manifest = {"intake": json.loads(report.to_json()), "plan": dataclasses.asdict(plan), "stats": stats}
    StoragePath(prefix_join(plan.cache_path, MANIFEST_NAME)).write_text(json.dumps(manifest, indent=2))
    logger.info("cache %s: %s", plan.train_cache, stats)
    return manifest
