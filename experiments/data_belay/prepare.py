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
import typing

from datasets import load_dataset_builder
from levanter.data.text.formats import TextLmDatasetFormat
from marin.processing.tokenize._core import FileEntry, drop_sidecars, glob_with_sizes
from marin.processing.tokenize.cache_stats import TokenizedCacheStats, read_tokenized_cache_stats
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.data_belay.intake import MAX_EPOCHS, IntakeReport

logger = logging.getLogger(__name__)

# Candidate caches live under the training cluster's MARIN_PREFIX, beside the base cache.
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
    """Documents per tokenize shard: ``FILE_MARGIN`` times the rows that intake estimates for ``need_tokens``."""
    estimated_tokens: int
    need_tokens: int


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


def hf_files(report: IntakeReport) -> list[FileEntry]:
    """The data files of ``report``'s split at its pinned revision."""
    data_files = load_dataset_builder(report.repo, report.config, revision=report.revision).config.data_files
    return drop_sidecars(glob_with_sizes(list(data_files[report.split])))


def plan_cache(
    report: IntakeReport,
    files: list[FileEntry],
    need_tokens: int,
    tokenizer: str,
    cache_root: str,
    seed: int = SAMPLE_SEED,
) -> CachePlan:
    """Choose among ``files``, ``report``'s dataset, and the cache path under ``cache_root`` that names them."""
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
    name = f"{report.config}-{report.split}-{digest}"
    estimated = round(sum(f.size for f in chosen) * report.estimated_tokens / total_bytes)
    estimated = min(estimated, round(max_rows * report.sample.tokens_per_row))
    path = prefix_join(cache_root, f"{slug}/{report.revision[:12]}/{name}")
    return CachePlan(path, paths, max_rows, estimated, need_tokens)


def cache_epochs(need_tokens: int, cache_tokens: int, dataset_tokens: int) -> float:
    """Epochs of the built cache that the largest rung reads.

    Raises above ``MAX_EPOCHS``, and above 1 when the whole dataset holds the need: then the sampled files held fewer
    tokens than intake estimated.
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


@dataclasses.dataclass(frozen=True)
class CacheManifest:
    """What ``prepare`` wrote beside the cache: the intake it passed, the plan, and the built cache's size."""

    intake: IntakeReport
    plan: CachePlan
    stats: TokenizedCacheStats
    epochs_at_need: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


@dataclasses.dataclass(frozen=True)
class PreparedCache:
    """A cache that ``prepare`` built, as ``train`` reads it from the cache's manifest."""

    root: str
    """The cache path that ``prepare`` printed: it holds the manifest and the ``train`` split."""
    repo: str
    config: str
    share: float
    """The share that the cache was sized for and its epochs were checked at."""
    tokens: int
    """Tokens in the built cache."""

    @property
    def train_cache(self) -> str:
        return prefix_join(self.root, "train")

    @property
    def candidate(self) -> str:
        """The run-name label: the dataset, its config and share, and a hash of the path, unique to this cache."""
        name = self.repo.split("/")[-1]
        if self.config.startswith(name):
            name = self.config
        elif self.config != "default":
            name = f"{name}-{self.config}"
        tag = hashlib.sha256(self.root.encode()).hexdigest()[:6]
        return f"{name}-p{round(self.share * 100)}-{tag}".lower()

    @classmethod
    def read(cls, root: str) -> typing.Self:
        root = root.rstrip("/")
        manifest = StoragePath(prefix_join(root, MANIFEST_NAME))
        if not manifest.exists():
            raise ValueError(f"no {MANIFEST_NAME} in {root}: pass the cache path that prepare printed")
        written = json.loads(manifest.read_text())
        intake = written["intake"]
        return cls(root, intake["repo"], intake["config"], float(intake["share"]), int(written["stats"]["total_tokens"]))

    def check_rung(self, rung: str, rung_tokens: int) -> None:
        """Raise when a rung of ``rung_tokens`` training tokens reads this cache more than ``MAX_EPOCHS`` times."""
        epochs = self.share * rung_tokens / self.tokens
        if epochs > MAX_EPOCHS:
            raise ValueError(
                f"{rung} reads {self.candidate} {epochs:.2f} times, past {MAX_EPOCHS:g}: above its top rung"
            )


def build_cache(report: IntakeReport, plan: CachePlan, tokenizer: str, *, allow_test_in_train: bool) -> CacheManifest:
    """Tokenize the plan's files (skipped when the cache exists), check its epochs, then write the cache manifest.

    ``train`` reads the share and the run name from the manifest, so a cache without one cannot be trained on.
    ``allow_test_in_train`` turns tokenize's refusal of paths that contain ``test`` into a warning, for files known to
    be training data whose names contain the word (Datakit's ``methods2test`` sources).
    """
    config = TokenizeConfig(
        train_paths=list(plan.files),
        validation_paths=[],
        cache_path=plan.cache_path,
        tokenizer=tokenizer,
        format=TextLmDatasetFormat(text_key=report.text_field),
        sample_count=plan.max_rows,
        allow_test_in_train=allow_test_in_train,
    )
    tokenize(config)
    stats = read_tokenized_cache_stats(plan.cache_path, "train")
    manifest = CacheManifest(
        report, plan, stats, cache_epochs(plan.need_tokens, stats.total_tokens, report.estimated_tokens)
    )
    StoragePath(prefix_join(plan.cache_path, MANIFEST_NAME)).write_text(manifest.to_json())
    logger.info("cache %s: %s", plan.cache_path, stats)
    return manifest
