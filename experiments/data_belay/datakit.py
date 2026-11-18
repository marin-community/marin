# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit registry sources as Data Belay candidates: the normalized Parquet output of one or more sources.

The sources are already in Marin's Datakit registry, so intake records the licenses on their Hugging Face cards and
does not refuse one off the allowlist. A candidate is sized for the largest rung that it can supply within
``MAX_EPOCHS``; sources too small for d768 alone are tested together with the rest of their family.
"""

import dataclasses
import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi
from levanter.tokenizers import MarinTokenizer
from marin.datakit.sources import all_sources
from marin.processing.tokenize._core import FileEntry, drop_sidecars, glob_with_sizes
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.data_belay.intake import (
    MAX_EPOCHS,
    SAMPLE_ROWS,
    IntakeReport,
    RungNeed,
    SampleStats,
    card_licenses,
    license_verdict,
    rung_needs,
    sample_stats,
)

# Registry name -> materialized artifact paths, relative to MARIN_PREFIX, of the hero data build.
HERO_DATA_PATHS = Path(__file__).resolve().parents[1] / "datakit" / "hero_data_paths.json"
TEXT_FIELD = "text"
# Hugging Face storage buckets, which some sources download from, are not datasets and have no card.
HF_BUCKET_PREFIX = "buckets/"
SAMPLE_SEED = 0
SAMPLE_BYTES = 32 * 1024 * 1024
SAMPLE_BATCH_ROWS = 8
TOKENIZE_BYTES = 1024 * 1024
# The lowest rung a candidate must supply: the ladder always runs d512 and d768.
MIN_TOP_RUNG = "d768"


def normalized_root(name: str, prefix: str) -> str:
    """The normalized output of registry source ``name`` under ``prefix``, as the hero data build wrote it."""
    rel = json.loads(HERO_DATA_PATHS.read_text()).get(f"normalized/{name}")
    if rel is None:
        raise ValueError(f"{name} has no normalized output in {HERO_DATA_PATHS.name}")
    return prefix_join(prefix, rel)


def source_files(name: str, prefix: str) -> list[FileEntry]:
    files = drop_sidecars(glob_with_sizes([prefix_join(normalized_root(name, prefix), "outputs/main/*.parquet")]))
    if not files:
        raise ValueError(f"{name} has no Parquet files under {normalized_root(name, prefix)}")
    return files


def source_repos(name: str) -> list[tuple[str, str | None]]:
    """The Hugging Face datasets, with revisions, that the source's step chain downloads."""
    repos, stack, seen = set(), list(all_sources()[name].normalize_steps), set()
    while stack:
        step = stack.pop()
        if step.name in seen:
            continue
        seen.add(step.name)
        if repo := step.hash_attrs.get("hf_dataset_id"):
            repos.add((repo, step.hash_attrs.get("revision")))
        stack.extend(step.deps)
    return sorted(repos)


def source_licenses(name: str) -> tuple[str, ...]:
    """The licenses on the dataset cards of the source's Hugging Face downloads; a storage bucket has no card."""
    licenses = []
    for repo, revision in source_repos(name):
        if repo.startswith(HF_BUCKET_PREFIX):
            continue
        licenses += card_licenses(HfApi().dataset_info(repo, revision=revision))
    return tuple(licenses)


def _texts(parquet: pq.ParquetFile) -> Iterator[str]:
    for batch in parquet.iter_batches(batch_size=SAMPLE_BATCH_ROWS, columns=[TEXT_FIELD]):
        yield from batch.column(TEXT_FIELD).to_pylist()


def file_sample(entry: FileEntry, tokenizer: MarinTokenizer) -> SampleStats:
    """Rows, text bytes and estimated tokens of a whole Parquet file.

    The rows and text bytes come from the file's metadata. The tokens per text byte come from the file's first
    ``SAMPLE_ROWS`` rows, up to ``SAMPLE_BYTES``, each tokenized up to ``TOKENIZE_BYTES``: rows can be whole
    repositories or books, so the first rows' lengths are a biased sample of the file's, and the caps bound memory.
    """
    rows_seen = cut_bytes = 0
    cut_texts: list[str] = []
    with StoragePath(entry.path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        meta = parquet.metadata
        # Metadata indexes leaf columns, which nested columns before the text field shift from its Arrow index.
        leaf = next(j for j in range(meta.num_columns) if meta.row_group(0).column(j).path_in_schema == TEXT_FIELD)
        file_text_bytes = sum(meta.row_group(i).column(leaf).total_uncompressed_size for i in range(meta.num_row_groups))
        for text in _texts(parquet):
            cut = text.encode("utf-8")[:TOKENIZE_BYTES].decode("utf-8", errors="ignore")
            rows_seen += 1
            cut_bytes += len(cut.encode("utf-8"))
            cut_texts.append(cut)
            if rows_seen >= SAMPLE_ROWS or cut_bytes >= SAMPLE_BYTES:
                break
    if not rows_seen:
        raise ValueError(f"{entry.path} has no rows")
    cut_stats = sample_stats(({TEXT_FIELD: t} for t in cut_texts), TEXT_FIELD, tokenizer)
    # BOS and EOS per document, as the training caches store them; the rest scales with the text bytes.
    tokens_per_byte = (cut_stats.tokens - 2 * rows_seen) / max(cut_bytes, 1)
    return SampleStats(meta.num_rows, file_text_bytes, round(file_text_bytes * tokens_per_byte) + 2 * meta.num_rows)


def top_rung(needs: Sequence[RungNeed]) -> str | None:
    """The largest rung such that it and every rung below it read the data at most ``MAX_EPOCHS`` times."""
    top = None
    for need in needs:
        if need.epochs > MAX_EPOCHS:
            break
        top = need.rung
    return top


@dataclasses.dataclass(frozen=True)
class DatakitIntake:
    report: IntakeReport
    files: tuple[FileEntry, ...]
    top_rung: str | None


def datakit_intake(
    label: str,
    sources: Sequence[str],
    share: float,
    rung_tokens: dict[str, int],
    tokenizer: MarinTokenizer,
    prefix: str,
) -> DatakitIntake:
    """Size the candidate ``label``, the union of ``sources``' normalized files, from one seeded file per source.

    Each source scales its sampled file's rows and tokens by its Parquet bytes over the file's, so a family of unlike
    sources is sized source by source. The report's sample is the sampled files.
    """
    files: list[FileEntry] = []
    licenses: list[str] = []
    rows = text_bytes = tokens = 0
    estimated_rows = estimated_tokens = 0.0
    for name in sources:
        entries = source_files(name, prefix)
        chosen = random.Random(SAMPLE_SEED).choice(sorted(entries, key=lambda f: f.path))
        stats = file_sample(chosen, tokenizer)
        scale = sum(f.size for f in entries) / chosen.size
        estimated_rows += stats.rows * scale
        estimated_tokens += stats.tokens * scale
        rows, text_bytes, tokens = rows + stats.rows, text_bytes + stats.text_bytes, tokens + stats.tokens
        files += entries
        licenses += source_licenses(name)
    roots = sorted(normalized_root(name, prefix) for name in sources)
    unique_licenses = tuple(sorted(set(licenses)))
    needs = rung_needs(round(estimated_tokens), share, rung_tokens)
    report = IntakeReport(
        repo=f"datakit/{label}",
        revision=hashlib.sha256(json.dumps(roots).encode()).hexdigest(),
        config="default",
        split="train",
        text_field=TEXT_FIELD,
        licenses=unique_licenses,
        license_from_card=True,
        license_verdict=license_verdict(unique_licenses),
        gated=False,
        num_rows=round(estimated_rows),
        sample=SampleStats(rows, text_bytes, tokens),
        estimated_tokens=round(estimated_tokens),
        share=share,
        needs=needs,
    )
    return DatakitIntake(report, tuple(files), top_rung(needs))
