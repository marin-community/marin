# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw a few detokenized documents from every source of the 10.05 Grug SFT mixture.

The 10.05 SFT (run ``grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped``) mixed 216 SFT token
stores with long-context pretraining replay. ``mix/`` pins that run's allocation and replay definition.
This script reads a fixed number of whole documents from each SFT store and each replay cache, decodes
them with the run's tokenizer and writes ``sources.json`` and ``samples.parquet`` below ``--output``.

Run it in us-central2, next to the caches. ``load.py`` stages the result into the viewer.
"""

import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.store.cache import TreeCache
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

MIX_DIR = Path(__file__).with_name("mix")
MIX_PATH = MIX_DIR / "special_token_full_ab_grouped_mix.json"
REPLAY_PATH = MIX_DIR / "replay_skew8.json"
STORES_MANIFEST = "gs://marin-us-central2/grug_sft/stores/full-ab-packed-2026.09.29.json"
TOKENIZER = "gs://marin-us-central2/grug_sft/tokenizer/2026.09.18"
EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}
REPLAY_PATH_PATTERN = re.compile(r"/cluster=(\d+)/quality=(\d+)/length=(\w+)$")
READ_WORKERS = 32
# A document longer than HEAD_CHARS + TAIL_CHARS keeps only its two ends.
HEAD_CHARS = 48_000
TAIL_CHARS = 16_000
NUL_PLACEHOLDER = "\u2400"
SOURCES_FILE = "sources.json"
SAMPLES_FILE = "samples.parquet"


@dataclass(frozen=True)
class MixSource:
    """One token cache of the mixture and the share of training positions it was scheduled for."""

    name: str
    kind: str  # "sft" or "replay"
    family: str
    share: float
    cache_path: str


@dataclass(frozen=True)
class Sample:
    source: str
    doc_index: int
    tokens: int
    chars: int
    text: str


def sft_sources(mix: dict, stores: dict[str, dict]) -> list[MixSource]:
    """Each SFT store's share is its packed-position allocation within the SFT fraction."""
    allocations = mix["allocations_tokens"]
    budget = sum(allocations.values())
    return [
        MixSource(
            name=name,
            kind="sft",
            family=name.split("/")[0],
            share=mix["sft_fraction"] * tokens / budget,
            cache_path=f"{stores[name]['cache_path']}/train",
        )
        for name, tokens in sorted(allocations.items())
    ]


def replay_sources(mix: dict, replay: dict[str, dict]) -> list[MixSource]:
    """A replay component splits its weight across its caches by copy count (long documents are oversampled)."""
    sources = []
    for _, record in sorted(replay.items()):
        copies = record["copies"]
        total_copies = sum(copies.values())
        for path, count in sorted(copies.items()):
            match = REPLAY_PATH_PATTERN.search(path)
            if match is None:
                raise ValueError(f"Unrecognized replay cache path: {path}")
            cluster, quality, length = match.groups()
            sources.append(
                MixSource(
                    name=f"replay/c{int(cluster):02d}/q{quality}/{length}",
                    kind="replay",
                    family=f"replay/c{int(cluster):02d}",
                    share=(1.0 - mix["sft_fraction"]) * record["weight"] * count / total_copies,
                    cache_path=path,
                )
            )
    return sources


def clip_text(text: str) -> str:
    """Keep the head and tail of a long document and say how much was dropped between them."""
    text = text.replace("\x00", NUL_PLACEHOLDER)  # Postgres text cannot hold NUL
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    dropped = len(text) - HEAD_CHARS - TAIL_CHARS
    return f"{text[:HEAD_CHARS]}\n\n[… {dropped:,} characters omitted …]\n\n{text[-TAIL_CHARS:]}"


def draw_samples(source: MixSource, per_source: int, rng: np.random.Generator, tokenizer) -> tuple[int, list[Sample]]:
    """Read ``per_source`` distinct documents; returns the cache's document count with them."""
    cache = TreeCache.load(source.cache_path, EXEMPLAR)
    documents = len(cache)
    indices = sorted(int(i) for i in rng.choice(documents, size=min(per_source, documents), replace=False))
    samples = []
    for index, row in zip(indices, cache.get_batch_sync(indices), strict=True):
        ids = np.asarray(row["input_ids"])
        text = tokenizer.decode(ids, skip_special_tokens=False)
        samples.append(Sample(source.name, index, len(ids), len(text), clip_text(text)))
    logger.info("%s: %d of %d documents", source.name, len(samples), documents)
    return documents, samples


def sample_mix(output: str, sft_per_source: int, replay_per_source: int, seed: int) -> None:
    mix = json.loads(MIX_PATH.read_text())
    stores = json.loads(StoragePath.parse(STORES_MANIFEST).read_text())
    sources = sft_sources(mix, stores) + replay_sources(mix, json.loads(REPLAY_PATH.read_text()))
    tokenizer = load_tokenizer(TOKENIZER)
    per_source = {"sft": sft_per_source, "replay": replay_per_source}

    # Each source has its own seeded generator, so its draw does not depend on read order.
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        draws = list(
            pool.map(
                lambda item: draw_samples(
                    item[1], per_source[item[1].kind], np.random.default_rng([seed, item[0]]), tokenizer
                ),
                enumerate(sources),
            )
        )

    source_rows = [
        {**asdict(source), "documents": documents} for source, (documents, _) in zip(sources, draws, strict=True)
    ]
    sample_rows = [asdict(sample) for _, samples in draws for sample in samples]
    root = StoragePath.parse(output)
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(sample_rows), buffer, compression="zstd")
    (root / SAMPLES_FILE).write_bytes(buffer.getvalue())
    (root / SOURCES_FILE).write_text(json.dumps({"seed": seed, "sources": source_rows}, indent=1))
    logger.info(
        "Wrote %d samples from %d sources (%d parquet bytes) to %s",
        len(sample_rows),
        len(source_rows),
        buffer.tell(),
        output,
    )


@click.command(help=__doc__)
@click.option("--output", required=True, help="Directory or object-storage prefix for the snapshot.")
@click.option("--sft-per-source", type=int, default=8, show_default=True)
@click.option("--replay-per-source", type=int, default=4, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(output: str, sft_per_source: int, replay_per_source: int, seed: int) -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    sample_mix(output, sft_per_source, replay_per_source, seed)


if __name__ == "__main__":
    main()
