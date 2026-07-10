# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: decon-filter two sources' tokenize attributes + build a Levanter store.

End-to-end exercise of the decon → tokenize → filter → store path inside a
single iris job:

1. For each (decon, tokenize) source pair, load the contaminated ``id``s
   from the decon attribute parquets into one in-memory set.
2. Concatenate all tokenize attribute parquets across sources into one
   zephyr ``Dataset``, filter records whose ``id`` lives in the contam set.
3. Run :func:`build_from_datasets` to write a single sharded Levanter cache
   covering both sources.

All upstream artifacts (decon, tokenize) are produced by the all-sources
runs and reused via fixed paths -- no recompute.

Submit on iris (eu-west4):

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive \\
        -- python experiments/tokenize/smoke_test_decon_filter_store.py
"""

import logging
import os

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.processing.tokenize.attributes import TokenizedAttrData
from marin.processing.tokenize.store_builder import build_from_datasets, write_stats_json
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file

logger = logging.getLogger(__name__)


# Two sources with non-zero contamination, small enough for a smoke test.
# (label, decon_output_dir, tokenize_output_dir)
SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "cp/project_gutenberg",  # 55K records, 164 contam (0.30%)
        "gs://marin-eu-west4/tmp/ttl=7d/rav/decon-all-sources-v0/datakit/decon/cp/project_gutenberg_93d48265",
        "gs://marin-eu-west4/datakit/tokenize/cp/project_gutenberg_008d9453",
    ),
    (
        "coderforge",  # 258K records, 17 contam
        "gs://marin-eu-west4/tmp/ttl=7d/rav/decon-all-sources-v0/datakit/decon/coderforge_5dfa72f4",
        "gs://marin-eu-west4/datakit/tokenize/coderforge_c9fd4bbe",
    ),
)

STORE_PATH = "gs://marin-eu-west4/datakit/store/smoke-decon-filter"
SPLIT = "train"
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="10g")


def _list_parquets(directory: str) -> list[str]:
    return sorted(str(p) for p in StoragePath(directory).ls() if str(p).endswith(".parquet"))


def _load_contam_ids(decon_dirs: list[str]) -> set[str]:
    """Union of `id`s flagged as contaminated across all sources' decon parquets.

    Decon emits the datakit ``{id, partition_id, attributes: {contaminated, ...}}``
    shape; flatten the struct via pyarrow ``StructArray.field`` so we don't
    materialize the matched_hashes column.
    """
    ids: set[str] = set()
    for d in decon_dirs:
        for path in _list_parquets(d):
            table = pq.read_table(path, columns=["id", "attributes"])
            contaminated = table.column("attributes").combine_chunks().field("contaminated").to_pylist()
            ids_col = table.column("id").to_pylist()
            for i, c in zip(ids_col, contaminated, strict=True):
                if c:
                    ids.add(i)
    return ids


def _all_tokenize_shards(split: str) -> list[str]:
    shards: list[str] = []
    for _, _, tok_dir in SOURCES:
        tok = read_artifact(tok_dir, TokenizedAttrData)
        split_dir = tok.output_dirs.get(split)
        if split_dir is None:
            raise FileNotFoundError(f"{tok_dir}: no '{split}' split")
        shards.extend(_list_parquets(split_dir))
    return shards


def _exemplar(shards: list[str]) -> dict:
    """First non-empty record from the first shard, with 'id' stripped.

    Uses ``pq.ParquetFile.iter_batches`` so we don't load full-table tokens
    into the launcher — a single book-length doc's ``input_ids`` list can
    blow a 1 GB launcher container.
    """
    for path in shards:
        with StoragePath(path).open("rb") as fh:
            pf = pq.ParquetFile(fh)
            if pf.metadata.num_rows == 0:
                continue
            first_batch = next(pf.iter_batches(batch_size=1))
        record = first_batch.to_pylist()[0]
        record.pop("id", None)
        return record
    raise FileNotFoundError("All input shards empty")


def main() -> None:
    configure_logging(logging.INFO)

    decon_dirs = [decon for _, decon, _ in SOURCES]
    logger.info("loading contam ids from %d decon dirs", len(decon_dirs))
    contam_ids = _load_contam_ids(decon_dirs)
    logger.info("union contam ids: %d", len(contam_ids))

    shards = _all_tokenize_shards(SPLIT)
    logger.info("tokenize shards across sources: %d", len(shards))

    exemplar = _exemplar(shards)
    logger.info("exemplar keys: %s", sorted(exemplar.keys()))

    ctx = ZephyrContext(
        resources=WORKER_RESOURCES,
        max_workers=min(1024, len(shards)),
        name="smoke-decon-filter-store",
    )

    dataset = Dataset.from_list(shards).flat_map(load_file).filter(lambda r, c=contam_ids: r["id"] not in c)

    split_output = os.path.join(STORE_PATH, SPLIT)
    ledger = build_from_datasets(
        ctx=ctx,
        dataset=dataset,
        output_path=split_output,
        exemplar=exemplar,
    )
    stats_path, stats = write_stats_json(split_output, ledger)
    logger.info(
        "smoke done: docs=%d tokens=%d shards=%d → %s (stats: %s)",
        stats["total_elements"],
        stats["total_tokens"],
        len(shards),
        split_output,
        stats_path,
    )


if __name__ == "__main__":
    main()
