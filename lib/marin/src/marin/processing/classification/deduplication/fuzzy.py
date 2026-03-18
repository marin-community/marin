# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Sequence
from functools import partial
import dupekit
import logging
from marin.utils import rebase_file_path
import pyarrow as pa
from fray.v2.local_backend import LocalClient
from marin.processing.classification.deduplication.dedup_commons import (
    DEFAULT_FILETYPES,
    DedupMode,
    DupCounters,
    _collect_input_files,
    _find_base_path,
    _get_extension,
    _init_wandb,
    _load_batches,
)
from fray.v2 import ResourceConfig
from marin.processing.classification.deduplication.connected_components import connected_components
from marin.utilities.time_logger import log_time
import wandb
from zephyr import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import InputFileSpec, load_file

logger = logging.getLogger(__name__)


def _compute_fuzzy_dedup_stats(shards: list[str] | Sequence[str], method: str, level: str) -> DupCounters:
    with log_time(f"Compute fuzzy deduplication stats from {len(shards)} shards"):
        ctx = ZephyrContext(client=LocalClient(), name="fuzzy-dup-counts")
        result: DupCounters = ctx.execute(  # type: ignore[bad-assignment]
            Dataset.from_list(shards)
            .load_parquet(columns=["component_id"])
            # Compute the per-component statistics and then roll them up into a single counter group
            .group_by(
                key=lambda r: r["component_id"],
                reducer=lambda _, items: DupCounters(
                    method=method,
                    level=level,
                    total=(total := sum(1 for _ in items)),
                    dups=total if total > 1 else 0,
                    unique=1,
                ),
            )
            .reduce(partial(sum, start=DupCounters(method=method, level=level))),
        )[0]
    return result


def _load_fuzzy_dupe_map_shard(shards: list[str]) -> dict[str, bool]:
    if not shards:
        logger.warning("No fuzzy duplicate documents found.")
        return {}

    # Map record ID -> is duplicate (bool)
    shard_dup_map = {}

    def add_to_dup_map(record: dict):
        shard_dup_map[record["id"]] = record["fuzzy_duplicate"]

    with log_time(f"Load fuzzy duplicate map from {len(shards)} shards"):
        ctx = ZephyrContext(client=LocalClient(), name="fuzzy-dup-map")
        ctx.execute(
            Dataset.from_list(shards).load_parquet().map(add_to_dup_map),
        )

    return shard_dup_map


def dedup_fuzzy_document(
    *,
    input_paths: str | list[str],
    output_path: str,
    text_field: str = "text",
    filetypes: list[str] | None = None,
    fuzzy_minhash_num_perms: int = 286,
    fuzzy_minhash_num_bands: int = 26,
    fuzzy_minhash_ngram_size: int = 5,
    fuzzy_minhash_seed: int = 42,
    max_parallelism: int | None = None,
    worker_resources: ResourceConfig | None = None,
) -> dict:
    """Perform fuzzy document-level deduplication"""

    if fuzzy_minhash_num_perms % fuzzy_minhash_num_bands != 0:
        raise ValueError(
            f"minhash_num_perms ({fuzzy_minhash_num_perms}) must be divisible by "
            f"minhash_num_bands ({fuzzy_minhash_num_bands})"
        )

    if filetypes is None:
        filetypes = DEFAULT_FILETYPES

    input_files = _collect_input_files(input_paths=input_paths, filetypes=filetypes)

    _init_wandb(mode=DedupMode.FUZZY_DOCUMENT, input_paths=input_paths)

    def compute_minhash_lsh_batches(batch: pa.RecordBatch) -> Iterator[dict]:
        """
        Runs the Rust-optimized MinHash LSH pipeline on a RecordBatch.
        Yields {bucket: str, id: Any} for each bucket hit.
        """
        pipeline = [
            dupekit.Transformation.CleanText(input_col=text_field, output_col="clean_text"),
            dupekit.Transformation.MinHash(
                input_col="clean_text",
                output_col="signature",
                num_perms=fuzzy_minhash_num_perms,
                ngram_size=fuzzy_minhash_ngram_size,
                seed=fuzzy_minhash_seed,
            ),
            dupekit.Transformation.MinHashLSH(
                input_col="signature", output_col="buckets", num_bands=fuzzy_minhash_num_bands
            ),
            dupekit.Transformation.SelectColumns(columns=["id", "buckets"]),
        ]

        result_batch = dupekit.transform(batch, pipeline)

        ids = result_batch["id"]
        buckets = result_batch["buckets"]

        for doc_id, doc_buckets in zip(ids, buckets, strict=True):
            if not doc_buckets.is_valid:
                continue

            doc_id_val = doc_id.as_py()
            for b in doc_buckets.as_py():
                yield {"bucket": str(b), "id": doc_id_val}

    ctx = ZephyrContext(
        name="fuzzy-dedup",
        max_workers=max_parallelism,
        resources=worker_resources or ResourceConfig(cpu=1, ram="32g", disk="5g"),
    )
    doc_minhash_lsh = (
        Dataset.from_list(input_files)
        .flat_map(lambda f: _load_batches(f, columns=[text_field, "id"]))
        .flat_map(compute_minhash_lsh_batches)
    )
    converged, cc_files = connected_components(doc_minhash_lsh, ctx, output_dir=f"{output_path}/metadata/cc")
    if not converged:
        # TODO (rav): log the number of changed nodes?
        logger.warning("Connected components did not converge")
    fuzzy_dup_shards = ctx.execute(
        Dataset.from_list(cc_files)
        .flat_map(load_file)
        .map(
            lambda r: {
                "id": r["node_id"]["record_id"],
                "fuzzy_duplicate": r["component_id"] != r["node_id"]["record_id_norm"],
            }
        )
        .reshard(num_shards=42)
        .write_parquet(f"{output_path}/metadata/fuzzy-dup-key-{{shard:05d}}-of-{{total:05d}}.parquet"),
        verbose=True,
    )

    fuzzy_cnt = _compute_fuzzy_dedup_stats(cc_files, method="fuzzy", level="document")
    logger.info(str(fuzzy_cnt))

    if wandb.run:
        wandb.log(fuzzy_cnt.to_dict())

    def mark_dup_documents(docs: Iterator[dict], _) -> Iterator[dict]:
        fuzzy_dup_map = _load_fuzzy_dupe_map_shard(fuzzy_dup_shards)

        for doc in docs:
            is_fuzzy_dup = fuzzy_dup_map.get(doc["id"], False)
            doc["attributes"] = doc.get("attributes", {})
            assert DedupMode.FUZZY_DOCUMENT not in doc["attributes"]
            doc["attributes"][str(DedupMode.FUZZY_DOCUMENT)] = is_fuzzy_dup
            yield doc

    base_path = _find_base_path(input_paths, input_files)
    ctx.execute(
        Dataset.from_list(input_files).flat_map(lambda p: load_file(InputFileSpec(path=p, columns=["id"])))
        # NOTE/TODO: we can't reshard here to increase parallelism because afaiu we want to match
        # the shards of the input files for rebase_file_path to work correctly.
        .map_shard(mark_dup_documents).write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                input_files[shard_idx],
                f"{output_path}/data",
                old_extension=_get_extension(input_files[shard_idx]),
                new_extension=".jsonl.gz",
            ),
            skip_existing=True,
        ),
        verbose=True,
    )

    if wandb.run:
        wandb.finish()

    return {"success": True, "mode": str(DedupMode.FUZZY_DOCUMENT)} | fuzzy_cnt.to_dict()
