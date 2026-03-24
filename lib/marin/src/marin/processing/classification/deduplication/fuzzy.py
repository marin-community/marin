# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
import dupekit
import logging
from marin.utils import rebase_file_path
import pyarrow as pa
from marin.processing.classification.deduplication.dedup_commons import (
    DEFAULT_FILETYPES,
    DedupMode,
    DupCounters,
    _collect_input_files,
    _find_base_path,
    _get_extension,
    _init_wandb,
    _load_batches,
    group_files,
)
from fray.v2 import ResourceConfig
from marin.processing.classification.deduplication.connected_components import connected_components
import wandb
from zephyr import ZephyrContext, write_vortex_file
from zephyr.dataset import Dataset

logger = logging.getLogger(__name__)


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
    coordinator_resources: ResourceConfig | None = None,
) -> dict:
    """Perform fuzzy document-level deduplication.

    Args:
        worker_resources: Resource config per Zephyr worker. The map stage runs
            dupekit's Rust MinHash pipeline, which uses a native thread pool and
            may consume up to ~2 cores beyond the Python thread. Size cpu
            accordingly (e.g. cpu=5 reserves headroom for this).
    """

    if fuzzy_minhash_num_perms % fuzzy_minhash_num_bands != 0:
        raise ValueError(
            f"minhash_num_perms ({fuzzy_minhash_num_perms}) must be divisible by "
            f"minhash_num_bands ({fuzzy_minhash_num_bands})"
        )

    if filetypes is None:
        filetypes = DEFAULT_FILETYPES

    input_files = _collect_input_files(input_paths=input_paths, filetypes=filetypes)
    idx_to_path = dict(list(enumerate(sorted(input_files))))
    path_to_idx = {v: k for k, v in idx_to_path.items()}

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
        coordinator_resources=coordinator_resources or ResourceConfig(cpu=1, ram="5g"),
    )
    # Group input files into at most max_parallelism shards so shard count <= worker count.
    # Each shard processes a group of files sequentially, reducing coordinator overhead
    # when num_files >> max_parallelism (e.g. 13k files with 2k workers).
    file_groups: list[list[str]] = (
        group_files(input_files, max_parallelism) if max_parallelism else [[f] for f in sorted(input_files)]
    )
    doc_minhash_lsh = Dataset.from_list(file_groups).flat_map(
        lambda paths: (
            {**record, "file_idx": path_to_idx[path]}
            for path in paths
            for batch in _load_batches(path, columns=[text_field, "id"])
            for record in compute_minhash_lsh_batches(batch)
        )
    )
    converged, cc_files = connected_components(doc_minhash_lsh, ctx, output_dir=f"{output_path}/metadata/cc")
    if not converged:
        # TODO (rav): log the number of changed nodes?
        logger.warning("Connected components did not converge")

    def aggregate_and_write_to_corresponding_files(file_idx: int, records: Iterator[dict]) -> dict:
        input_path = idx_to_path[file_idx]
        output_file = rebase_file_path(
            _find_base_path(input_paths, [input_path]),
            input_path,
            f"{output_path}/data/",
            old_extension=_get_extension(input_path),
            new_extension=".vortex",
        )

        # NOTE: this is per file stat, we aggregate across files later
        stats = DupCounters(method="fuzzy", level="document")

        def counting_iter():
            nonlocal stats
            for record in records:
                is_dup: bool = record["is_dup"]
                stats += DupCounters(
                    method="fuzzy",
                    level="document",
                    total=1,
                    dups=int(is_dup),
                    unique=int(not is_dup),
                )
                yield record

        def skip_non_dups(records: Iterator[dict]) -> Iterator[dict]:
            for record in records:
                if record["is_dup"]:
                    yield {"id": record["id"], "attributes": {"dup_doc": True}}

        result = write_vortex_file(skip_non_dups(counting_iter()), output_file)
        return {**result, "stats": stats}

    shard_results = list(
        ctx.execute(
            Dataset.from_list(cc_files)
            .load_vortex()
            .map(
                lambda r: {
                    "id": r["record_id"],
                    "is_dup": r["component_id"] != r["id_norm"],
                    "file_idx": r["file_idx"],
                }
            )
            .group_by(
                lambda r: r["file_idx"],
                sort_by=lambda r: r["id"],
                reducer=aggregate_and_write_to_corresponding_files,
            ),
            verbose=True,
        ),
    )

    fuzzy_cnts = sum((r["stats"] for r in shard_results), start=DupCounters(method="fuzzy", level="document"))
    logger.info(str(fuzzy_cnts))

    if wandb.run:
        wandb.log(fuzzy_cnts.to_dict())
        wandb.finish()

    return {"success": True, "mode": str(DedupMode.FUZZY_DOCUMENT)} | fuzzy_cnts.to_dict()
