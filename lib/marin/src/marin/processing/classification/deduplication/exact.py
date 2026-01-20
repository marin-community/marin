# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator
from functools import partial
from marin.utils import rebase_file_path
import pyarrow as pa
import logging
from fray.job.context import create_job_ctx
from marin.processing.classification.deduplication import dedup_commons
import wandb
from zephyr.backends import Backend
from zephyr.dataset import Dataset

logger = logging.getLogger(__name__)


def dedup_exact_paragraph(config: dedup_commons.DedupConfig):
    import dupekit
    from dupekit import Transformation

    input_files = dedup_commons.collect_input_files(input_paths=config.input_paths, filetypes=config.filetypes)

    # TODO(rav): measure and tune the memory limits
    ctx = create_job_ctx("auto", memory=config.ray_memory, num_cpus=config.ray_num_cpus)
    dedup_commons.init_wandb(config)

    def compute_paragraph_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="resolved_id"),
            Transformation.SplitParagraphs(text_col=config.text_field, id_col="resolved_id"),
            Transformation.Hash(input_col="paragraph_text", output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
            Transformation.SelectColumns(columns=["hash", "doc_id"]),
        ]
        return dupekit.transform(batch, pipeline)

    # first compute the full set of duplicate keys.
    duplicate_key_shards = list(
        Backend.execute(
            Dataset.from_list(input_files).flat_map(dedup_commons.load_batches)
            # NOTE: when do we want to trigger reshard. Keep in mind that reshard will materialize the
            #   text field!
            # TODO: the resharding logic should be improved, based on size and/or max_parallelism
            .reshard(num_shards=config.processes if len(input_files) > 3 and len(input_files) < 42 else None)
            .map(compute_paragraph_hashes)
            .flat_map(lambda batch: batch.to_pylist())
            .group_by(
                lambda key_fn: key_fn["hash"],
                partial(dedup_commons.count_reduce, canonical_id="doc_id"),
                num_output_shards=42,
            )
            .write_parquet(f"{config.output_path}/metadata/dup-key-{{shard:05d}}-of-{{total:05d}}.parquet"),
            context=ctx,
            max_parallelism=config.processes,
            verbose=True,
        ),
    )

    exact_cnts = dedup_commons.compute_dedup_stats(duplicate_key_shards, method="exact", level="paragraph")
    logger.info(str(exact_cnts))

    if wandb.run:
        wandb.log(exact_cnts.to_dict())

    def mark_exact_dups_paragraphs(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Mark duplicate paragraphs in a single record using exact hash matching."""

        dup_map = dedup_commons.load_dupe_map_shard(duplicate_key_shards)

        for batch in batches:
            yield dupekit.mark_paragraph_duplicates(
                batch,
                dup_map,
                attribute_name=str(dedup_commons.DedupMode.EXACT_PARAGRAPH),
                algorithm=dupekit.HashAlgorithm.Xxh3_128,
            )

    base_path = dedup_commons.find_base_path(config.input_paths, input_files)
    Backend.execute(
        Dataset.from_list(input_files)
        .flat_map(dedup_commons.load_batches)
        .map_shard(mark_exact_dups_paragraphs)
        .flat_map(lambda batch: batch.to_pylist())
        .write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                input_files[shard_idx],
                f"{config.output_path}/data",
                old_extension=dedup_commons.get_extension(input_files[shard_idx]),
                new_extension=".jsonl.gz",
            ),
            skip_existing=True,
        ),
        context=ctx,
        verbose=True,
    )

    if wandb.run:
        wandb.finish()

    return {"success": True, "mode": str(dedup_commons.DedupMode.EXACT_PARAGRAPH)} | exact_cnts.to_dict()


def dedup_exact_document(config: dedup_commons.DedupConfig):
    """Exact document deduplication: identify duplicate documents based on full text hash"""
    import dupekit
    from dupekit import Transformation

    input_files = dedup_commons.collect_input_files(input_paths=config.input_paths, filetypes=config.filetypes)

    ctx = create_job_ctx("auto", memory=config.ray_memory, num_cpus=config.ray_num_cpus)
    dedup_commons.init_wandb(config)

    def compute_document_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="resolved_id"),
            Transformation.Hash(input_col=config.text_field, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
            Transformation.SelectColumns(columns=["hash", "resolved_id"]),
        ]
        return dupekit.transform(batch, pipeline)

    # first compute the full set of duplicate keys.
    duplicate_key_shards = list(
        Backend.execute(
            Dataset.from_list(input_files).flat_map(dedup_commons.load_batches)
            # NOTE: when do we want to trigger reshard. Keep in mind that reshard will materialize the
            #   text field!
            # TODO: the resharding logic should be improved, based on size and/or max_parallelism
            .reshard(num_shards=config.processes if len(input_files) > 3 and len(input_files) < 42 else None)
            .map(compute_document_hashes)
            .flat_map(lambda batch: batch.to_pylist())
            .group_by(
                lambda key_fn: key_fn["hash"],
                partial(dedup_commons.count_reduce, canonical_id="resolved_id"),
                num_output_shards=42,
            )
            .write_parquet(f"{config.output_path}/metadata/dup-key-{{shard:05d}}-of-{{total:05d}}.parquet"),
            context=ctx,
            max_parallelism=config.processes,
            verbose=True,
        )
    )

    exact_cnts = dedup_commons.compute_dedup_stats(duplicate_key_shards, method="exact", level="document")
    logger.info(str(exact_cnts))

    if wandb.run:
        wandb.log(exact_cnts.to_dict())

    def mark_dup_documents(batches: Iterator[pa.RecordBatch]) -> Iterator[dict]:
        """Mark exact duplicate documents using exact hash matching."""
        dup_map = dedup_commons.load_dupe_map_shard(duplicate_key_shards)

        for batch in batches:
            prepared_batch = dupekit.transform(
                batch,
                [
                    Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="id"),
                    Transformation.Hash(
                        input_col=config.text_field, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128
                    ),
                ],
            )
            b = dupekit.mark_document_duplicates(
                prepared_batch,
                dup_map,
                attribute_name=str(dedup_commons.DedupMode.EXACT_DOCUMENT),
                hash_col="hash",
            )
            yield from b.to_pylist()

    base_path = dedup_commons.find_base_path(config.input_paths, input_files)
    Backend.execute(
        Dataset.from_list(input_files).flat_map(dedup_commons.load_batches)
        # NOTE/TODO: we can't reshard here to increase parallelism because afaiu we want to match
        # the shards of the input files for rebase_file_path to work correctly.
        .map_shard(mark_dup_documents).write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                input_files[shard_idx],
                f"{config.output_path}/data",
                old_extension=dedup_commons.get_extension(input_files[shard_idx]),
                new_extension=".jsonl.gz",
            ),
            skip_existing=True,
        ),
        context=ctx,
        verbose=True,
    )

    if wandb.run:
        wandb.finish()

    return {"success": True, "mode": str(dedup_commons.DedupMode.EXACT_DOCUMENT)} | exact_cnts.to_dict()
