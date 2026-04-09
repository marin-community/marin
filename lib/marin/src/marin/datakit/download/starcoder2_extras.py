# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download subsets of the bigcode/starcoder2data-extras dataset from HuggingFace.

Subsets: ir_cpp, ir_python, ir_rust, ir_low_resource, documentation, kaggle.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "bigcode/starcoder2data-extras"
HF_REVISION = "1ba0d4f"

SUBSETS = ["ir_cpp", "ir_python", "ir_rust", "ir_low_resource", "documentation", "kaggle"]


def download_starcoder2_extras_step(subset: str) -> StepSpec:
    """Download a single subset of the starcoder2data-extras dataset."""
    return download_hf_step(
        f"raw/starcoder2_extras/{subset}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{subset}/*.parquet"],
        override_output_path=f"raw/starcoder2_extras-{HF_REVISION}/{subset}",
    )


def reshard_starcoder2_extras_step(subset: str, target_shard_mb: int = 200) -> StepSpec:
    """Reshard a downloaded subset into more evenly-sized parquet files."""
    raw = download_starcoder2_extras_step(subset)
    raw_output_path = raw.output_path

    def _run(output_path: str) -> None:
        import logging

        import pyarrow.parquet as pq
        from rigging.filesystem import url_to_fs

        logger = logging.getLogger(__name__)
        input_path = raw_output_path
        fs, _ = url_to_fs(input_path)
        files = sorted(f"gs://{f}" for f in fs.glob(f"{input_path}/**/*.parquet") if not f.endswith("/.parquet"))

        # Read all files, split into evenly-sized output shards
        target_bytes = target_shard_mb * 1024 * 1024
        shard_idx = 0
        for file_path in files:
            meta = pq.read_metadata(file_path)
            if meta.serialized_size <= target_bytes:
                # Small file — copy as-is
                out = f"{output_path}/shard-{shard_idx:05d}.parquet"
                table = pq.read_table(file_path)
                pq.write_table(table, out)
                logger.info(f"Copied {file_path} -> {out} ({table.num_rows} rows)")
                shard_idx += 1
            else:
                # Big file — split by row groups or by row count
                table = pq.read_table(file_path)
                rows_per_shard = max(1, (table.num_rows * target_bytes) // meta.serialized_size)
                offset = 0
                while offset < table.num_rows:
                    chunk = table.slice(offset, min(rows_per_shard, table.num_rows - offset))
                    out = f"{output_path}/shard-{shard_idx:05d}.parquet"
                    pq.write_table(chunk, out)
                    logger.info(
                        f"Split {file_path}[{offset}:{offset + chunk.num_rows}] -> {out} ({chunk.num_rows} rows)"
                    )
                    shard_idx += 1
                    offset += chunk.num_rows
                del table

        logger.info(f"Resharded {len(files)} files into {shard_idx} shards")

    return StepSpec(
        name=f"resharded/starcoder2_extras/{subset}",
        fn=_run,
        deps=[raw],
    )


def download_all_starcoder2_extras_steps() -> list[StepSpec]:
    """Download all selected subsets of starcoder2data-extras."""
    return [download_starcoder2_extras_step(subset) for subset in SUBSETS]
