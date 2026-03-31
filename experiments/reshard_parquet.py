# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reshard large parquet files into smaller chunks for tokenization.

Reads parquet files and rewrites them with a target shard size, so that
no single file is too large for tokenization workers to handle.
"""

import dataclasses

from zephyr import Dataset, ZephyrContext, load_parquet


@dataclasses.dataclass(frozen=True)
class ReshardConfig:
    input_path: str
    output_path: str
    input_glob: str = "**/*.parquet"
    filter_null_text: bool = False


def reshard_parquet(config: ReshardConfig):
    """Read parquet files and rewrite as smaller JSONL shards."""
    from fray.cluster import ResourceConfig

    pattern = f"{config.input_path}/{config.input_glob}" if config.input_glob else str(config.input_path)
    ds = Dataset.from_files(pattern).flat_map(load_parquet)
    if config.filter_null_text:
        ds = ds.filter(lambda r: r.get("text") is not None and len(r.get("text", "")) > 0)
    pipeline = ds.write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    ctx = ZephyrContext(
        name="reshard-parquet",
        # 120g needed because load_parquet decompresses entire parquet files in memory;
        # SFT-Math has 8.7GB parquets that expand to ~50GB+ in memory during read.
        resources=ResourceConfig(cpu=2, ram="120g"),
    )
    ctx.execute(pipeline)
