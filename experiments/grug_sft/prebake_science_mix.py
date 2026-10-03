# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize each science curriculum as a sharded, training-ready Levanter cache."""

import argparse
import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import jax
import numpy as np
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.schedule import BatchSchedule
from levanter.store.cache import CACHE_LAYOUT_SHARDED, CacheLedger, CacheMetadata, write_levanter_cache
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.grug_sft.head_only_train import build_train_dataset
from experiments.grug_sft.science_curriculum import data_config
from experiments.grug_sft.science_mix import BATCH, CONTEXT, MIX_BUDGETS, STEPS, ScienceMix

logger = logging.getLogger(__name__)

SEQUENCES_PER_SHARD = BATCH
READ_BATCH_SIZE = 8
FIELDS = ("input_ids", "loss_weight", "segment_ids")


def _record(example) -> dict[str, np.ndarray]:
    if example.attn_mask.segment_ids is None:
        raise ValueError("Science SFT example has no document segment IDs")
    return {
        "input_ids": np.asarray(jax.device_get(example.tokens), dtype=np.int32),
        "loss_weight": np.asarray(jax.device_get(example.loss_weight), dtype=np.float32),
        "segment_ids": np.asarray(jax.device_get(example.attn_mask.segment_ids[0]), dtype=np.int32),
    }


def _records(dataset, start: int, stop: int):
    for batch_start in range(start, stop, READ_BATCH_SIZE):
        indices = range(batch_start, min(batch_start + READ_BATCH_SIZE, stop))
        for example in asyncio.run(dataset.get_batch(indices)):
            yield _record(example)


def _copy_shard(local_path: Path, remote_path: str) -> None:
    subprocess.run(["fsutil", "rsync", str(local_path), remote_path], check=True)
    StoragePath(prefix_join(remote_path, "shard.ready")).write_text("")


def _expected_metadata() -> CacheMetadata:
    fmt = PrebuiltLmDatasetFormat(loss_weights_key="loss_weight", segment_ids_key="segment_ids")
    processor = fmt.build_preprocessor(None)
    return CacheMetadata(preprocessor_metadata=processor.metadata)


def prebake(mix: ScienceMix, output_root: str) -> str:
    """Write one fixed sequence cache for the selected science curriculum."""
    configure_coreweave_s3()
    output_path = prefix_join(output_root, mix.value)
    data_key = jax.random.split(jax.random.PRNGKey(0), 2)[0]
    dataset = build_train_dataset(
        data_config(mix),
        max_seq_len=CONTEXT,
        batch_schedule=BatchSchedule(BATCH),
        key=data_key,
    )
    total_sequences = STEPS * BATCH
    shard_names = []
    shard_rows = {}
    field_counts_by_shard = {}
    for start in range(0, total_sequences, SEQUENCES_PER_SHARD):
        end = min(start + SEQUENCES_PER_SHARD, total_sequences)
        shard_name = f"shard-{start // SEQUENCES_PER_SHARD:05d}"
        shard_path = prefix_join(output_path, shard_name)
        ready = StoragePath(prefix_join(shard_path, "shard.ready"))
        if not ready.exists():
            with tempfile.TemporaryDirectory(prefix="science-sft-prebake-") as temporary:
                local_path = Path(temporary) / shard_name
                result = write_levanter_cache(
                    _records(dataset, start, end),
                    str(local_path),
                    metadata={"mix": mix.value, "sequence_start": start, "context": CONTEXT},
                    batch_size=READ_BATCH_SIZE,
                )
                if result["count"] != end - start:
                    raise RuntimeError(f"{shard_name}: wrote {result['count']} of {end - start} sequences")
                _copy_shard(local_path, shard_path)
            logger.info("Uploaded %s (%d/%d sequences)", shard_name, end, total_sequences)
        ledger = CacheLedger.load(shard_path)
        if not ledger.is_finished or ledger.total_num_rows != end - start:
            raise ValueError(f"Incomplete prebaked shard: {shard_path}")
        shard_names.append(shard_name)
        shard_rows[shard_name] = end - start
        field_counts_by_shard[shard_name] = {field: (end - start) * CONTEXT for field in FIELDS}

    root_ledger = CacheLedger(
        total_num_rows=total_sequences,
        shard_rows=shard_rows,
        is_finished=True,
        finished_shards=shard_names,
        field_counts={field: total_sequences * CONTEXT for field in FIELDS},
        field_counts_by_shard=field_counts_by_shard,
        layout=CACHE_LAYOUT_SHARDED,
        metadata=_expected_metadata(),
    )
    StoragePath(prefix_join(output_path, "shard_ledger.json")).write_text(root_ledger.to_json())
    manifest = {
        "mix": mix.value,
        "source_pool": marin_prefix(),
        "cache_path": output_path,
        "sequence_count": total_sequences,
        "context": CONTEXT,
        "shard_count": len(shard_names),
        "budget_billion_tokens": MIX_BUDGETS[mix].as_dict(),
    }
    StoragePath(prefix_join(output_path, "manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", type=ScienceMix, choices=ScienceMix, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(prebake(args.mix, args.output_root))
