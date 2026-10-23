# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load bounded batches through candidate 11's exact native training-data pipeline."""

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import draccus
import jax
import numpy as np
from levanter.data.text.datasets import LmDataConfig
from levanter.schedule import BatchSchedule
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.train import build_train_dataset
from experiments.grug.noise_belay.candidate11 import REFERENCE


def native_train_dataset(tokenizer_path: Path):
    """Use the recorded data configuration and native seed splitting/packing."""
    record = json.loads(REFERENCE.read_text())["config"]
    trainer = record["trainer"]["trainer"]
    data_key, _ = jax.random.split(jax.random.PRNGKey(trainer["seed"]), 2)
    if record["trainer"]["data_seed"] is not None:
        data_key = jax.random.PRNGKey(record["trainer"]["data_seed"])
    config = dataclasses.replace(
        draccus.decode(LmDataConfig, record["data"]),
        tokenizer=str(tokenizer_path.resolve()),
    )
    schedule = BatchSchedule(trainer["train_batch_size"])
    dataset = build_train_dataset(
        config,
        max_seq_len=record["model"]["max_seq_len"],
        batch_schedule=schedule,
        key=data_key,
    )
    return dataset, schedule, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.start_step < 0 or args.steps < 1:
        raise ValueError("Require a nonnegative start step and positive batch count")
    configure_coreweave_s3()
    began = time.perf_counter()
    dataset, schedule, config = native_train_dataset(args.tokenizer)
    dataset = dataset.as_sync_dataset()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for step in range(args.start_step, args.start_step + args.steps):
        start = schedule.global_data_offset_by_step(step)
        count = schedule.batch_size_at_step(step)
        examples = dataset.get_batch(range(start, start + count))
        arrays = {
            "tokens": np.stack([np.asarray(x.tokens) for x in examples]),
            "loss_weight": np.stack([np.asarray(x.loss_weight) for x in examples]),
            "segment_ids": np.stack([np.asarray(x.attn_mask.segment_ids[0]) for x in examples]),
        }
        path = args.output / f"batch-{step:06d}.npz"
        np.savez(path, **arrays)
        record = {
            "step": step,
            "dataset_indices": [start, start + count],
            "shape": list(arrays["tokens"].shape),
            "token_min": int(arrays["tokens"].min()),
            "token_max": int(arrays["tokens"].max()),
            "loss_weight_sum": float(arrays["loss_weight"].sum()),
            "document_starts": int(np.sum(arrays["segment_ids"][:, 1:] != arrays["segment_ids"][:, :-1])),
            "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "tokenizer": config.tokenizer,
        "tokenizer_file_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.tokenizer.iterdir()) if p.is_file()
        },
        "eos_token_id": config.the_tokenizer.eos_token_id,
        "pipeline": "experiments.grug.fast_track.train.build_train_dataset",
        "elapsed_seconds": time.perf_counter() - began,
        "batches": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
