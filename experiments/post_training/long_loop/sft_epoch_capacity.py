# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure full cache capacities on regional CPUs before an epoch launch."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import click
import draccus
import jax
import numpy as np
from fray.types import CpuConfig, ResourceConfig
from haliax import Axis
from iris.cluster.client.job_info import get_job_info
from levanter.data.dataset import PermutationDataset, SlicedAsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.utils.thread_utils import blocking_wait
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from rigging.filesystem.cluster_config import StoreType
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training.long_loop.run_spec import FilePin, read_pin
from experiments.post_training.long_loop.sft_epoch import packing_code_sha256


def measure_capacity(
    *, output: str, data_pin: FilePin, sequence_length: int, packing_cap: int, lookahead: int | None
) -> None:
    """Read document offsets and native pack indices, without token bodies or training."""
    configure_coreweave_s3()
    if credentials := s3_credentials(StoreType.COREWEAVE):
        key, secret = credentials
        os.environ.update(AWS_ACCESS_KEY_ID=key, AWS_SECRET_ACCESS_KEY=secret)
    raw = read_pin(data_pin, Path.cwd())
    data = draccus.decode(LmDataConfig, raw)
    data = dataclasses.replace(
        data,
        components={
            name: dataclasses.replace(component, pack=packing_cap, pack_lookahead_window=lookahead)
            for name, component in data.components.items()
        },
    )
    if not data.cache_catalog or any(
        raw.get(key) is not None for key in ("max_train_batches", "target_budget", "experiment_budget")
    ):
        raise ValueError("Epoch capacity requires a complete catalog and refuses dataset slices")
    catalog_bytes = StoragePath(data.cache_catalog).read_bytes()
    catalog = json.loads(catalog_bytes)
    datasets = data.train_sets(Axis("position", sequence_length), key=jax.random.PRNGKey(0))
    sources = {}
    for name, dataset in datasets.items():
        length = blocking_wait(dataset.async_len())
        indices = np.arange(length, dtype=np.int64)
        while isinstance(dataset, (PermutationDataset, SlicedAsyncDataset)):
            if isinstance(dataset, PermutationDataset):
                indices = np.asarray(blocking_wait(dataset._get_permutation())(indices), dtype=np.int64)
            else:
                indices = indices + dataset.start_index
            dataset = dataset.dataset
        packed = dataset.packed
        lengths = packed._lengths["input_ids"]
        tokens = np.asarray(
            [sum(min(int(lengths[index]), sequence_length) for index in pack) for pack in packed._pack_indices],
            dtype=np.int64,
        )
        ledger = catalog["splits"]["train"][name]["ledger"]
        if int(lengths.sum()) != ledger["field_counts"]["input_ids"]:
            raise ValueError(f"Native offsets disagree with the sealed token population: {name}")
        sources[name] = {
            "sequence_length": sequence_length,
            "documents": len(lengths),
            "tokens": int(lengths.sum()),
            "packed_sequences": len(tokens),
            "training_sequences": length,
            "training_tokens": int(tokens[indices].sum()),
            "clipped_tokens": int(lengths.sum() - tokens.sum()),
            "reserved_validation_sequences": len(tokens) - length,
        }
        print(f"SFT_CAPACITY {name} {json.dumps(sources[name], sort_keys=True)}", flush=True)
    receipt = {
        "schema": 1,
        "data_pin": dataclasses.asdict(data_pin),
        "catalog_uri": data.cache_catalog,
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "sequence_length": sequence_length,
        "packing_cap": packing_cap,
        "lookahead": lookahead,
        "packing_sha256": packing_code_sha256(),
        "validation_reservations": raw["num_validation_sequences"],
        "sources": sources,
        "token_bodies_read": False,
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(output) / "capacity.json"
    target.write_bytes(payload)
    assert target.read_bytes() == payload
    print(
        "SFT_CAPACITY_COMPLETE " + json.dumps({"path": str(target), "sha256": hashlib.sha256(payload).hexdigest()}),
        flush=True,
    )


@click.command(help=__doc__)
@click.option("--data", required=True)
@click.option("--sha256", required=True)
@click.option("--output-prefix", required=True)
@click.option("--sequence-length", type=int, default=262144)
@click.option("--packing-cap", type=int, default=256)
@click.option(
    "--lookahead", type=click.IntRange(min=0), default=128, help="Packing lookahead; 0 selects sequential packing."
)
@build_options
def main(
    data: str, sha256: str, output_prefix: str, sequence_length: int, packing_cap: int, lookahead: int
) -> ArtifactStep:
    configure_coreweave_s3()
    if not data.startswith("s3://marin-us-east-08a/") or not output_prefix.startswith("s3://marin-us-east-08a/"):
        raise click.UsageError("Full-store capacity measurement must stay in the US-EAST-08A data region")
    if any(part == "tmp" or part.startswith("ttl=") for part in StoragePath(output_prefix).segments):
        raise click.UsageError("Capacity receipts require a permanent output prefix")
    os.environ["MARIN_PREFIX"] = output_prefix
    return apply(
        "long-loop/sft-epoch-capacity",
        remote(
            measure_capacity,
            resources=ResourceConfig(
                device=CpuConfig(),
                cpu=8,
                ram="64GB",
                disk="64GB",
                target_cluster=None if get_job_info() is not None else "cw-us-east-08a",
            ),
            timeout=Duration.from_hours(2),
        ),
        output=OUT,
        data_pin=FilePin(data, sha256),
        sequence_length=sequence_length,
        packing_cap=packing_cap,
        lookahead=lookahead or None,
    )


if __name__ == "__main__":
    main()
