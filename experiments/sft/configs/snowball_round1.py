# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Snowball on the published Round 1 mixture and export the final checkpoint."""

import gzip
import hashlib
from dataclasses import replace
from pathlib import Path

from fray.types import ResourceConfig
from marin.execution.lazy import OUT, apply
from rigging.filesystem.storage_path import StoragePath

from experiments.sft.configs import snowball_h100
from experiments.sft.launcher import LevanterSft, PreparedData, run_sft_cli

NODES = 32
RESOURCES = ResourceConfig.with_gpu(
    "H100", count=8, cpu=64, ram="1500GB", disk="500GB", replicas=NODES, preemptible=False
)
TOKENIZER = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"

# The 377 components and 19 weight phases come from the pinned campaign
# manifest at c6676c91efe773f251d455f97c59e060b49faf1b (#11120). Phase
# boundaries are expressed in steps of eight 262,144-token sequences.
_MANIFEST = "snowball_round1_data.json"
_MANIFEST_SHA256 = "5fe5daea24431f1ce864058a3153c0656b5cd69d711f8fd66bda892ba231ad9b"


def _publish_data_manifest(*, output: str, sha256: str) -> None:
    contents = gzip.decompress(Path(__file__).with_name(f"{_MANIFEST}.gz").read_bytes())
    if hashlib.sha256(contents).hexdigest() != sha256:
        raise ValueError("Snowball Round 1 data manifest does not match its SHA-256 pin")
    directory = StoragePath(output)
    directory.mkdirs()
    (directory / _MANIFEST).write_bytes(contents)


_DATA = apply(
    "documents/snowball-round1/mixture",
    _publish_data_manifest,
    version="2026.11.19.1",
    output=OUT,
    sha256=_MANIFEST_SHA256,
)
SPEC = replace(
    snowball_h100.SPEC,
    name="checkpoints/snowball-round1",
    version="2026.11.19.1",
    model=replace(snowball_h100.SPEC.model, tokenizer_path=TOKENIZER, task_timeout_seconds=72_000),
    datasets=(),
    prepared_data=PreparedData(_DATA, _MANIFEST, TOKENIZER),
    mesh=snowball_h100.snowball_mesh(NODES),
    seq_len=262_144,
    num_train_steps=23_872,
    batch_size=8,
    pack=True,
    per_device_parallelism=1,
    xla_flags=("--xla_gpu_shard_autotuning=false",),
    wandb_project="marin-long-loop",
)
SFT = LevanterSft(SPEC)


if __name__ == "__main__":
    run_sft_cli(SFT, RESOURCES)
