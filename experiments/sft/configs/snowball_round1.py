# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Snowball on the published Round 1 mixture and export the final checkpoint."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import yaml
from fray.types import ResourceConfig
from marin.execution.lazy import OUT, apply
from rigging.filesystem.storage_path import StoragePath

from experiments.sft.configs import snowball_h100
from experiments.sft.launcher import LevanterSft, PreparedData, run_sft_cli

NODES = 32
RESOURCES = ResourceConfig.with_gpu(
    "H100",
    count=8,
    cpu=64,
    ram="1500GB",
    disk="500GB",
    replicas=NODES,
    preemptible=False,
    regions=("US-EAST-02A",),
    target_cluster="cw-us-east-02a",
)
TOKENIZER = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"
SEQ_LEN = 262_144
BATCH_SIZE = 8

# The 377 components and 19 weight phases come from the pinned campaign
# manifest at c6676c91efe773f251d455f97c59e060b49faf1b (#11120). Phase
# boundaries are expressed in steps of eight 262,144-token sequences.
_MANIFEST = "snowball_round1_data.json"
_MANIFEST_SHA256 = "5fe5daea24431f1ce864058a3153c0656b5cd69d711f8fd66bda892ba231ad9b"
_SOURCE_STEP_POSITIONS = 67_108_864
_SFT_PACK = 256
_SFT_FORMAT = {
    "type": "serialized_chat",
    "message_start_id": 128006,
    "assistant_prefix": [128006, 78191, 128007, 198],
}


def _publish_data_manifest(*, output: str, sha256: str) -> None:
    sources = Path(__file__).parent
    data = yaml.safe_load((sources / "sft-data.yaml").read_text())["data"]
    data["components"] = json.loads((sources / "sft-components.json").read_text())
    weights = json.loads((sources / "sft-train_weights.json").read_text())
    target_step_positions = SEQ_LEN * BATCH_SIZE
    if any(step * _SOURCE_STEP_POSITIONS % target_step_positions for step, _ in weights):
        raise ValueError("Source schedule boundaries do not align with the Round 1 training batch")
    data["train_weights"] = [(step * _SOURCE_STEP_POSITIONS // target_step_positions, phase) for step, phase in weights]
    for name, component in data["components"].items():
        if name.startswith("sft/"):
            children = component["children"].values() if component["type"] == "concat" else (component,)
            for child in children:
                child["format"] = _SFT_FORMAT
                child["pack"] = _SFT_PACK
    data["block_cross_document_attention"] = bool(_SFT_PACK)
    contents = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
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
    seq_len=SEQ_LEN,
    num_train_steps=23_872,
    batch_size=BATCH_SIZE,
    hf_save_dtype="float32",
    per_device_parallelism=1,
    xla_flags=("--xla_gpu_shard_autotuning=false",),
    wandb_project="marin-long-loop",
)
SFT = LevanterSft(SPEC)


if __name__ == "__main__":
    run_sft_cli(SFT, RESOURCES)
