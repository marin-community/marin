# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball chat SFT presets for four-node OpenThoughts and the 32-node mixture."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import yaml
from fray.types import ResourceConfig
from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import OUT, apply
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.chat_template import MARIN_CHAT_TEMPLATE

from experiments.sft.launcher import DatasetSpec, HFModel, PreparedData, SFTPreset, SFTSpec, main

HF_MODEL = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
HF_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
DATASET_REVISION = "45fb28fcc38d352133cb28a1c8a43a2f14fea97b"
OPENTHOUGHTS_NODES = 4


def snowball_resources(nodes: int) -> ResourceConfig:
    """Allocate eight H100s per Snowball training node."""
    return ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512g", disk="256g", replicas=nodes, preemptible=False)


def snowball_mesh(nodes: int) -> MeshConfig:
    """Shard sequence positions across nodes and experts within each node."""
    return MeshConfig(
        axes={"data": 1, "replica": 1, "model": 1, "expert": -1},
        dcn_axes={"context": nodes},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
    )


_BASE_MODEL = HFModel(
    model_ref=f"{HF_MODEL}@{HF_REVISION}",
    model=SnowballConfig(),
    eos_token_ids=(128001, 128009),
)

OPENTHOUGHTS_SPEC = SFTSpec(
    name="checkpoints/snowball-openthoughts-agent-sft",
    version="2026.11.19.1",
    model=_BASE_MODEL,
    chat_template=MARIN_CHAT_TEMPLATE,
    datasets=[
        DatasetSpec(
            slug="openthoughts-agent",
            hf_dataset_id="open-thoughts/OpenThoughts-Agent-SFT-100K",
            revision=DATASET_REVISION,
            adapter_kwargs={"conversation_column": "conversations"},
            weight=1.0,
        )
    ],
    optimizer=AdamConfig(
        learning_rate=5e-5,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_schedule="cosine",
        warmup=0.0,
    ),
    mesh=snowball_mesh(OPENTHOUGHTS_NODES),
    seq_len=32768,
    pack=1,
    batch_size=8,
    num_train_steps=10,
    wandb_project="snowball-sft",
)
OPENTHOUGHTS_32K_4NODE = SFTPreset("openthoughts-32k-4node", OPENTHOUGHTS_SPEC, snowball_resources(OPENTHOUGHTS_NODES))

MIXTURE_NODES = 32
MIXTURE_RESOURCES = replace(
    snowball_resources(MIXTURE_NODES),
    cpu=64,
    ram="1500GB",
    disk="500GB",
    regions=("US-EAST-02A",),
    target_cluster="cw-us-east-02a",
)
MIXTURE_TOKENIZER = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"
MIXTURE_SEQ_LEN = 262_144
MIXTURE_BATCH_SIZE = 256

# The mixture uses the component and weight schedule from Snowball SFT Experiment 2.
_MIXTURE_NAME = "mixture-262k-32node"
_MANIFEST = f"{_MIXTURE_NAME}-data.json"
_MANIFEST_SHA256 = "cba9684f56fcaa686ee4d1832701ef20ebf343aeb0cf9a8963a85cc7bb535c0b"
_SFT_PACK = 256
_SFT_FORMAT = {
    "type": "serialized_chat",
    "message_start_id": 128006,
    "assistant_prefix": [128006, 78191, 128007, 198],
}


def _publish_data_manifest(*, output: str, sha256: str) -> None:
    sources = Path(__file__).with_name("snowball_sft_data")
    data = yaml.safe_load((sources / "sft-data.yaml").read_text())["data"]
    data["components"] = json.loads((sources / "sft-components.json").read_text())
    data["train_weights"] = json.loads((sources / "sft-train_weights.json").read_text())
    for name, component in data["components"].items():
        if name.startswith("sft/"):
            children = component["children"].values() if component["type"] == "concat" else (component,)
            for child in children:
                child["format"] = _SFT_FORMAT
                child["pack"] = _SFT_PACK
    contents = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if hashlib.sha256(contents).hexdigest() != sha256:
        raise ValueError(f"{_MIXTURE_NAME} data manifest does not match its SHA-256 pin")
    directory = StoragePath(output)
    directory.mkdirs()
    (directory / _MANIFEST).write_bytes(contents)


_DATA = apply(
    f"documents/snowball/{_MIXTURE_NAME}",
    _publish_data_manifest,
    version="2026.11.19.1",
    output=OUT,
    sha256=_MANIFEST_SHA256,
)
MIXTURE_SPEC = replace(
    OPENTHOUGHTS_SPEC,
    name=f"checkpoints/snowball/{_MIXTURE_NAME}",
    version="2026.11.19.1",
    model=replace(_BASE_MODEL, tokenizer_path=MIXTURE_TOKENIZER, task_timeout_seconds=72_000),
    datasets=(),
    prepared_data=PreparedData(_DATA, _MANIFEST, MIXTURE_TOKENIZER),
    mesh=snowball_mesh(MIXTURE_NODES),
    seq_len=MIXTURE_SEQ_LEN,
    num_train_steps=204,
    batch_size=MIXTURE_BATCH_SIZE,
    hf_save_dtype="float32",
    per_device_parallelism=1,
    xla_flags=("--xla_gpu_shard_autotuning=false",),
)
MIXTURE_262K_32NODE = SFTPreset(_MIXTURE_NAME, MIXTURE_SPEC, MIXTURE_RESOURCES)
VLLM_PARITY_SPEC = replace(
    OPENTHOUGHTS_SPEC,
    name="checkpoints/snowball-openthoughts-agent-sft-vllm-parity",
    model=replace(
        _BASE_MODEL,
        model=SnowballConfig(moe_capacity_factor=8.0, router_dot_fp32=True, fp32_activations=True),
    ),
    field_dtype_overrides={"router_bias": "float32"},
)
VLLM_PARITY_32K_4NODE = SFTPreset("vllm-parity-32k-4node", VLLM_PARITY_SPEC, OPENTHOUGHTS_32K_4NODE.resources)
SNOWBALL_SFT = (OPENTHOUGHTS_32K_4NODE, MIXTURE_262K_32NODE, VLLM_PARITY_32K_4NODE)


if __name__ == "__main__":
    main(SNOWBALL_SFT)
