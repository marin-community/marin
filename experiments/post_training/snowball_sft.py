# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball chat SFT presets for four-node OpenThoughts and the 32-node mixture."""

import hashlib
import json
import math
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import jmp
import yaml
from fray.types import ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.callbacks.profiler import XlaDumpUploadConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from levanter.eval_audit import InitialEvalMatchConfig
from levanter.models.snowball import ExpertWeightSharding, SnowballConfig
from levanter.models.snowball_sft import SnowballSftConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rolloutengine.chat_template import MARIN_CHAT_TEMPLATE

from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
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
EXP7_TOKENIZER = MIXTURE_TOKENIZER
_EXP7_RELEASE = (
    "s3://marin-us-east-08a/marin/users/benfeuer/nemotron-overthinking/2026.11.25-v10.1-balanced-toolconv/release-v10.1"
)
_EXP7_SPECIAL_TOKENS = (128006, 128007, 128002, 128003, 128005, 128011, 128009)
_EXP7_ANCHORS = (3560, 1984, 1781, 4320, 5507, 842, 128001)
_EXP6_C4_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3"
_EXP6_C4_REFERENCE = prefix_join(_EXP6_C4_ROOT, "long-loop-exp3/initializer-proof-inputs/2026.11.15.47")
_EXP6_C4_RESULT = prefix_join(_EXP6_C4_REFERENCE, "c4-reference.json")
_EXP7_CHECKPOINTER = CheckpointerConfig(save_interval=timedelta(hours=2), keep=None, keep_last_temporary_checkpoints=3)

# Frozen controls and their source artifact/revision are pinned in
# snowball_sft_data/exp6-sft-controls.json. The cooldown uses update progress
# instead of wall time, preserving its 15% duration and 0.1 floor.
EXP7_SPEC = replace(
    MIXTURE_SPEC,
    name="checkpoints/snowball/exp7-262k-32node",
    version="2026.11.25.7",
    model=HFModel(
        model_ref=prefix_join(_EXP6_C4_ROOT, "evidence/initializer-hf/2026.11.15.39"),
        tokenizer_path=EXP7_TOKENIZER,
        model=SnowballConfig(
            max_seq_len=MIXTURE_SEQ_LEN,
            moe_implementation="ring_gather_combine",
            expert_weight_sharding=ExpertWeightSharding(expert=("expert", "context"), hidden=("data",)),
        ),
        eos_token_ids=(128001, 128009),
        priority=priority_band_value("production"),
    ),
    prepared_data=PreparedData(
        ArtifactStep.adopt("sft-mixes/exp3-v10-1/release", "2026.11.25", _EXP7_RELEASE),
        "data.json",
        EXP7_TOKENIZER,
    ),
    prepared_data_format=TextLmDatasetFormat(),
    pack=256,
    pack_lookahead_window=128,
    optimizer=GrugMoeMuonHConfig(
        learning_rate=5e-5,
        adam_lr=1.153846153846154e-5,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        weight_decay=0.1,
        max_grad_norm=None,
        warmup=30,
        decay=0.15,
        min_lr_ratio=0.1,
        lr_schedule="linear",
    ),
    mesh=MeshConfig(
        axes={"context": 8, "expert": 1, "model": 1, "replica": 1},
        dcn_axes={"replica_dcn": 1, "data": -1},
        compute_mapping={},
        param_mapping={"embed": "data"},
        shared_mapping={},
    ),
    trainer_config=TrainerConfig(
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        offload_opt_state=True,
        loader_fetch_batch_size=4,
        loader_max_buffered_batches=512,
        per_device_eval_parallelism=1,
        steps_per_eval=1000,
        log_jaxprs=False,
        log_xla_hlo=False,
        jax_config={
            "jax_explain_cache_misses": "True",
            "jax_log_compiles": "True",
            "jax_softmax_custom_jvp": "True",
            "jax_threefry_partitionable": "True",
        },
        jax_compilation_cache_dir="s3://marin-us-east-08a/tmp/ttl=30d/compilation-cache",
        watch=WatchConfig(watch_targets=["params", "updates", "grads"], include_per_parameter_norms=False, interval=1),
        xla_dump_upload=XlaDumpUploadConfig(enabled=True),
        tracker=WandbConfig(entity="marin-community", project="marin-long-loop", tags=["exp7", "sft"]),
        progress_watchdog=ProgressWatchdogConfig(
            process_timeout=timedelta(minutes=30),
            startup_timeout=timedelta(minutes=90),
            step_timeout=timedelta(minutes=10),
            startup_grace_period=timedelta(0),
        ),
    ),
    z_loss_weight=1e-4,
    global_gradient_clip=1,
    snowball_sft=SnowballSftConfig(_EXP7_SPECIAL_TOKENS, _EXP7_ANCHORS, math.sqrt(32), True, True),
    token_weighted_accumulation=True,
    data_seed=0,
    tensorstore_cache_bytes=100_000_000,
    initial_eval_match=InitialEvalMatchConfig(
        data=LmDataConfig(
            tokenizer=prefix_join(_EXP6_C4_REFERENCE, "tokenizer"),
            components={
                "c4_en": DatasetComponent(
                    cache_dir=prefix_join(_EXP6_C4_ROOT, "evidence/c4-reference/2026.11.15.28/c4"),
                    split="validation",
                    format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
                )
            },
            train_weights={"c4_en": 1.0},
            auto_build_caches=False,
        ),
        reference_path=_EXP6_C4_RESULT,
        pre_reseed_reference_path=_EXP6_C4_RESULT,
        sequence_length=MIXTURE_SEQ_LEN,
        max_examples_per_dataset=1,
        special_token_ids=_EXP7_SPECIAL_TOKENS,
        absolute_tolerance=0.5,
    ),
    checkpointer=_EXP7_CHECKPOINTER,
    num_train_steps=1893,
    hf_save_dtype="bfloat16",
    wandb_project="marin-long-loop",
    xla_flags=(
        "--xla_gpu_shard_autotuning=false",
        "--xla_dump_to=/tmp/long-loop-xla",
        "--xla_dump_hlo_as_text",
        "--xla_gpu_first_collective_call_terminate_timeout_seconds=600",
    ),
)
EXP7_262K_32NODE = SFTPreset(
    "exp7-262k-32node", EXP7_SPEC, replace(MIXTURE_RESOURCES, disk="512GB", target_cluster="cw-rno2a", regions=())
)
SNOWBALL_SFT = (OPENTHOUGHTS_32K_4NODE, MIXTURE_262K_32NODE, EXP7_262K_32NODE)


if __name__ == "__main__":
    main(SNOWBALL_SFT)
