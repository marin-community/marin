# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the approved Snowflake mask grid to native SFTSpec configuration."""

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import draccus
from fray.types import ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.models.snowball import GrugMoeHfConfig, SnowballConfig
from levanter.models.snowball_sft import SnowballSftConfig
from levanter.optim.config import OptimizerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import StepContext
from marin.training.training import TrainLmOnPodConfig, apply_output_path

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig  # noqa: F401 -- optimizer registration
from experiments.sft.launcher import HFModel, SFTSpec

INPUTS = Path(__file__).with_name("inputs")
SEQ_LEN = 262144
BATCH_SIZE = 8
SCHEDULE_STEPS = 4750
TASK_TIMEOUT = 259200
PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/snowball-isoflop"
VERSION = "2026.11.13.1"
BUDGETS = (3e18, 1e19, 3e19)
SIZES = (512, 768, 1024)
SPECIAL_IDS = (128006, 128007, 128002, 128003, 128005, 128011, 128009)
ANCHOR_IDS = (3560, 1984, 1781, 4320, 5507, 842, 128001)


class Mask(StrEnum):
    FULL = "full"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Cell:
    size: int
    mask: Mask
    seed: int
    stop_step: int
    checkpoint_steps: tuple[int, ...]
    phase: str

    @property
    def name(self) -> str:
        return f"{self.phase}-d{self.size}-{self.mask}-seed{self.seed}-n{self.stop_step}"


def recipe() -> dict:
    return json.loads((INPUTS / "exp1-recipe.json").read_text())


def model_receipt(size: int) -> dict:
    return json.loads((INPUTS / f"d{size}-hf-config.json").read_text())


def training_flops_per_position(size: int) -> float:
    c = model_receipt(size)["config"]
    layers = c["num_hidden_layers"]
    return 3 * lm_flops_per_token(
        hidden_dim=size,
        intermediate_dim=c["moe_intermediate_size"],
        num_layers=layers,
        num_heads=c["num_attention_heads"],
        num_kv_heads=c["num_key_value_heads"],
        seq_len=SEQ_LEN,
        vocab_size=c["vocab_size"],
        glu=True,
        num_experts=256,
        num_experts_per_tok=4,
        num_shared_experts=1,
        shared_intermediate_dim=size,
        sliding_window=2048,
        num_full_attention_layers=sum(i % 4 == 3 or i == layers - 1 for i in range(layers)),
    )


def budget_steps(size: int, budget: float) -> int:
    return math.floor(budget / (training_flops_per_position(size) * BATCH_SIZE * SEQ_LEN))


def calibration_cells() -> tuple[Cell, ...]:
    return tuple(Cell(d, mask, 0, 20, (20,), "calibration") for d in SIZES for mask in Mask)


def primary_cells() -> tuple[Cell, ...]:
    cells = []
    for d in SIZES:
        endpoints = tuple(budget_steps(d, c) for c in BUDGETS)
        for mask in Mask:
            cells.append(Cell(d, mask, 0, endpoints[-1], endpoints, "A"))
            for seed in (1, 2):
                cells.append(Cell(d, mask, seed, endpoints[1], (endpoints[1],), "A"))
    return tuple(cells)


def data_config(mask: Mask) -> LmDataConfig:
    data = recipe()["data"]
    # Only phase 0 is reached at these exposure budgets. Keep its exact weights.
    data["train_weights"] = [data["train_weights"][0]]
    if mask == Mask.ASSISTANT:
        for component in data["components"].values():
            for leaf in component.get("children", {"self": component}).values():
                if "format" in leaf and "sft" in (leaf.get("tags") or ()):
                    leaf["format"] = {
                        "type": "serialized_chat",
                        "message_start_id": 128006,
                        "assistant_prefix": [128006, 78191, 128007, 198],
                        "text_key": "text",
                    }
    return draccus.decode(LmDataConfig, data)


def build_spec(cell: Cell) -> SFTSpec:
    e1 = recipe()
    receipt = model_receipt(cell.size)
    hf_config = receipt["config"]
    # HF supplies geometry; these runtime fields must survive its reload.
    model = replace(
        SnowballConfig.from_hf_config(GrugMoeHfConfig(**hf_config)),
        attention_implementation="gpu_fa4_cute",
        moe_implementation="sonic",
        moe_capacity_factor=1.0,
        router_dot_fp32=False,
        fp32_activations=False,
    )
    mesh = MeshConfig(
        axes={"context": 8, "data": 1, "expert": 1, "replica": 1, "model": 1},
        dcn_axes={"replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
        param_mapping={"embed": "data"},
    )
    raw_trainer = e1["trainer"]["trainer"]
    raw_trainer["id"] = cell.name
    raw_trainer["seed"] = cell.seed
    raw_trainer["tracker"].update(project="snowball-isoflop", id=f"{VERSION}-{cell.name}", name=cell.name, group=VERSION)
    trainer = draccus.decode(TrainerConfig, raw_trainer)
    trainer = replace(
        trainer,
        offload_opt_state=True,
        loader_fetch_batch_size=4,
        loader_max_buffered_batches=512,
        gc_interval=1,
        gradient_accumulation_dtype="float32",
        steps_per_eval=15,
        # Preserve the original eight held-out sequences/source while scoring one
        # at a time on this single-node CP mesh.
        per_device_eval_parallelism=1,
        max_eval_batches=8,
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=None,
        require_accelerator=True,
    )
    endpoints = sorted(set(cell.checkpoint_steps))
    # Native checkpoint labels are the zero-based update index; state.step is
    # the number of completed updates used for budgets and resume offsets.
    keep = [dict(every=step - 1, until=step - 1 if i < len(endpoints) - 1 else None) for i, step in enumerate(endpoints)]
    checkpointer = CheckpointerConfig(
        save_interval=timedelta(minutes=30),
        keep=keep,
        keep_last_temporary_checkpoints=None,
        delete_old_temp_checkpoints=False,
        append_run_id_to_base_path=False,
    )
    return SFTSpec(
        name=f"checkpoints/snowball-isoflop-{cell.name}",
        version=VERSION,
        model=HFModel(
            model_ref=f"{receipt['repo']}@{receipt['revision']}",
            tokenizer_path=e1["data"]["tokenizer"],
            model_type="snowball",
            task_timeout_seconds=TASK_TIMEOUT,
            processes_per_task=8,
            priority=priority_band_value("interactive"),
        ),
        model_config=model,
        trainer_config=trainer,
        chat_template=(INPUTS / "chat-template.jinja").read_text(),
        datasets=(),
        prepared_data=data_config(cell.mask),
        evaluation_data=data_config(Mask.ASSISTANT),
        optimizer=draccus.decode(OptimizerConfig, e1["optimizer"]),
        optimizer_num_train_steps=SCHEDULE_STEPS,
        mesh=mesh,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        per_device_parallelism=1,
        num_train_steps=cell.stop_step,
        z_loss_weight=1e-4,
        snowball_sft=SnowballSftConfig(SPECIAL_IDS, ANCHOR_IDS, math.sqrt(32), True),
        token_weighted_accumulation=True,
        tensorstore_cache_bytes=2**31,
        data_seed=cell.seed,
        hf_save_dtype="bfloat16",
        xla_flags=("--xla_gpu_shard_autotuning=false", "--xla_gpu_nccl_termination_timeout_seconds=600"),
        jax_port=32768 + int(hashlib.sha256(cell.name.encode()).hexdigest()[:8], 16) % 20000,
        wandb_project="snowball-isoflop",
        checkpointer=checkpointer,
    )


def resources() -> ResourceConfig:
    return ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="1500GB", disk="500GB", replicas=1, preemptible=False)


def build_pod(cell: Cell, output_prefix: str = PREFIX) -> TrainLmOnPodConfig:
    spec = build_spec(cell)
    output = f"{output_prefix}/{VERSION}/{cell.name}"
    pod = spec.model.build_train_config(
        StepContext.for_run(output, output_prefix), spec, spec.prepared_data, resources(), cell.stop_step
    )
    return replace(pod, train_config=apply_output_path(pod.train_config, output))
