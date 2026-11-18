# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The V1 production-shape smoke: the real 67B model on one 8xH100 node.

Entry point (on an Iris GPU node): ``python -m
experiments.deletion_tuning.v1_production_smoke``.

Everything matches the resolved V1 production recipe except the recorded
scale overrides: one node (8 H100, expert=8 ICI, no DCN), global batch 8
(one sequence per device, the production per-device footprint), three
optimizer steps, Wanda calibration on the first 2 training batches, the
in-region staged base copy instead of the 08a initializer receipt (this node
is in 02a), no training-budget deadline, and W&B disabled. The data is the
Exp3 V5 release through the production supervision path; a few steps read
only tens of megabytes of token caches.

The run exercises the full production path at real scale: the HF load of the
67B base, the Wanda calibration, the gate-logit init (eta 1/2048 over the
real 21.8B gate weights), the AdamW logit-only optimizer state, the alpha
ramp, and the hard-mask HF export with the packed mask artifact.
"""

import dataclasses
import json
import logging
import tempfile
from datetime import timedelta
from pathlib import Path

import click
import draccus
from iris.rpc.proto_display import priority_band_value
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.models.snowball import SnowballConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.rl.deletion_tuning import read_mask_artifact
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.deletion_tuning.levanter_train import DeletionHFModel, run_deletion_train_lm
from experiments.deletion_tuning.v1_sft import VERSION, v1_data_release, v1_optimizer
from experiments.post_training.long_loop.exp3_production import production_sft_recipe
from experiments.post_training.long_loop.exp3_recipe import exp3_controls
from experiments.post_training.long_loop.source_loss import prepared_supervision_data, source_loss_metadata
from experiments.sft.launcher import SFTSpec, _levanter_pod_config

logger = logging.getLogger(__name__)

SMOKE_NAME = "deletion-tuning/v1-production-smoke"
BASE_02A = "s3://marin-us-east-02a/marin/users/benfeuer/snowball-deletion-tuning/inputs/base-hf/2026.11.18.1"
TOKENIZER = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"
SMOKE_STEPS = 3
SMOKE_BATCH = 8
SMOKE_CALIBRATION_BATCHES = 2
SMOKE_TIMEOUT = timedelta(hours=4)

# Recorded scale overrides versus the resolved V1 production recipe: one node
# (8 H100, expert=8 ICI, no DCN), batch 8 (one sequence per device), three
# steps, calibration on 2 batches, the 02a staged base copy, no budget
# deadline, W&B disabled. Model, context, packing, controls, precision, and
# the deletion block (eta 1/2048) are unchanged.
_SCALE_NOTE = (
    "one node 8xH100 (expert=8 ICI, no DCN), batch 8 (1 seq/device), 3 steps, "
    "calibration 2 batches, 02a staged base copy, no budget deadline, W&B disabled"
)


def _smoke_spec(release) -> SFTSpec:
    """The V1 production recipe with the one-node scale overrides applied."""
    recipe = production_sft_recipe(release)
    trainer = recipe["trainer"]
    trainer["num_train_steps"] = SMOKE_STEPS
    trainer["train_batch_size"] = SMOKE_BATCH
    trainer["training_budget"] = None
    trainer["checkpointer"] = draccus.encode(
        CheckpointerConfig(save_interval=timedelta(hours=6), keep=None, keep_last_temporary_checkpoints=1)
    )
    mesh = MeshConfig(
        axes={"data": 1, "expert": 8, "model": 1, "replica": 1},
        dcn_axes={"context": 1, "replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
    )
    trainer["mesh"] = draccus.encode(mesh)
    trainer["progress_watchdog"] = draccus.encode(
        ProgressWatchdogConfig(
            startup_timeout=timedelta(minutes=45),
            step_timeout=timedelta(minutes=30),
            process_timeout=timedelta(minutes=30),
            startup_grace_period=timedelta(0),
        )
    )
    recipe["stop_after_steps"] = SMOKE_STEPS
    recipe["batch_size"] = SMOKE_BATCH
    recipe["task_timeout_seconds"] = int(SMOKE_TIMEOUT.total_seconds())
    recipe["target_positions"] = SMOKE_STEPS * SMOKE_BATCH * recipe["seq_len"]
    recipe["stage_guard"] = {"h100_hour_limit": 8 * 4}

    model = draccus.decode(SnowballConfig, recipe["model"])
    dt = dataclasses.replace(model.deletion_tuning, calibration_sequences=SMOKE_CALIBRATION_BATCHES)
    model = dataclasses.replace(model, deletion_tuning=dt)
    recipe["model"] = draccus.encode(model)
    recipe["model"].pop("type", None)

    # the production data path: the release's caches and supervision metadata
    metadata = source_loss_metadata(recipe["source_loss_metadata"], recipe["source_loss_metadata_sha256"])
    data = prepared_supervision_data(
        draccus.decode(LmDataConfig, recipe["data"]),
        metadata,
        packing_cap=recipe["packing_cap"],
        lookahead=recipe["packing_lookahead"],
    )
    trainer_config = draccus.decode(TrainerConfig, trainer)
    trainer_config = dataclasses.replace(
        trainer_config,
        tracker=WandbConfig(entity=recipe["wandb_entity"], project=recipe["wandb_project"], mode="disabled"),
    )
    return SFTSpec(
        name=SMOKE_NAME,
        version=VERSION,
        model=DeletionHFModel(
            model_ref=BASE_02A,
            tokenizer_path=TOKENIZER,
            model_type="snowball",
            eos_token_ids=(128001, 128009),
            task_timeout_seconds=recipe["task_timeout_seconds"],
            priority=priority_band_value("production"),
        ),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=(),
        prepared_data=data,
        model_config=model,
        trainer_config=trainer_config,
        optimizer=v1_optimizer(),
        optimizer_num_train_steps=None,
        z_loss_weight=1e-4,
        global_gradient_clip=1,
        snowball_sft=exp3_controls(),
        token_weighted_accumulation=True,
        data_seed=0,
        tensorstore_cache_bytes=recipe["tensorstore_cache_bytes"],
        mesh=mesh,
        seq_len=recipe["seq_len"],
        batch_size=SMOKE_BATCH,
        per_device_parallelism=1,
        num_train_steps=SMOKE_STEPS,
        checkpointer=trainer_config.checkpointer,
        pack=True,
        hf_save_dtype="bfloat16",
        wandb_project=recipe["wandb_project"],
        wandb_entity=recipe["wandb_entity"],
        xla_flags=tuple(recipe["xla_flags"]),
    )


def run_production_smoke(output_root: Path | None = None) -> dict:
    """Run the real-model V1 smoke on this node and report what ran."""
    configure_coreweave_s3()
    release = v1_data_release(require_ready=False)
    spec = _smoke_spec(release)
    with tempfile.TemporaryDirectory(prefix="v1-production-smoke-") as tmp:
        output_path = Path(tmp) / "run"
        from marin.training.training import TrainLmOnPodConfig

        pod_config = TrainLmOnPodConfig(
            output_path=str(output_path),
            resources=_gpu_resources(),
            env_vars={"WANDB_MODE": "disabled", "JAX_TRACEBACK_FILTERING": "off"},
            train_config=_pod_train_config(spec, str(output_path)),
        )
        run_deletion_train_lm(pod_config)

        exports = sorted(str(path) for path in output_path.rglob("*.safetensors"))
        assert exports, "the production smoke must produce the end-of-run HF export"
        mask_dirs = sorted(output_path.rglob("deletion-mask"))
        assert mask_dirs, "the production smoke export must carry the deletion-mask artifact"
        keep_masks = read_mask_artifact(mask_dirs[0])
        assert keep_masks, "the mask artifact must carry the routed gate entries"
        kept = sum(int(mask.sum()) for mask in keep_masks.values())
        total = sum(int(mask.size) for mask in keep_masks.values())
        summary = {
            "scale_note": _SCALE_NOTE,
            "steps": SMOKE_STEPS,
            "batch_size": SMOKE_BATCH,
            "sequence_length": spec.seq_len,
            "calibration_batches": SMOKE_CALIBRATION_BATCHES,
            "hf_export": exports[:1],
            "mask_artifact": {"entries": len(keep_masks), "kept": kept, "total": total},
        }
        logger.info("production smoke complete: %s", json.dumps(summary, indent=2))
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "v1-production-smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def _pod_train_config(spec: SFTSpec, output_path: str):
    """The pod config the launcher would dispatch, built against a local output path."""

    class _Ctx:
        is_fingerprint = False
        output_path = output_path

        @staticmethod
        def runtime_arg(_key):
            return _gpu_resources()

    return _levanter_pod_config(
        _Ctx(),
        spec,
        spec.prepared_data,
        _gpu_resources(),
        spec.num_train_steps,
        model_config=spec.model_config,
        initialize_from_hf=spec.model.model_ref,
        initialize_model_from_checkpoint_path=None,
        use_hf_model_config=True,
        eos_token_ids=spec.model.eos_token_ids,
    )


def _gpu_resources():
    from fray.types import GpuConfig, ResourceConfig

    return ResourceConfig(
        device=GpuConfig(variant="H100", count=8),
        cpu=64,
        ram="800GB",
        disk="512GB",
        replicas=1,
    )


@click.command(help=__doc__)
def main() -> None:
    run_production_smoke()


if __name__ == "__main__":
    main()
