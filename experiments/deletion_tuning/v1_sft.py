# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Deletion Tuning V1 entry: the Exp3 production SFT recipe, retargeted at the DT main.

V1 trains a deletion mask on the routed experts' gate projection with SFT, per the
paper's SFT recipe (arXiv 2611.10866) and the plan-of-record decisions fixed on
2026-11-18:

- Data: the Exp3 V5 release exactly as the Exp3 production run consumed it
  (``release-complete-v5.json``), consumed through the same
  ``production_sft_recipe`` / ``prepared_supervision_data`` machinery, seed 0.
- Length: the fixed token budget (~5.03B positions) on the 64-GPU plan: 300
  updates at batch 64 x 262,144 (8 nodes, EP=8 per node, CP=8 across nodes,
  one sequence per device-slot — the production per-device load), per the
  user's efficiency decision; the original 75 x 256 plan had the same product.
- Mask: routed-expert ``gate_proj`` only; eta 1/2048; lambda 1e-5; alpha ramp
  1e2 -> 1e3 over the actual run (horizon == steps); AdamW lr 5e-4 constant on
  the gate logits only.
- Wanda calibration on the first 64 training batches.

Decisions the fixed plan does not determine are recorded here and in the plan
of record's Decision Log:

- AdamW weight decay 0.0: the paper specifies no weight decay for the logit
  optimizer, and the -lambda ||W~||^2 regularizer is the only weight-space
  term.
- The Exp3 initial-eval-match gate is dropped: the Deletion init deliberately
  perturbs the masked weights (the soft mask at alpha=1e2 scales keep-group
  weights to about 0.95), so a 0.5-nat identity gate against the unmodified
  base would reject a correct initialization. Checkpoint identity is certified
  upstream by the verified initializer receipt.
- Duration and guard scale from the Exp3 convention (guard = devices x
  deadline): a 5 h deadline over a ~3.8 h point estimate gives the 75 updates
  headroom, so the row is not truncated.

Dry-run (default): resolve the full production spec locally without
submitting -- verify the release pins, build the recipe and spec through the
Exp3 gates, build the launch step, fingerprint it, and construct the dispatch
request against a local output directory. The toy-scale CPU smoke is a
separate entry module: ``python -m experiments.deletion_tuning.v1_smoke``.
"""

import copy
import dataclasses
import json
import logging
import tempfile
from datetime import timedelta
from pathlib import Path

import click
import draccus
from fray.types import ResourceConfig
from levanter.models.snowball import SnowballConfig
from levanter.models.snowball_deletion import DeletionTuningConfig, DeletionTuningScope
from levanter.optim.config import AdamConfig
from marin.execution.lazy import StepContext
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.deletion_tuning.levanter_train import DeletionHFModel, run_deletion_train_lm
from experiments.post_training.long_loop.exp3_production import (
    GEOMETRY,
    DataRelease,
    production_sft_recipe,
    read_data_release,
)
from experiments.post_training.long_loop.exp3_recipe import exp3_sft_spec
from experiments.sft.launcher import SFTSpec, levanter_train_request, sft_step

logger = logging.getLogger(__name__)

NAME = "deletion-tuning/v1-sft"
VERSION = "2026.11.19.1"
OUTPUT_PREFIX = "s3://marin-us-east-02a/marin/users/benfeuer/snowball-deletion-tuning"

DATA_RELEASE_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1"
    "/release-v5/release-complete-v5.json"
)
DATA_RELEASE_SHA256 = "3e2161945e1adc60c9edcdb6b0d7395946966dc4e0d16b88660fbc38989603f5"

EXP3_UPDATES = GEOMETRY.updates
# The 64-GPU V1 plan (the user's efficiency decision): 8 nodes x 8 H100, EP=8
# per node, CP=8 across nodes, batch 64 (one sequence per device-slot, the
# production per-device token load), and 300 steps so the token budget equals
# the original 75 x 256 x 262,144 plan exactly.
V1_DEVICES = 64
V1_NODES = 8
V1_BATCH = 64
V1_UPDATES = 300
V1_TARGET_POSITIONS = V1_UPDATES * V1_BATCH * GEOMETRY.sequence_length
assert (
    V1_TARGET_POSITIONS == EXP3_UPDATES * 256 * GEOMETRY.sequence_length
), "the 64-GPU plan must preserve the original token budget"
# the release freezes mixture_block_size at the production 256-sequence phase
# geometry (64,000 = 250 updates x 256); the schedule boundaries rescale with
# the batch, so the block follows the rescaled phase length (250 x 64)
V1_MIXTURE_BLOCK = 250 * V1_BATCH
V1_CALIBRATION_BATCHES = 64
V1_ALPHA_START = 1e2
V1_ALPHA_END = 1e3
V1_ETA = 1 / 2048
V1_LAMBDA = 1e-5
V1_LOGIT_LR = 5e-4
V1_DURATION = timedelta(hours=12)
V1_STARTUP_TIMEOUT = timedelta(hours=2)
# The guard convention: H100-hour ceiling = devices x deadline.
V1_H100_HOUR_LIMIT = V1_DEVICES * int(V1_DURATION.total_seconds() // 3600)

# H100-hour estimate factors: per-step wall stays at the Exp3 per-device rate
# (each device still processes one sequence-equivalent per step), with the
# startup, calibration, and export phases measured from the production row.
ESTIMATE_STARTUP_HOURS = 1.5
ESTIMATE_FINALIZATION_HOURS = 0.4
EXP3_WALL_HOURS = 20.0
EXP3_STARTUP_HOURS = 1.0
EXP3_SECONDS_PER_STEP = (EXP3_WALL_HOURS - EXP3_STARTUP_HOURS) * 3600 / EXP3_UPDATES
CALIBRATION_FRACTION_OF_STEP = 1 / 3


def v1_deletion_config() -> DeletionTuningConfig:
    """The fixed V1 Deletion Tuning block."""
    return DeletionTuningConfig(
        scope=DeletionTuningScope.ROUTED_GATE,
        eta=V1_ETA,
        lam=V1_LAMBDA,
        alpha_start=V1_ALPHA_START,
        alpha_end=V1_ALPHA_END,
        lr=V1_LOGIT_LR,
        calibration_sequences=V1_CALIBRATION_BATCHES,
        seed=0,
    )


def v1_optimizer() -> AdamConfig:
    """AdamW on the gate logits only (the trainable filter scopes it): constant lr, no decay."""
    return AdamConfig(learning_rate=V1_LOGIT_LR, weight_decay=0.0, warmup=0, lr_schedule="constant")


def v1_recipe(production_recipe: dict) -> dict:
    """Overlay the fixed V1 decisions on the Exp3 production SFT recipe.

    A pure transform of the dict ``production_sft_recipe`` returns, so the Exp3
    controls (mesh, resources, packing, gates) carry over untouched except for
    the recorded overrides.
    """
    assert EXP3_UPDATES == 746, f"the token-budget equivalence assumes 746 Exp3 updates, got {EXP3_UPDATES}"
    recipe = copy.deepcopy(production_recipe)
    if recipe["format"] != "sft_spec" or recipe["profile"] != "snowball_exp3":
        raise ValueError("V1 must build on the Exp3 production SFT recipe")
    if recipe["seq_len"] != GEOMETRY.sequence_length:
        raise ValueError("V1 keeps the Exp3 context")

    trainer = recipe["trainer"]
    trainer["num_train_steps"] = V1_UPDATES
    trainer["train_batch_size"] = V1_BATCH
    trainer["training_budget"]["duration"] = draccus.encode(V1_DURATION)
    # the Deletion calibration (64 forward-only batches, roughly half an hour)
    # runs after the base load and before the first step, so the startup
    # watchdog budget scales with it (the Exp3 budget covered load + first
    # compile only)
    trainer["progress_watchdog"]["startup_timeout"] = draccus.encode(V1_STARTUP_TIMEOUT)
    # the 64-GPU mesh: EP=8 within a node, CP=8 across the 8 nodes
    trainer["mesh"]["dcn_axes"]["context"] = V1_NODES
    trainer["mesh"]["dcn_axes"]["replica_dcn"] = 1
    recipe["stop_after_steps"] = V1_UPDATES
    recipe["batch_size"] = V1_BATCH
    recipe["task_timeout_seconds"] = int(V1_DURATION.total_seconds())
    recipe["target_positions"] = V1_TARGET_POSITIONS
    recipe["stage_guard"]["h100_hour_limit"] = V1_H100_HOUR_LIMIT
    resources = recipe["resources"]
    assert resources["device"]["count"] == 8 and resources["replicas"] == 2 * V1_NODES, resources["replicas"]
    resources["replicas"] = V1_NODES
    # the Deletion init deliberately perturbs the masked weights, so the Exp3
    # identity gate against the unmodified base would reject a correct init.
    recipe["initial_eval_match"] = None

    model = draccus.decode(SnowballConfig, recipe["model"])
    model = dataclasses.replace(model, deletion_tuning=v1_deletion_config())
    recipe["model"] = draccus.encode(model)
    recipe["model"].pop("type", None)
    return recipe


def deletion_sft_spec(name: str, version: str, recipe: dict) -> SFTSpec:
    """Build the V1 spec: every Exp3 production gate, then the Deletion optimizer and dispatch.

    ``exp3_sft_spec`` runs all Exp3 controls (stage guard, source-loss pins,
    verified initializer, packing, FP32/BF16, H100, mesh shapes); the V1 entry
    swaps the MuonH optimizer for the paper's logit-only AdamW and sends the
    pod config to the DT main.
    """
    spec = exp3_sft_spec(name, version, recipe)
    settings = draccus.decode(SnowballConfig, recipe["model"]).deletion_tuning
    if settings is None:
        raise ValueError("the V1 recipe must carry the deletion_tuning block")
    # the release freezes mixture_block_size at the production phase geometry;
    # the schedule boundaries rescale with the batch, so the block follows the
    # rescaled phase length (the shuffle granularity is a recorded override)
    prepared = spec.prepared_data
    if prepared is not None:
        prepared = dataclasses.replace(prepared, mixture_block_size=V1_MIXTURE_BLOCK)
    model = spec.model
    return dataclasses.replace(
        spec,
        prepared_data=prepared,
        optimizer=v1_optimizer(),
        optimizer_num_train_steps=None,
        model=DeletionHFModel(
            model_ref=model.model_ref,
            tokenizer_path=model.tokenizer_path,
            model_type=model.model_type,
            eos_token_ids=model.eos_token_ids,
            task_timeout_seconds=model.task_timeout_seconds,
            priority=model.priority,
        ),
    )


def v1_data_release(*, require_ready: bool) -> DataRelease:
    """Read and verify the pinned Exp3 V5 release (URI and digest per #11658)."""
    release = read_data_release(DATA_RELEASE_URI, require_ready=require_ready)
    if release.pin.sha256 != DATA_RELEASE_SHA256:
        raise ValueError("the V5 release digest moved; re-pin before resolving V1")
    return release


def v1_step(release: DataRelease) -> tuple:
    """The V1 launch step and its spec and resources (the graph the runner would submit)."""
    recipe = v1_recipe(production_sft_recipe(release))
    spec = deletion_sft_spec(NAME, VERSION, recipe)
    resources = draccus.decode(ResourceConfig, recipe["resources"])
    return sft_step(spec, resources), spec, resources


def estimated_h100_hours() -> float:
    """The point estimate: startup + Wanda calibration + 300 steps + finalization.

    Each device still processes one sequence-equivalent per step (batch 64 over
    64 devices), so the per-step wall stays at the Exp3 per-device rate.
    """
    calibration_hours = V1_CALIBRATION_BATCHES * EXP3_SECONDS_PER_STEP * CALIBRATION_FRACTION_OF_STEP / 3600
    step_hours = V1_UPDATES * EXP3_SECONDS_PER_STEP / 3600
    return ESTIMATE_STARTUP_HOURS + calibration_hours + step_hours + ESTIMATE_FINALIZATION_HOURS


def _resolve_dispatch(spec: SFTSpec, resources: ResourceConfig, task_timeout_seconds: int, output_path: str):
    """Build the run-context pod config and dispatch request without submitting.

    The consumed-config record lands beside ``output_path``; the dry-run points
    that at a local directory, so nothing is written remotely.
    """
    ctx = StepContext.for_run(
        output_path=output_path,
        prefix=OUTPUT_PREFIX,
        runtime_args={"train_resources": resources},
    )
    pod_config = spec.model.build_train_config(ctx, spec, spec.prepared_data, resources, spec.num_train_steps)
    # the dispatch DeletionHFModel.run would submit: the DT pod runner over the same pod config
    return pod_config, levanter_train_request(
        pod_config, task_timeout_seconds=task_timeout_seconds, run_fn=run_deletion_train_lm
    )


def _job_shape(spec: SFTSpec, resources: ResourceConfig) -> dict:
    device = resources.device
    gpu_count = device.count * resources.replicas
    mesh = spec.trainer_config.mesh
    return {
        "gpus": {
            "variant": device.variant,
            "count": device.count,
            "replicas": resources.replicas,
            "total": gpu_count,
        },
        "mesh": {
            "axes": dict(mesh.axes),
            "dcn_axes": dict(mesh.dcn_axes),
            "compute_mapping": dict(mesh.compute_mapping),
        },
        "num_train_steps": spec.num_train_steps,
        "global_batch": spec.batch_size,
        "sequence_length": spec.seq_len,
        "target_positions": spec.num_train_steps * spec.batch_size * spec.seq_len,
        "calibration_batches": V1_CALIBRATION_BATCHES,
        "duration_hours": V1_DURATION.total_seconds() / 3600,
        "h100_hour_point_estimate": round(estimated_h100_hours() * gpu_count),
        "h100_hour_guard": V1_H100_HOUR_LIMIT,
    }


def dry_run(output: Path | None) -> dict:
    """Resolve the V1 production job end to end locally; no submission, no remote writes."""
    configure_coreweave_s3()
    release = v1_data_release(require_ready=False)
    step, spec, resources = v1_step(release)
    fingerprint = step.fingerprint()

    with tempfile.TemporaryDirectory(prefix="v1-sft-dry-run-") as tmp:
        _pod_config, request = _resolve_dispatch(
            spec, resources, int(V1_DURATION.total_seconds()), str(Path(tmp) / "v1-sft")
        )
        consumed = json.loads((Path(tmp) / "v1-sft" / "consumed-config.json").read_text())
    entrypoint = request.entrypoint.callable_entrypoint.callable
    assert entrypoint is run_deletion_train_lm, "the dispatch must target the Deletion Tuning pod runner"
    train_config = consumed["train_config"]

    summary = {
        "name": NAME,
        "version": VERSION,
        "release": {"uri": release.pin.uri, "sha256": release.pin.sha256, "training_ready": release.training_ready},
        "job_shape": _job_shape(spec, resources),
        "deletion_tuning": _deletion_summary(train_config),
        "optimizer": train_config["optimizer"],
        "initial_eval_match": train_config["initial_eval_match"],
        "fingerprint": fingerprint,
        "dispatch": {
            "entrypoint": f"{entrypoint.__module__}:{entrypoint.__qualname__}",
            "job_name": request.name,
            "resources": consumed["resources"],
            "timeout_seconds": int(V1_DURATION.total_seconds()),
            "output_path": OUTPUT_PREFIX,
        },
        "not_yet_wired": [
            "hard-mask export (deleted weights zeroed) and the packed mask artifact on the "
            "Levanter path: the resolved run would export the standard HF checkpoint of the "
            "frozen weights; the mask artifact export must land before the Micro210 row"
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _deletion_summary(train_config: dict) -> dict:
    """The resolved Deletion Tuning block from the consumed train config."""
    dt = train_config["model"].get("deletion_tuning")
    assert dt is not None, "the consumed config lost the deletion_tuning block"
    assert dt["scope"] == "routed_gate", dt["scope"]
    return dt


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write the resolution summary JSON here.")
def main(output: Path | None) -> None:
    summary = dry_run(output)
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
