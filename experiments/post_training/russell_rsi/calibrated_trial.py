# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared artifact bindings for one calibrated four-update trial."""

from copy import deepcopy

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.training.training import LevanterCheckpoint

from experiments.evaluation.pipeline import eval_step
from experiments.post_training.russell_rsi.bootstrap_loop import PILOT_UPDATES
from experiments.post_training.russell_rsi.launch import (
    CLUSTER,
    OptimizerStepConfig,
    SamplingMode,
    evaluation_model,
    require_optimizer_updates,
    train_step,
)
from experiments.post_training.russell_rsi.replay import GROUPS_PER_UPDATE, ROLLOUTS_PER_GROUP
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, resolve_skyrl_model


def bounded_schedule(schedule: dict, protocol: str, limits: list[str], source_sha256: str | None = None) -> dict:
    """Give a sampler result a fresh trial namespace and bounded compute limits."""
    result = deepcopy(schedule)
    result.pop("schedule_sha256")
    result["legacy_sampler"] = {"protocol": result["protocol"], "pilot_number": result.pop("pilot_number")}
    if source_sha256 is not None:
        result["source_schedule_sha256"] = source_sha256
    result["replay_namespace"] = result.pop("frozen_identity")
    result["protocol"] = protocol
    for entry in result["schedule"]:
        entry["occurrence_id"] = f"{protocol}-update-{entry['update']}-group-{entry['group']}"
    result["experiment_limits"] = {
        "runs": 1,
        "updates": PILOT_UPDATES,
        "groups": GROUPS_PER_UPDATE * PILOT_UPDATES,
        "rollouts": GROUPS_PER_UPDATE * PILOT_UPDATES * ROLLOUTS_PER_GROUP,
        "additional_seeds": 0,
    }
    result["limits"] = limits
    result["schedule_sha256"] = compact_json_sha256(result)
    return result


def four_update_trial(
    data: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint],
    version: str,
    retention: ArtifactStep[Artifact],
    machine_config: dict,
    protocol: str,
) -> dict[str, ArtifactStep]:
    """Bind the trial, its optimizer gate, and its one-item serving reload."""
    trained = train_step(
        data,
        model,
        "pilot",
        version,
        retention,
        machine_config,
        protocol,
        sampling_mode=SamplingMode.CALIBRATED_REPLAY,
    )

    def update_config(ctx: StepContext):
        if ctx.is_fingerprint:
            return {"trained": artifact_identity(trained), "updates": PILOT_UPDATES}
        result = ctx.resolved(trained)
        return OptimizerStepConfig(PILOT_UPDATES, result.global_step, result.hf_model_uri)

    updates = ArtifactStep(
        name=f"documents/russell-rsi-{protocol}-optimizer-gate",
        version=version,
        artifact_type=Artifact,
        deps=(trained,),
        build_config=update_config,
        run=require_optimizer_updates,
    )
    reload_model = evaluation_model(f"russell-rsi-{protocol}-reload", SKYRL_POLICY_LOCATION, None)
    reload = eval_step(
        reload_model,
        "mmlu-smoke",
        version=version,
        deps=(trained, updates),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, reload_model),
        limit=1,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"rl": trained, "updates": updates, "reload": reload}
