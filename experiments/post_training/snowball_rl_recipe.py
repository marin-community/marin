# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball preset composition and native execution resources."""

from __future__ import annotations

from dataclasses import asdict

import yaml
from iris.rpc import job_pb2
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections

from experiments.post_training import snowball_rl_presets as presets

RUN_TIMEOUT_HOURS = 72
DEFAULT_WANDB_ENTITY = "marin-community"


def _training_config(preset: presets.Preset) -> RecipePatch:
    recipe = preset.recipe
    micro_batch, in_flight = recipe.layout.workload.batch_parameters()
    geometry = asdict(recipe.layout.megatron)
    return recipe.config.merge(
        RecipePatch(
            trainer=sections.Trainer(
                max_steps=preset.updates,
                micro_forward_batch_size_per_gpu=micro_batch,
                eval_interval=preset.updates if preset.evaluations else -1,
                resume_mode="none",
                policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(**geometry)),
                ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**geometry)),
                rollout_buffer=sections.RolloutBuffer(
                    max_in_flight=in_flight,
                ),
            ),
        ),
    )


def _model_input(preset: presets.Preset, init: ArtifactStep | None) -> ArtifactHfModel:
    if init is None:
        return preset.model
    return ArtifactHfModel(init, preset.model.tokenizer_uri, preset.model.tokenizer_revision)


def run_spec(
    preset: presets.Preset,
    *,
    name: str,
    version: str,
    tasks: ArtifactStep | None = None,
    init: ArtifactStep | None = None,
) -> SkyRLSpec:
    """Compose a preset into a SkyRL specification, optionally on other tasks or initial weights."""
    if len(preset.evaluations) > 1:
        raise ValueError(f"{preset.name} declares {len(preset.evaluations)} evaluations; one is supported")
    config = _training_config(preset)
    selection = tasks or presets.selection_artifact(preset.tasks)
    workload = preset.recipe.layout.workload
    validation_data = ()
    evaluation = preset.evaluations[0] if preset.evaluations else None
    if evaluation is not None:
        config, validation_data = presets.configure_evaluation(
            evaluation, config, evaluation.tasks(selection, preset.eval_lite)
        )
    config = config.merge(RecipePatch(trainer=sections.Trainer(callbacks=workload.callbacks(preset, evaluation))))
    plan = workload.plan
    return SkyRLSpec(
        name=name,
        version=version,
        config_yaml=yaml.safe_dump(SkyRLRecipe.from_document(config.to_skyrl()).to_skyrl(), sort_keys=False),
        model=_model_input(preset, init),
        train_data=workload.training_data(selection),
        validation_data=validation_data,
        topology=SkyRLTopology(
            num_nodes=plan.num_nodes, gpus_per_node=plan.policy_num_gpus_per_node, gpu_variant="H100", role_plan=plan
        ),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
        seed=presets.SEED,
    )


def run_step(preset: presets.Preset, spec: SkyRLSpec) -> ArtifactStep[SkyRLRun]:
    """Bind one Snowball specification to its native execution resources."""
    layout = preset.recipe.layout
    return skyrl_step(
        spec,
        IrisSkyRLExecution(
            cluster=layout.cluster,
            cluster_config=presets.cluster_config_path(layout.cluster),
            cpu=32,
            memory=layout.host_memory,
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=layout.cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=RUN_TIMEOUT_HOURS,
            job_timeout_seconds=RUN_TIMEOUT_HOURS * 3600,
            wandb_entity=DEFAULT_WANDB_ENTITY,
            required_env_vars=("WANDB_API_KEY",),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=preset.publishes,
    )
