# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball preset composition and native execution resources."""

from __future__ import annotations

from dataclasses import asdict, replace

import click
import yaml
from iris.rpc import job_pb2
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLModelSource,
    SkyRLPolicyModel,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from rigging.filesystem.storage_path import prefix_join
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections
from skyrlrecipe.declared import (
    EvaluationSampling,
)
from skyrlrecipe.documents import MISSING, get_path

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.task_curation.selection import GraderKind

RUN_TIMEOUT_HOURS = 72
DEFAULT_WANDB_ENTITY = "marin-community"


def _training_config(
    preset: presets.Preset,
    settings: tuple[str, ...],
) -> RecipePatch:
    recipe = preset.recipe
    micro_batch, in_flight = recipe.layout.workload.batch_parameters(recipe.config)
    geometry = asdict(recipe.layout.megatron)
    budget = recipe.config.context_budget
    assert budget is not None
    if recipe.fixed_reply_tokens is not None:
        budget = budget.replace(max_new_tokens_per_turn=recipe.fixed_reply_tokens)
    config = recipe.config.merge(
        RecipePatch(
            context_budget=budget,
            trainer=sections.Trainer(
                max_steps=preset.updates or 1,
                micro_forward_batch_size_per_gpu=micro_batch,
                eval_interval=max(1, preset.updates) if preset.evaluations else -1,
                resume_mode="latest" if preset.checkpoint_every else "none",
                policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(**geometry)),
                ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**geometry)),
                rollout_buffer=sections.RolloutBuffer(
                    max_in_flight=in_flight,
                ),
            ),
            generator=sections.Generator(
                sampling_params=sections.SamplingParams(ignore_eos=recipe.fixed_reply_tokens is not None),
            ),
        ),
    )
    return presets.apply_settings(config, settings, preset)


def _model_input(preset: presets.Preset, init: ArtifactStep | None) -> tuple[SkyRLModelSource, str | None]:
    if init is not None and init.artifact_type is SkyRLRun:
        record = read_record(init.path())
        uri = (record.result or {}).get("hf_model_uri") if record is not None else None
        if not isinstance(uri, str) or not uri:
            raise click.BadParameter("the initializer requires a completed HF export")
        return SkyRLPolicyModel(init), uri
    model = (
        preset.model
        if init is None
        else ArtifactHfModel(init, preset.model.tokenizer_uri, preset.model.tokenizer_revision)
    )
    if not preset.checkpoint_every:
        return model, None
    if model.relative_path is not None:
        return model, prefix_join(model.step.path(), model.relative_path)
    exports = discover_hf_checkpoints(model.step.path())
    if not exports:
        raise click.BadParameter("the initializer artifact has no HF export")
    return model, exports[-1]


def _refuse_judged(summary: dict, filename: str) -> None:
    if summary["grader_kind_counts"][filename].get(GraderKind.REMOTE_JUDGE, 0):
        raise click.BadParameter("Snowball selections must contain no AI-judge-graded tasks")


def run_spec(
    preset: presets.Preset,
    *,
    name: str,
    tasks: ArtifactStep | None = None,
    init: ArtifactStep | None = None,
    settings: tuple[str, ...] = (),
) -> SkyRLSpec:
    """Build the complete default recipe with explicit candidate inputs and settings."""
    config = _training_config(preset, settings)
    explicit = presets.override_document(settings)
    authored = config.to_skyrl()
    batch_changes = {
        field: value
        for key, field in presets.BATCH_SETTINGS.items()
        if (value := get_path(authored, key)) is not MISSING
    }
    if batch_changes:
        workload = preset.recipe.layout.workload.with_batches(batch_changes)
        preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, workload=workload)))
        micro_batch, in_flight = workload.batch_parameters(config)
        derived = {}
        if get_path(explicit, "trainer.micro_forward_batch_size_per_gpu") is MISSING:
            derived["micro_forward_batch_size_per_gpu"] = micro_batch
        if get_path(explicit, "trainer.rollout_buffer.max_in_flight") is MISSING:
            derived["rollout_buffer"] = {"max_in_flight": in_flight}
        config = config.merge(RecipePatch.from_document({"trainer": derived}))
    default, selection = presets.task_selections(preset, tasks)
    _refuse_judged(presets.selection_summary(selection), "train")
    chosen = preset.evaluations
    workload = preset.recipe.layout.workload
    preset = preset.with_updates(config.trainer.max_steps)
    model, initial_model_uri = _model_input(preset, init)
    validation_data = ()
    evaluation_data = []
    profiles = {}
    expected_tasks = 0
    for index, evaluation in enumerate(chosen):
        evaluation_tasks = evaluation.tasks(selection, default, preset.eval_lite)
        _refuse_judged(presets.selection_summary(evaluation_tasks), evaluation.file)
        profile_config, sources = presets.configure_evaluation(
            evaluation, config, evaluation_tasks, revision=preset.revision
        )
        if index == 0:
            config = profile_config
            validation_data = sources
            answer_count = presets.evaluation_answer_count(evaluation, evaluation_tasks)
            expected_tasks = answer_count // evaluation.answers
            config = config.merge(workload.evaluation_patch(answer_count))
        else:
            evaluation_data.append((evaluation.name, sources))
            profiles[evaluation.name] = EvaluationSampling(
                sampling_params=profile_config.generator.eval_sampling_params,
                n_samples_per_prompt=profile_config.generator.eval_n_samples_per_prompt,
                schedule=workload.named_schedule,
            )
    config = config.merge(
        RecipePatch(
            trainer=sections.Trainer(
                callbacks=workload.callbacks(
                    preset,
                    chosen[0] if chosen else None,
                    profiles,
                    expected_tasks,
                    initial_model_uri if preset.checkpoint_every else None,
                ),
            )
        )
    )
    plan = workload.plan
    return SkyRLSpec(
        name=name,
        version=preset.revision,
        config_yaml=yaml.safe_dump(SkyRLRecipe.from_document(config.to_skyrl()).to_skyrl(), sort_keys=False),
        model=model,
        train_data=workload.training_data(selection),
        validation_data=validation_data,
        evaluation_data=tuple(evaluation_data),
        topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=5 if preset.checkpoint_every else 2),
        seed=presets.SEED,
    )


def run_step(preset: presets.Preset, spec: SkyRLSpec, *, wandb_entity: str | None = None) -> ArtifactStep[SkyRLRun]:
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
            wandb_entity=wandb_entity or DEFAULT_WANDB_ENTITY,
            required_env_vars=(("WANDB_API_KEY", "WANDB_ENTITY") if wandb_entity else ("WANDB_API_KEY",)),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=preset.publishes,
    )
