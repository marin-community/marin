# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball preset composition and native execution resources."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, replace

import click
import yaml
from iris.rpc import job_pb2
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
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
from skyrlrecipe import ContextBudget, RecipePatch, SkyRLRecipe, sections
from skyrlrecipe.declared import (
    BestCheckpointCallback,
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    EvaluationSampling,
    HFModelSaveCallback,
)
from skyrlrecipe.documents import MISSING, get_path
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.task_curation.selection import TRAIN_FILENAME, GraderKind

RUN_TIMEOUT_HOURS = 72
DEFAULT_WANDB_ENTITY = "marin-community"


def _training_config(
    preset: presets.Preset,
    settings: tuple[str, ...],
) -> RecipePatch:
    recipe = preset.recipe
    plan = recipe.layout.role_plan
    geometry = asdict(recipe.layout.megatron)
    config = recipe.config.merge(
        RecipePatch(
            context_budget=ContextBudget(
                request_window_tokens=recipe.request_window_tokens,
                max_new_tokens_per_turn=(
                    recipe.fixed_reply_tokens if recipe.fixed_reply_tokens is not None else recipe.max_new_tokens
                ),
                max_turns=1,
            ),
            trainer=sections.Trainer(
                max_steps=preset.updates or 1,
                micro_forward_batch_size_per_gpu=plan.micro_train_batch_size_per_gpu,
                eval_interval=max(1, preset.updates) if preset.evaluations else -1,
                resume_mode="latest" if preset.focus == presets.Focus.TRAIN else "none",
                policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(**geometry)),
                ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**geometry)),
                rollout_buffer=sections.RolloutBuffer(
                    max_staleness_steps=recipe.staleness,
                    max_in_flight=plan.train_batch_size,
                ),
            ),
            generator=sections.Generator(
                sampling_params=sections.SamplingParams(ignore_eos=recipe.fixed_reply_tokens is not None),
            ),
        ),
        preset.patch,
    )
    return presets.apply_settings(config, settings, preset)


def _model_input(preset: presets.Preset, init: ArtifactStep | None) -> SkyRLModelSource:
    if init is None:
        return preset.model
    if init.artifact_type is SkyRLRun:
        record = read_record(init.path())
        if record is None:
            raise click.BadParameter("the initializer requires a completed published artifact record")
        result = record.result or {}
        if not result.get("hf_model_uri"):
            raise click.BadParameter("the initializer run has no HF export")
        return SkyRLPolicyModel(init)
    return ArtifactHfModel(init, preset.model.tokenizer_uri, preset.model.tokenizer_revision)


def _callbacks(
    preset: presets.Preset,
    evaluation: presets.Evaluation | None,
    profiles: dict[str, EvaluationSampling],
    expected_tasks: int,
    model: SkyRLModelSource,
) -> tuple[Callback, ...]:
    if preset.recipe.layout.role_plan.generate_only:
        return (EvaluationCallback(type="evaluation", additional_evaluations=profiles),) if profiles else ()
    callbacks: list[Callback] = []
    if preset.checkpoint_every:
        callbacks.append(CheckpointCallback(type="checkpoint", save_steps=preset.checkpoint_every))
    if evaluation is not None:
        callbacks.append(
            EvaluationCallback(
                type="evaluation",
                eval_steps=evaluation.every or preset.updates + 1,
                eval_before_train=evaluation.at_start,
                eval_on_train_end=not evaluation.every or preset.updates % evaluation.every != 0,
                **({"additional_evaluations": profiles} if profiles else {}),
            )
        )
    if preset.focus == presets.Focus.TRAIN:
        callbacks.extend(
            [
                BestCheckpointCallback(
                    type="best_checkpoint",
                    initial_model_uri=_initial_model_uri(model),
                    expected_tasks=expected_tasks,
                    minimum_scored_tasks=math.ceil(presets.MINIMUM_PANEL_COVERAGE * expected_tasks),
                ),
                HFModelSaveCallback(
                    type="hf_model_save",
                    save_steps=((preset.updates // preset.checkpoint_every) + 1) * preset.checkpoint_every,
                    save_on_train_end=True,
                ),
            ]
        )
    if preset.keep_weights and preset.focus != presets.Focus.TRAIN:
        callbacks.extend(
            (
                CheckpointCallback(type="checkpoint", save_steps=preset.updates + 1, save_on_train_end=True),
                HFModelSaveCallback(type="hf_model_save", save_steps=-1, save_on_train_end=True),
            )
        )
    return tuple(callbacks)


def _initial_model_uri(model: SkyRLModelSource) -> str:
    if isinstance(model, SkyRLPolicyModel):
        record = read_record(model.step.path())
        uri = (record.result or {}).get("hf_model_uri") if record is not None else None
        if not isinstance(uri, str) or not uri:
            raise click.BadParameter("the initializer requires a completed HF export")
        return uri
    if model.relative_path is not None:
        return prefix_join(model.step.path(), model.relative_path)
    exports = discover_hf_checkpoints(model.step.path())
    if not exports:
        raise click.BadParameter("the initializer artifact has no HF export")
    return exports[-1]


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
    initial = _training_config(preset, settings).to_skyrl()
    batch_changes = {
        field: value for key, field in presets.BATCH_SETTINGS.items() if (value := get_path(initial, key)) is not MISSING
    }
    if batch_changes:
        plan = replace(preset.recipe.layout.role_plan, **batch_changes)
        preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, role_plan=plan)))
    default = presets.default_selection(preset)
    selection = tasks or default
    _refuse_judged(presets.selection_summary(selection), "train")
    chosen = preset.evaluations
    generating = preset.recipe.layout.role_plan.generate_only
    config = _training_config(preset, settings).to_skyrl()
    preset = replace(preset, updates=config["trainer"]["max_steps"])
    model = _model_input(preset, init)
    validation_data = ()
    evaluation_data = []
    profiles = {}
    expected_tasks = 0
    for index, evaluation in enumerate(chosen):
        evaluation_tasks = evaluation.tasks(selection, default, preset.eval_lite)
        _refuse_judged(presets.selection_summary(evaluation_tasks), evaluation.file)
        profile_config = config if index == 0 else deepcopy(config)
        sources = presets.configure_evaluation(evaluation, profile_config, evaluation_tasks, revision=preset.revision)
        if index == 0:
            validation_data = sources
            answer_count = presets.evaluation_answer_count(evaluation, evaluation_tasks)
            expected_tasks = answer_count // evaluation.answers
            if generating:
                config["trajectory_runner"]["max_concurrent_tasks"] = answer_count
        else:
            evaluation_data.append((evaluation.name, sources))
            profiles[evaluation.name] = EvaluationSampling(
                sampling_params=EvalSamplingParams.model_validate_json(
                    json.dumps(profile_config["generator"]["eval_sampling_params"])
                ),
                n_samples_per_prompt=profile_config["generator"]["eval_n_samples_per_prompt"],
                **({"schedule": "final"} if not generating else {}),
            )
    config["trainer"]["callbacks"] = [
        callback.to_skyrl()
        for callback in _callbacks(preset, chosen[0] if chosen else None, profiles, expected_tasks, model)
    ]
    plan = preset.recipe.layout.role_plan
    return SkyRLSpec(
        name=name,
        version=preset.revision,
        config_yaml=yaml.safe_dump(SkyRLRecipe.from_document(config).to_skyrl(), sort_keys=False),
        model=model,
        train_data=(() if plan.generate_only else (ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),)),
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
        export_hf=preset.keep_weights,
    )
