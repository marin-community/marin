# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball preset composition and native execution resources."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, replace

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    BEST_CHECKPOINT_CALLBACK,
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
    set_config_value,
    skyrl_step,
)

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.task_curation.selection import TRAIN_FILENAME, GraderKind

RUN_TIMEOUT_HOURS = 72
DEFAULT_WANDB_ENTITY = "marin-community"


def _training_config(
    preset: presets.Preset,
    settings: tuple[presets.Setting, ...],
) -> dict:
    recipe = preset.recipe
    plan = recipe.layout.role_plan
    config = {
        "context_budget": {
            "request_window_tokens": recipe.request_window_tokens,
            "max_new_tokens_per_turn": (
                recipe.fixed_reply_tokens if recipe.fixed_reply_tokens is not None else recipe.max_new_tokens
            ),
        },
        "trainer": {
            "max_steps": preset.updates or 1,
            "micro_forward_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
            "eval_interval": max(1, preset.updates) if preset.evaluations else -1,
            "resume_mode": "latest" if preset.focus == presets.Focus.TRAIN else "none",
            "seed": presets.SEED,
            "algorithm": deepcopy(dict(recipe.estimator)),
            "policy": {"optimizer_config": deepcopy(dict(recipe.optimizer))},
            "rollout_buffer": {"max_staleness_steps": recipe.staleness, "max_in_flight": plan.train_batch_size},
        },
        "generator": {"sampling_params": {"ignore_eos": recipe.fixed_reply_tokens is not None}},
    }
    geometry = asdict(recipe.layout.megatron)
    config["trainer"]["policy"]["megatron_config"] = geometry
    config["trainer"]["ref"] = {"megatron_config": deepcopy(geometry)}
    for text in preset.settings:
        setting = presets.parse_setting(text)
        set_config_value(config, setting.key, setting.value)
    for setting in settings:
        presets.apply_setting(config, setting, preset)
    return config


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
    profiles: dict,
    expected_tasks: int,
) -> list[dict]:
    if preset.recipe.layout.role_plan.generate_only:
        return [{"type": "evaluation", "additional_evaluations": profiles}] if profiles else []
    callbacks = []
    if preset.checkpoint_every:
        callbacks.append({"type": "checkpoint", "save_steps": preset.checkpoint_every})
    if evaluation is not None:
        for callback in presets.evaluation_callbacks(evaluation, updates=preset.updates):
            if profiles and callback["type"] == "evaluation":
                callback["additional_evaluations"] = profiles
            callbacks.append(callback)
    if preset.focus == presets.Focus.TRAIN:
        callbacks.extend(
            [
                {
                    "type": BEST_CHECKPOINT_CALLBACK,
                    "expected_tasks": expected_tasks,
                    "minimum_scored_tasks": math.ceil(presets.MINIMUM_PANEL_COVERAGE * expected_tasks),
                },
                {
                    "type": "hf_model_save",
                    "save_steps": ((preset.updates // preset.checkpoint_every) + 1) * preset.checkpoint_every,
                    "save_on_train_end": True,
                },
            ]
        )
    if preset.keep_weights and preset.focus != presets.Focus.TRAIN:
        callbacks.extend(
            (
                {"type": "checkpoint", "save_steps": preset.updates + 1, "save_on_train_end": True},
                {"type": "hf_model_save", "save_steps": -1, "save_on_train_end": True},
            )
        )
    return callbacks


def _refuse_judged(summary: dict, filename: str) -> None:
    if summary["grader_kind_counts"][filename].get(GraderKind.REMOTE_JUDGE, 0):
        raise click.BadParameter("Snowball selections must contain no AI-judge-graded tasks")


def run_spec(
    preset: presets.Preset,
    *,
    name: str,
    tasks: ArtifactStep | None = None,
    init: ArtifactStep | None = None,
    settings: tuple[presets.Setting, ...] = (),
) -> SkyRLSpec:
    """Build the complete default recipe with explicit candidate inputs and settings."""
    settings = tuple(leaf for setting in settings for leaf in presets.setting_leaves(setting))
    defaults = tuple(leaf for text in preset.settings for leaf in presets.setting_leaves(presets.parse_setting(text)))
    batch_changes = {
        field: setting.value for setting in (*defaults, *settings) if (field := presets.BATCH_SETTINGS.get(setting.key))
    }
    if batch_changes:
        plan = replace(preset.recipe.layout.role_plan, **batch_changes)
        preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, role_plan=plan)))
    default = presets.default_selection(preset)
    selection = tasks or default
    _refuse_judged(presets.selection_summary(selection), "train")
    chosen = preset.evaluations
    generating = preset.recipe.layout.role_plan.generate_only
    config = _training_config(preset, settings)
    preset = replace(preset, updates=config["trainer"]["max_steps"])
    model = _model_input(preset, init)
    validation_data = ()
    evaluation_data = []
    profiles = {}
    expected_tasks = 0
    for index, evaluation in enumerate(chosen):
        if index and (evaluation.every is not None or evaluation.at_start):
            raise click.BadParameter(f"non-primary evaluation {evaluation.name} accepts neither every nor at_start")
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
            profiles[evaluation.name] = {
                "sampling_params": profile_config["generator"]["eval_sampling_params"],
                "n_samples_per_prompt": profile_config["generator"]["eval_n_samples_per_prompt"],
            }
            if not generating:
                profiles[evaluation.name]["schedule"] = "final"
    config["trainer"]["callbacks"] = _callbacks(preset, chosen[0] if chosen else None, profiles, expected_tasks)
    plan = preset.recipe.layout.role_plan
    return SkyRLSpec(
        name=name,
        version=preset.revision,
        config_yaml=yaml.safe_dump(config, sort_keys=False),
        model=model,
        train_data=(() if plan.generate_only else (ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),)),
        validation_data=validation_data,
        evaluation_data=tuple(evaluation_data),
        topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=5 if preset.checkpoint_every else 2),
        seed=config["trainer"]["seed"],
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
