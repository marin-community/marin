# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose model-owned RL presets and native execution resources."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path, PurePosixPath

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from finestore.rl.strata import MembershipState, SelectionTaskRow, read_selection_manifest
from finestore.schema import arrow_schema
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact, is_mutable_version, read_record, result_type_name
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    ResolvedModelLocator,
    SkyRLModelSource,
    SkyRLPolicyModel,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.documents import MISSING, get_path, merge_mappings, set_path
from skyrlrecipe.settings import parse_setting as typed_setting_value
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_presets import Evaluation, HfInitializer, PolicyInitializer, Preset, StoredInitializer
from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.selection import (
    SELECTION_PREFIX,
    SUMMARY_FILENAME,
    TASK_MANIFEST_FILENAME,
    TASK_SPEC_COLUMN,
    GraderKind,
)

REPO_ROOT = find_project_root(Path(__file__))
SELECTION_ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / SELECTION_PREFIX)


def override_document(settings: tuple[str, ...]) -> dict:
    """Canonical explicit user values for preset ownership and stored input identity."""
    document = {}
    for text in settings:
        key, _, raw = text.partition("=")
        patch = {}
        set_path(patch, key, typed_setting_value(RecipePatch, key, raw))
        document = merge_mappings(document, patch)
    return document


BATCH_SETTINGS = {
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
    "trainer.micro_train_batch_size_per_gpu": "micro_train_batch_size_per_gpu",
}

RUNNER_KEYS = frozenset(
    {
        "entrypoint",
        "trainer.resume_mode",
        "trainer.dump_eval_results",
        "trainer.placement",
        "trainer.callbacks",
        "trainer.ckpt_interval",
        "trainer.hf_save_interval",
        "generator.trajectory_retention",
        "generator.run_engines_locally",
    }
)
_GATED_KEYS = frozenset(
    {
        "trainer.max_steps",
        "trainer.logger",
        "trainer.project_name",
        "context_budget",
        "generator.sampling_params.ignore_eos",
        "generator.sampling_params.min_tokens",
    }
)


def selection_summary(selection: ArtifactStep) -> dict:
    return json.loads((StoragePath(selection.path()) / SUMMARY_FILENAME).read_text())


def evaluation_answer_count(evaluation: Evaluation, tasks: ArtifactStep) -> int:
    key = "training_rows" if not evaluation.held_out else f"{evaluation.file}_rows"
    return (evaluation.sample or selection_summary(tasks)[key]) * evaluation.answers


def _write_sample(selection: str, filename: str, count: int, seed: int, output: str) -> None:
    fs, key = filesystem_for(prefix_join(selection, f"{filename}.parquet"))
    with fs.open(key, "rb") as stream:
        source = pq.ParquetFile(stream)
        schema = source.schema_arrow
        _, rows = seeded_sample(
            (row for batch in source.iter_batches(batch_size=256) for row in batch.to_pylist()),
            size=count,
            key=lambda row: seeded_order(json.loads(row[TASK_SPEC_COLUMN])["id"], seed),
        )
    if len(rows) != count:
        raise ValueError(f"selection has only {len(rows)} eligible tasks; {count} are required")
    fs, key = filesystem_for(prefix_join(output, f"{filename}.parquet"))
    fs.makedirs(str(PurePosixPath(key).parent), exist_ok=True)
    with fs.open(key, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), stream)
    manifest = read_selection_manifest(selection)
    if manifest is not None:
        annotations = []
        for row in rows:
            document = json.loads(row[TASK_SPEC_COLUMN])
            task_id = document["id"]
            annotation = manifest.tasks.get(task_id) or SelectionTaskRow(
                task_id=task_id, **manifest.identity(task_id).strata.model_dump()
            )
            annotations.append(
                annotation.model_copy(
                    update={"task_digest": canonical_sha256(document), "state": MembershipState.ACTIVE}
                )
            )
        destination = StoragePath(output)
        with (destination / TASK_MANIFEST_FILENAME).open("wb") as stream:
            pq.write_table(
                pa.Table.from_pylist(
                    [row.model_dump(mode="json") for row in annotations], schema=arrow_schema(SelectionTaskRow)
                ),
                stream,
            )
        definition = manifest.definition.model_copy(
            update={
                "task_files": (f"{filename}.parquet",),
                "active_files": (f"{filename}.parquet",),
                "family_quotas": {},
                "sampling": f"seeded sample of {filename}",
            }
        )
        (destination / SUMMARY_FILENAME).write_text(json.dumps({"selection_strata": definition.model_dump(mode="json")}))


def sampled_tasks_step(
    selection: ArtifactStep, filename: str, count: int, seed: int, *, revision: str, namespace: str
) -> ArtifactStep:
    identity = fingerprint_hash(json.dumps((selection.name, selection.version, filename, count, seed)))
    return apply(
        f"{namespace}/sample-{identity}",
        _write_sample,
        version=revision,
        selection=selection,
        filename=filename,
        count=count,
        seed=seed,
        output=OUT,
    )


def apply_settings(config: RecipePatch, settings: tuple[str, ...], preset: Preset) -> RecipePatch:
    """Validate typed settings, then enforce the preset's input and measurement ownership."""
    try:
        updated = config.with_settings(settings)
        values = override_document(settings)
    except (ValueError, TypeError) as error:
        raise click.BadParameter(str(error)) from error

    def has(path: str) -> bool:
        return get_path(values, path) is not MISSING

    trainer = values.get("trainer", {})
    generator = values.get("generator", {})
    workload = preset.recipe.layout.workload
    setting_sections = workload.setting_sections
    derived = workload.plan.config_values().keys() - BATCH_SETTINGS.keys()
    owned_keys = RUNNER_KEYS | derived | preset.owned_keys
    if preset.gated:
        owned_keys |= _GATED_KEYS
    owners = {
        "use the preset's declared owned keys": any(has(path) for path in owned_keys),
        "choose network transport in Layout": any(key.startswith("NCCL_") for key in values.get("extra_env", {})),
        "evaluation requests are fixed by the preset": any(key.startswith("eval_") for key in (*trainer, *generator)),
        "select curated data with --tasks": has("data") and bool(values["data"].keys() - preset.data_settings),
        "use the preset's permitted setting sections": (
            setting_sections is not None and (bool(values.keys() - setting_sections) or has("generator.sampling_params"))
        ),
    }
    fix = next((message for message, owned in owners.items() if owned), None)
    if fix:
        raise click.BadParameter(f"setting is owned by the preset: {fix}")
    return updated


def task_selections(preset: Preset, tasks: ArtifactStep | None = None) -> tuple[ArtifactStep, ArtifactStep]:
    default = selection_artifact(preset.tasks)
    return default, tasks or default


def cluster_config_path(cluster: str) -> str:
    return f"lib/iris/config/{cluster}.yaml"


def artifact_prefix(cluster: str) -> str:
    config = yaml.safe_load((REPO_ROOT / cluster_config_path(cluster)).read_text())
    return config["defaults"]["task_env"]["MARIN_PREFIX"]


def selection_artifact(value: str) -> ArtifactStep:
    name, separator, version = value.rpartition("@")
    if not separator or is_mutable_version(version):
        raise click.BadParameter("use NAME@VERSION with an immutable task selection version")
    return ArtifactStep.adopt(f"{SELECTION_PREFIX}/{name}", version, str(StoragePath(SELECTION_ROOT) / name / version))


def published_model(value: str, *, preset: Preset) -> SkyRLModelSource:
    """Resolve a completed model artifact and its recorded tokenizer reference."""
    name, separator, version = value.rpartition("@")
    if not separator or is_mutable_version(version):
        raise click.BadParameter("use NAME@VERSION with an immutable model version")
    root = run_path(preset.recipe.layout.cluster, name, version)
    record = read_record(root)
    if (StoragePath(root) / RESOLVED_LAUNCH_FILENAME).exists() or (
        record is not None and record.result_type == result_type_name(SkyRLRun)
    ):
        return SkyRLPolicyModel(
            ArtifactStep.adopt(name, version, root, kind=SkyRLRun, config=record.config if record else None)
        )
    if record is None:
        raise click.BadParameter(f"no published artifact record for {name}@{version}")
    config = record.config or {}
    return ArtifactHfModel(
        ArtifactStep.adopt(name, version, record.source or root, kind=Artifact, config=record.config),
        tokenizer_uri=config.get("tokenizer_uri"),
        tokenizer_revision=config.get("tokenizer_revision"),
    )


def stored_initializer(source: SkyRLModelSource, *, preset: Preset) -> StoredInitializer:
    """Record a model producer's identity and selector without reading its output."""
    step = source.step
    if is_mutable_version(step.version):
        raise click.BadParameter("the initializer requires an immutable artifact version")
    values = dict(
        artifact_name=step.name,
        artifact_version=step.version,
        artifact_root=step.path(artifact_prefix(preset.recipe.layout.cluster)),
        fingerprint_payload=step.fingerprint_payload(),
    )
    if isinstance(source, SkyRLPolicyModel):
        return PolicyInitializer(**values)
    return HfInitializer(
        **values,
        relative_path=source.relative_path,
        tokenizer_uri=source.tokenizer_uri,
        tokenizer_revision=source.tokenizer_revision,
    )


def model_from_reference(source: StoredInitializer, *, resolved: ResolvedModelLocator | None = None) -> SkyRLModelSource:
    """Compose a saved model identity at its recorded artifact or resolved export."""
    expected = fingerprint_hash(source.fingerprint_payload)
    identity = f"{source.artifact_name}@{source.artifact_version}:{expected}"
    if resolved is not None and resolved.identity != identity:
        raise ValueError("recorded initializer identity differs from the saved input")
    kind = SkyRLRun if isinstance(source, PolicyInitializer) else Artifact
    payload = json.loads(source.fingerprint_payload)
    step = ArtifactStep(
        name=source.artifact_name,
        version=source.artifact_version,
        artifact_type=kind,
        build_config=lambda _ctx: payload,
        run=lambda _config: kind.raw_load(source.artifact_root),
        override_path=resolved.uri if resolved is not None else source.artifact_root,
        expected_fingerprint=expected,
    )
    if isinstance(source, PolicyInitializer):
        if resolved is None:
            return SkyRLPolicyModel(step)
        return ArtifactHfModel(step, resolved.tokenizer_uri, resolved.tokenizer_revision, relative_path="")
    return ArtifactHfModel(
        step,
        source.tokenizer_uri,
        source.tokenizer_revision,
        relative_path="" if resolved is not None else source.relative_path,
    )


def run_path(cluster: str, name: str, version: str) -> str:
    return str(StoragePath(artifact_prefix(cluster)) / name / version)


def utc_millis(value: str) -> int:
    """Parse a native receipt's UTC timestamp for duration and dashboard windows."""
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


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
    return apply_settings(config, settings, preset)


def _refuse_judged(summary: dict, filename: str) -> None:
    if summary["grader_kind_counts"][filename].get(GraderKind.REMOTE_JUDGE, 0):
        raise click.BadParameter("task selections must contain no AI-judge-graded tasks")


def run_spec(
    preset: presets.Preset,
    *,
    name: str,
    tasks: ArtifactStep | None = None,
    init: SkyRLModelSource | None = None,
    settings: tuple[str, ...] = (),
    model_name: str | None = None,
) -> SkyRLSpec:
    """Build the complete default recipe with explicit candidate inputs and settings."""
    config = _training_config(preset, settings)
    explicit = override_document(settings)
    authored = config.to_skyrl()
    batch_changes = {
        field: value for key, field in BATCH_SETTINGS.items() if (value := get_path(authored, key)) is not MISSING
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
    default, selection = task_selections(preset, tasks)
    _refuse_judged(selection_summary(selection), "train")
    chosen = preset.evaluations
    workload = preset.recipe.layout.workload
    preset = preset.with_updates(config.trainer.max_steps)
    model = preset.model if init is None else init
    selects_best = preset.checkpoint_every or (preset.export_hf and config.trainer.hf_export_checkpoint == "best")
    initial_model_uri = model.step.path(artifact_prefix(preset.recipe.layout.cluster)) if selects_best else None
    validation_data = ()
    evaluation_data = []
    profiles = {}
    expected_tasks = 0
    for index, evaluation in enumerate(chosen):
        evaluation_tasks = evaluation_source(evaluation, selection, default, preset.held_out_tasks)
        _refuse_judged(selection_summary(evaluation_tasks), evaluation.file)
        tasks_for_evaluation = evaluation_tasks
        if evaluation.sample is not None:
            tasks_for_evaluation = sampled_tasks_step(
                evaluation_tasks,
                evaluation.file,
                evaluation.sample,
                preset.seed,
                revision=preset.revision,
                namespace=preset.sample_namespace,
            )
        sources = (ArtifactDataSource(tasks_for_evaluation, relative_path=f"{evaluation.file}.parquet"),)
        sampling = evaluation.profile.sampling_params
        if index == 0:
            validation_data = sources
            answer_count = evaluation_answer_count(evaluation, evaluation_tasks)
            expected_tasks = answer_count // evaluation.answers
            config = config.merge(
                RecipePatch(
                    trainer=sections.Trainer(eval_num_prompts=None, dump_eval_results=True),
                    generator=sections.Generator(
                        eval_n_samples_per_prompt=evaluation.answers, eval_sampling_params=sampling
                    ),
                )
            ).merge(workload.evaluation_patch(answer_count))
        else:
            evaluation_data.append((evaluation.name, sources))
            profiles[evaluation.name] = EvaluationProfile(
                sampling_params=sampling,
                n_samples_per_prompt=evaluation.answers,
                schedule=workload.additional_schedule,
            )
    config = config.merge(
        RecipePatch(
            trainer=sections.Trainer(
                callbacks=workload.callbacks(
                    preset,
                    chosen[0] if chosen else None,
                    profiles,
                    expected_tasks,
                    initial_model_uri,
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
        model_name=model_name,
        train_data=workload.training_data(selection),
        validation_data=validation_data,
        evaluation_data=tuple(evaluation_data),
        topology=SkyRLTopology(
            num_nodes=plan.num_nodes,
            gpus_per_node=plan.gpus_per_node,
            gpu_variant=preset.recipe.layout.gpu_variant,
            role_plan=plan,
        ),
        retention=preset.retention,
        seed=preset.seed,
    )


def run_step(
    preset: presets.Preset, spec: SkyRLSpec, *, wandb_entity: str | None = None, forward_env: tuple[str, ...] = ()
) -> ArtifactStep[SkyRLRun]:
    """Bind an RL specification to its declared execution resources."""
    layout = preset.recipe.layout
    required_env = ("WANDB_API_KEY", "WANDB_ENTITY") if wandb_entity else ("WANDB_API_KEY",)
    return skyrl_step(
        spec,
        IrisSkyRLExecution(
            cluster=layout.cluster,
            cluster_config=cluster_config_path(layout.cluster),
            cpu=layout.host_cpu,
            memory=layout.host_memory,
            disk=layout.host_disk,
            priority=layout.priority,
            max_retries=0,
            target_cluster=layout.cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=layout.coordinator_timeout_hours,
            job_timeout_seconds=layout.job_timeout_seconds,
            wandb_entity=wandb_entity or preset.wandb_entity,
            required_env_vars=tuple(dict.fromkeys((*required_env, *forward_env))),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=preset.export_hf,
    )


def evaluation_source(
    evaluation: Evaluation, selection: ArtifactStep, default: ArtifactStep, held_out_tasks: str | None
) -> ArtifactStep:
    if evaluation.source == "default":
        return default
    if evaluation.source == "tasks":
        return selection
    assert held_out_tasks is not None
    return selection_artifact(held_out_tasks)
