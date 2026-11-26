# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose model-owned RL presets and native execution resources."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path, PurePosixPath

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from finestore.rl.strata import MembershipState, SelectionTaskRow, read_selection_manifest
from finestore.schema import arrow_schema
from marin.evaluation.eval_env import DAYTONA_EVAL_API_KEY_ENV
from marin.execution.artifact import (
    RECORD_FILENAME,
    Artifact,
    is_mutable_version,
    read_record,
    result_type_name,
    validate_path_segment,
)
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply, artifact_identity
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
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME, read_skyrl_launch_config
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rolloutengine.agents.pi.agent import PI_HARNESS
from rolloutengine.agents.terminus.agent import TERMINUS_HARNESS
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.documents import MISSING, get_path, merge_mappings, set_path
from skyrlrecipe.settings import parse_setting as typed_setting_value
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_checkpoint_eval import (
    CHECKPOINT_DAYTONA_TTL,
    CHECKPOINT_HARNESS_OPTIONS,
    CHECKPOINT_PROFILE,
    CHECKPOINT_REPLY_TOKENS,
    CHECKPOINT_TASKS_FILENAME,
    checkpoint_tasks,
)
from experiments.post_training.rl_presets import Evaluation, HfInitializer, PolicyInitializer, Preset, StoredInitializer
from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.selection import (
    SELECTION_PREFIX,
    SUMMARY_FILENAME,
    TASK_MANIFEST_FILENAME,
    TASK_SPEC_COLUMN,
    GraderKind,
)
from experiments.post_training.task_curation.swesmith_training import SWESMITH_DATASET, native_swesmith_selection

REPO_ROOT = find_project_root(Path(__file__))
SELECTION_ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / SELECTION_PREFIX)
# Version of the stored run artifact. Preset versions are composed configuration hashes.
RUN_ARTIFACT_VERSION = "2026.11.25"


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

_DERIVED_RECIPE_KEYS = frozenset(
    {
        "entrypoint",
        "trainer.resume_mode",
        "trainer.dump_eval_results",
        "trainer.placement",
        "trainer.callbacks",
        "trainer.ckpt_interval",
        "trainer.checkpoint_retention",
        "trainer.checkpoint_keep_every",
        "trainer.hf_save_interval",
        "trainer.hf_export_checkpoint",
        "trainer.rollout_buffer.groups_per_family",
        "generator.enable_http_endpoint",
        "generator.num_inference_engines",
        "generator.inference_engine_tensor_parallel_size",
        "generator.inference_engine_pipeline_parallel_size",
        "generator.inference_engine_data_parallel_size",
        "generator.inference_engine_expert_parallel_size",
    }
)
RUNNER_KEYS = _DERIVED_RECIPE_KEYS | {"generator.trajectory_retention", "generator.run_engines_locally"}
_GATED_KEYS = frozenset(
    {
        "trainer.max_steps",
        "trainer.hf_hub_repo_id",
        "trainer.logger",
        "trainer.project_name",
        "context_budget",
        "generator.sampling_params.ignore_eos",
        "generator.sampling_params.min_tokens",
    }
)
_DSPARK_PROPOSALS = 4


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


def sampled_tasks_step(selection: ArtifactStep, filename: str, count: int, seed: int, *, namespace: str) -> ArtifactStep:
    identity = fingerprint_hash(json.dumps((selection.name, selection.version, filename, count, seed)))
    return apply(
        f"{namespace}/sample-{identity}",
        _write_sample,
        version=selection.version,
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
        "network transport is fixed by the preset recipe": any(
            key.startswith("NCCL_") for key in values.get("extra_env", {})
        ),
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


def run_prefix(preset: Preset) -> str:
    cluster = preset.recipe.layout.cluster
    prefix = artifact_prefix(cluster)
    if preset.output_ttl_days is None:
        return prefix
    output = os.environ.get("MARIN_PREFIX") or marin_temp_bucket(preset.output_ttl_days, source_prefix=prefix)
    scheme = StoragePath(prefix).scheme
    output_scheme = StoragePath(output).scheme
    if scheme and output_scheme != scheme:
        raise click.BadParameter(f"run output for {cluster} must use {scheme} storage, got {output_scheme or 'local'}")
    return output


def selection_artifact(value: str) -> ArtifactStep:
    name, separator, version = value.rpartition("@")
    if not separator or is_mutable_version(version):
        raise click.BadParameter("use NAME@VERSION with an immutable task selection version")
    return ArtifactStep.adopt(
        f"{SELECTION_PREFIX}/{name}", version, str(StoragePath(SELECTION_ROOT) / name / version), read_only=True
    )


def published_model(value: str, *, preset: Preset) -> SkyRLModelSource:
    """Resolve a completed model artifact and its recorded tokenizer reference."""
    name, separator, version = value.rpartition("@")
    if not separator or is_mutable_version(version):
        raise click.BadParameter("use NAME@VERSION with an immutable model version")
    root = _source_run_path(preset, name, version, RESOLVED_LAUNCH_FILENAME, RECORD_FILENAME)
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


def recovery_checkpoint(value: str, *, preset: Preset) -> presets.StoredResume:
    """Pin the newest retained recovery checkpoint and starting model from a stopped run."""
    name, separator, version = value.rpartition("@")
    if not name or not separator or is_mutable_version(version):
        raise click.BadParameter("use NAME@VERSION with an immutable run version")
    try:
        validate_path_segment("name", name)
        validate_path_segment("version", version)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    output = _source_run_path(preset, name, version, RESOLVED_LAUNCH_FILENAME)
    try:
        config = read_skyrl_launch_config(output)
        root = StoragePath(config["artifacts"]["checkpoint_root"])
        step = int((root / "latest_ckpt_global_step.txt").read_text().strip())
        checkpoint = root / f"global_step_{step}"
        committed = step > 0 and (checkpoint / "trainer_state.pt").exists()
    except (FileNotFoundError, KeyError, ValueError):
        committed = False
    if not committed:
        raise click.BadParameter(
            f"no committed native checkpoint for {name}@{version}. "
            "Checkpoint availability follows the source run's retention period."
        )
    recorded = config.get("inputs", {}).get("model")
    model = (
        ResolvedModelLocator(**{field.name: recorded[field.name] for field in fields(ResolvedModelLocator)})
        if recorded
        else None
    )
    return presets.StoredResume(
        artifact_name=name, artifact_version=version, step=step, checkpoint=str(checkpoint), model=model
    )


def stored_initializer(source: SkyRLModelSource, *, preset: Preset) -> StoredInitializer:
    """Record a model producer's identity and selector without reading its output."""
    step = source.step
    if is_mutable_version(step.version):
        raise click.BadParameter("the initializer requires an immutable artifact version")
    values = dict(
        artifact_name=step.name,
        artifact_version=step.version,
        artifact_root=step.path(run_prefix(preset)),
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


def run_path(preset: Preset, name: str, version: str) -> str:
    return str(StoragePath(run_prefix(preset)) / name / version)


def _source_run_path(preset: Preset, name: str, version: str, *record_names: str) -> str:
    output = StoragePath(run_path(preset, name, version))
    ttl = preset.output_ttl_days
    if ttl is None or any((output / record).exists() for record in record_names):
        return str(output)
    durable = artifact_prefix(preset.recipe.layout.cluster)
    temporary = marin_temp_bucket(ttl, source_prefix=durable)
    for prefix in dict.fromkeys((durable, temporary)):
        candidate = StoragePath(prefix) / name / version
        if candidate != output and candidate.scheme == output.scheme:
            if any((candidate / record).exists() for record in record_names):
                return str(candidate)
    return str(output)


def utc_millis(value: str) -> int:
    """Parse a native receipt's UTC timestamp for duration and dashboard windows."""
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _recipe_config_for_model(preset: presets.Preset, model: SkyRLModelSource | ResolvedModelLocator) -> RecipePatch:
    workload = preset.recipe.layout.workload
    if isinstance(workload, presets.Training) and workload.plan.colocate_all:
        return preset.recipe.config
    draft = preset.speculative_decoding_model
    if draft is not None and (
        model == preset.model
        or (isinstance(model, ResolvedModelLocator) and model.identity == artifact_identity(preset.model.step))
    ):
        decoding = sections.SpeculativeDecoding(method="dspark", model=draft, num_speculative_tokens=_DSPARK_PROPOSALS)
        return preset.recipe.config.merge(RecipePatch(generator=sections.Generator(speculative_decoding=decoding)))
    return preset.recipe.config


def _training_config(
    preset: presets.Preset,
    settings: tuple[str, ...],
    model: SkyRLModelSource | ResolvedModelLocator,
) -> RecipePatch:
    recipe = preset.recipe
    recipe_config = _recipe_config_for_model(preset, model)
    authored = recipe_config.to_skyrl()
    if owned := sorted(key for key in _DERIVED_RECIPE_KEYS if get_path(authored, key) is not MISSING):
        raise ValueError(f"recipe settings are derived by the runner: {', '.join(owned)}")
    micro_batch, _ = recipe.layout.workload.batch_parameters(recipe_config)
    geometry = asdict(recipe.layout.megatron)
    budget = recipe_config.context_budget
    assert budget is not None
    if recipe.fixed_reply_tokens is not None:
        budget = budget.replace(max_new_tokens_per_turn=recipe.fixed_reply_tokens)
    trainer = sections.Trainer(
        max_steps=preset.updates or 1,
        micro_forward_batch_size_per_gpu=micro_batch,
        eval_interval=max(1, preset.updates) if preset.evaluations else -1,
        resume_mode="latest" if preset.checkpoints != "none" else "none",
        ckpt_interval=0,
        hf_save_interval=-1,
        policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(**geometry)),
        ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**geometry)),
    )
    if preset.checkpoints != "none":
        trainer = trainer.replace(checkpoint_retention=preset.checkpoints)
    if preset.keep_every is not None:
        trainer = trainer.replace(checkpoint_keep_every=preset.keep_every)
    if preset.export == "best":
        trainer = trainer.replace(hf_export_checkpoint="best")
    if recipe.training_plan is not None:
        training_geometry = {}
        for key, value in recipe.training_plan.config_values().items():
            if key.startswith("trainer."):
                set_path(training_geometry, key, value)
        recipe_config = recipe_config.merge(RecipePatch.from_document(training_geometry))
    config = recipe_config.merge(
        RecipePatch(
            context_budget=budget,
            trainer=trainer,
            generator=sections.Generator(
                sampling_params=sections.SamplingParams(ignore_eos=recipe.fixed_reply_tokens is not None),
            ),
        ),
    )
    config = apply_settings(config, settings, preset)
    if settings:
        try:
            presets.validate_agentic_recipe(config)
        except ValueError as error:
            raise click.BadParameter(str(error)) from error
    return config


def _refuse_judged(summary: dict, filename: str) -> None:
    if summary["grader_kind_counts"][filename].get(GraderKind.REMOTE_JUDGE, 0):
        raise click.BadParameter("task selections must contain no AI-judge-graded tasks")


def _training_selection(selection: ArtifactStep, summary: dict, authored: dict) -> ArtifactStep:
    legacy = set(summary.get("harness_graded_datasets", ()))
    if not legacy:
        return selection
    if unsupported := legacy - {SWESMITH_DATASET}:
        raise click.BadParameter(f"Unsupported training datasets with harness graders: {', '.join(sorted(unsupported))}")
    spec_path = "environment.task_sessions.task_spec"
    harness = get_path(authored, f"{spec_path}.harness_by_dataset.{SWESMITH_DATASET}")
    machine = get_path(authored, f"{spec_path}.task_machine_by_dataset.{SWESMITH_DATASET}")
    if harness != PI_HARNESS or machine is MISSING:
        raise click.BadParameter("SWESmith training requires the Pi harness and a task machine")
    return native_swesmith_selection(selection)


def _checkpoint_session_patch(config: RecipePatch) -> RecipePatch:
    budget = config.context_budget
    assert budget is not None
    if CHECKPOINT_REPLY_TOKENS >= budget.effective_serving_window_tokens:
        raise click.BadParameter("Checkpoint reply tokens must leave room for input in the serving window")
    daytona = {
        "eval_api_key_env": DAYTONA_EVAL_API_KEY_ENV,
        "eval_ttl_minutes": CHECKPOINT_DAYTONA_TTL,
        "eval_image_build": "declarative",
    }
    if get_path(config.to_skyrl(), "environment.task_sessions.daytona.api_key_env") is MISSING:
        daytona.update(api_key_env=DAYTONA_EVAL_API_KEY_ENV, ttl_minutes=CHECKPOINT_DAYTONA_TTL)
    return RecipePatch(
        environment=sections.Environment(
            task_sessions={
                "daytona": daytona,
                "agent_harnesses": {TERMINUS_HARNESS: dict(CHECKPOINT_HARNESS_OPTIONS)},
            }
        )
    )


def run_spec(
    preset: presets.Preset,
    *,
    name: str,
    tasks: ArtifactStep | None = None,
    init: SkyRLModelSource | None = None,
    resume: presets.StoredResume | None = None,
    settings: tuple[str, ...] = (),
    model_name: str | None = None,
) -> SkyRLSpec:
    """Build the complete default recipe with explicit candidate inputs and settings."""
    model = (resume.model if resume is not None else None) or init or preset.model
    config = _training_config(preset, settings, model)
    if config.environment.env_class != "preference_pair" or config.trainer.evaluation_runner != "training":
        workers = config.trajectory_runner.rollout_workers
        required_cpus = workers.num_workers * workers.cpus_per_worker
        driver_cpus = preset.recipe.layout.host_cpu
        if required_cpus > driver_cpus:
            raise click.BadParameter(
                f"trajectory_runner.rollout_workers requests {required_cpus} CPUs "
                f"({workers.num_workers} workers x {workers.cpus_per_worker}); "
                f"the driver task's Iris CPU request is {driver_cpus}."
            )
    explicit = override_document(settings)
    authored = config.to_skyrl()
    batch_changes = {
        field: value for key, field in BATCH_SETTINGS.items() if (value := get_path(authored, key)) is not MISSING
    }
    workload = preset.recipe.layout.workload
    if batch_changes:
        workload = workload.with_batches(batch_changes)
        preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, workload=workload)))
    micro_batch, in_flight = workload.batch_parameters(config)
    derived = {}
    if batch_changes and get_path(explicit, "trainer.micro_forward_batch_size_per_gpu") is MISSING:
        derived["micro_forward_batch_size_per_gpu"] = micro_batch
    if get_path(authored, "trainer.rollout_buffer.max_in_flight") is MISSING:
        derived["rollout_buffer"] = {"max_in_flight": in_flight}
    config = config.merge(RecipePatch.from_document({"trainer": derived}))
    default, selection = task_selections(preset, tasks)
    summary = selection_summary(selection)
    _refuse_judged(summary, "train")
    training_selection = _training_selection(selection, summary, authored)
    chosen = preset.evaluations
    if preset.batch_mix is not None:
        assert isinstance(workload, presets.Training)
        try:
            groups = preset.batch_mix.groups(
                workload.plan.train_batch_size, set(summary.get("training_family_counts", {}))
            )
        except ValueError as error:
            raise click.BadParameter(str(error)) from error
        config = config.merge(
            RecipePatch(trainer=sections.Trainer(rollout_buffer=sections.RolloutBuffer(groups_per_family=groups)))
        )
    preset = preset.with_updates(config.trainer.max_steps)
    initial_model_uri = None
    if preset.checkpoints == "best":
        prefix = run_prefix(preset)
        initial_model_uri = model.uri if isinstance(model, ResolvedModelLocator) else model.step.path(prefix)
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
    if preset.checkpoint_evaluation:
        config = config.merge(_checkpoint_session_patch(config))
        for cohort in preset.checkpoint_evaluation:
            evaluation_data.append(
                (cohort.name, (ArtifactDataSource(checkpoint_tasks(cohort), relative_path=CHECKPOINT_TASKS_FILENAME),))
            )
            profiles[cohort.name] = CHECKPOINT_PROFILE
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
        version=RUN_ARTIFACT_VERSION,
        config_yaml=yaml.safe_dump(SkyRLRecipe.from_document(config.to_skyrl()).to_skyrl(), sort_keys=False),
        model=model,
        model_name=model_name,
        train_data=workload.training_data(training_selection),
        validation_data=validation_data,
        evaluation_data=tuple(evaluation_data),
        topology=SkyRLTopology(
            num_nodes=plan.num_nodes,
            gpus_per_node=plan.gpus_per_node,
            gpu_variant=preset.recipe.layout.gpu_variant,
            role_plan=plan,
        ),
        retention=(
            replace(preset.retention, checkpoints_in_run_output=True)
            if preset.output_ttl_days is not None
            else preset.retention
        ),
        seed=preset.seed,
        starting_checkpoint=(
            ArtifactStep.adopt(
                f"{resume.artifact_name}/resume/step-{resume.step}", resume.artifact_version, resume.checkpoint
            )
            if resume is not None
            else None
        ),
    )


def _recipe_env(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_env") and isinstance(child, str):
                yield child
            else:
                yield from _recipe_env(child)
    elif isinstance(value, list):
        for child in value:
            yield from _recipe_env(child)


def run_step(
    preset: presets.Preset, spec: SkyRLSpec, *, wandb_entity: str | None = None, forward_env: tuple[str, ...] = ()
) -> ArtifactStep[SkyRLRun]:
    """Bind an RL specification to its declared execution resources."""
    layout = preset.recipe.layout
    config = yaml.safe_load(spec.config_yaml)
    required_env = ("WANDB_API_KEY", "WANDB_ENTITY") if wandb_entity else ("WANDB_API_KEY",)
    if preset.output_ttl_days is not None:
        required_env += ("MARIN_PREFIX",)
    required_env += tuple(_recipe_env(config))
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
            container_profile=layout.container_profile,
            role_topology=layout.role_topology,
        ),
        export_hf=preset.export != "none",
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
