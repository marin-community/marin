# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from collections.abc import Callable
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec import get_filesystem_class, register_implementation
from fsspec.implementations.memory import MemoryFileSystem
from marin.execution.artifact import write_artifact
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import iceball_micro, rl, rl_presets, skyrl_port_smoke, snowball_online_eagle
from experiments.post_training import rl_recipe as recipe
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.cat_count_canary import launcher as canary
from experiments.post_training.curriculum_rl import launch as curriculum
from experiments.post_training.mismatch_probe import launch as mismatch
from experiments.post_training.skyrl_test_support import selection_summary
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, SUMMARY_FILENAME, TRAIN_FILENAME
from experiments.post_training.task_sessions import validation

VERSION = "2026.10.24"


class _MemoryObjectStore(MemoryFileSystem):
    protocol = "s3"
    store: ClassVar[dict[str, Any]] = {}
    pseudo_dirs: ClassVar[list[str]] = [""]

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(str(path).removeprefix("s3://"))

    def glob(self, path, **kwargs):
        return [match.lstrip("/") for match in super().glob(path, **kwargs)]


def _curriculum_step(scale, policy, arm) -> ArtifactStep[SkyRLRun]:
    return curriculum.build_arms(specs=(arm,), scale=scale, policy=policy, version=VERSION)[arm.name].rl


def _snowball_step(preset: rl_presets.Preset, settings: tuple[str, ...] = ()) -> ArtifactStep[SkyRLRun]:
    eval_lite = recipe.selection_artifact(preset.eval_lite)
    selection = recipe.selection_artifact(preset.tasks)
    for source, full_rows in ((selection, 910), (eval_lite, 20)):
        summary = selection_summary(full_rows)
        summary["recipe"]["exclusions"] = [
            {
                "uri": str(StoragePath(eval_lite.path()) / FULL_VALIDATION_FILENAME),
                "version": eval_lite.version,
            }
        ]
        path = StoragePath(source.path()) / SUMMARY_FILENAME
        path.parent.mkdirs()
        if source is selection and preset.batch_mix is not None:
            counts = dict.fromkeys(preset.batch_mix.shares, 256)
            summary["training_family_counts"] = counts
            families = [family for family, count in counts.items() for _ in range(count)]
            with (path.parent / TRAIN_FILENAME).open("wb") as stream:
                pq.write_table(pa.table({"family": families}), stream)
        path.write_text(json.dumps(summary))
    return rl.build_run(
        preset,
        tasks=selection if preset.tasks_input == "required" else None,
        settings=settings,
        table=snowball_model.SNOWBALL,
    ).runs[0]


def _iceball_step() -> ArtifactStep[SkyRLRun]:
    return iceball_micro.build_workflow(version=VERSION).rl


def _probe_step(artifact_root: Path, **settings: Any) -> ArtifactStep[SkyRLRun]:
    # The build-options wrapper displays/runs a graph. Its wrapped Click build body returns that graph.
    builder = inspect.unwrap(mismatch.main.callback)
    return builder(
        model_uri=str(artifact_root / "iceball-sft"),
        data_uri=str(artifact_root / "gsm8k"),
        input_version=VERSION,
        **settings,
    )


def _port_smoke_step(stage: str) -> ArtifactStep[SkyRLRun]:
    return getattr(skyrl_port_smoke.build_smoke(VERSION), stage)


def _one_factor(base, axes):
    return [base, *(base | {name: value} for name, choices in axes.items() for value in choices if value != base[name])]


def launch_builders(
    artifact_root: Path,
) -> dict[str, Callable[[], ArtifactStep[SkyRLRun]]]:
    """Enumerate current public presets and representative document-changing CLI options."""
    builders = {}
    for scale, policy, arm in product(curriculum.SCALES, curriculum.POLICIES.values(), curriculum.ARMS.values()):
        name = f"curriculum/{scale}/{policy.label}/{arm.name}"
        if (policy is curriculum.SNOWBALL_POLICY) != scale.startswith("snowball-"):
            continue
        builders[name] = partial(_curriculum_step, scale, policy, arm)
    for label, preset in snowball_model.SNOWBALL.by_name.items():
        builders[f"snowball/{label}"] = partial(_snowball_step, preset)
    builders["snowball/settings/optimizer"] = partial(
        _snowball_step, snowball_model.SNOWBALL.by_name["learning-medium"], ("trainer.policy.optimizer_config.lr=2e-6",)
    )
    builders["snowball/settings/context-budget"] = partial(
        _snowball_step, snowball_model.SNOWBALL.by_name["train-long"], ("context_budget.request_window_tokens=32768",)
    )
    canary_base = {"preset": "dry", "lane": "async", "model": canary.DEFAULT_MODEL, "checkpoint": False, "export": False}
    canary_axes = {
        "preset": canary.PRESETS,
        "lane": ("async", "sync"),
        "model": canary.MODELS,
        "checkpoint": (False, True),
        "export": (False, True),
    }
    for options in _one_factor(canary_base, canary_axes):
        key = (
            f"canary/{options['preset']}/{options['lane']}/{options['model']}"
            f"/checkpoint={options['checkpoint']}/export={options['export']}"
        )
        builders[key] = partial(canary.build_run, **options, version=VERSION)
    builders["canary/settings/context-budget"] = partial(
        canary.build_run,
        preset="dry",
        version=VERSION,
        settings=("context_budget.request_window_tokens=256", "context_budget.max_new_tokens_per_turn=128"),
    )
    builders["canary/seed-identity"] = partial(canary.build_run, preset="dry", version=VERSION, seed=18)
    for stage in ("train", "resume"):
        builders[f"port-smoke/{stage}"] = partial(_port_smoke_step, stage)
    builders["iceball/workflow"] = _iceball_step
    for mode in ("whole", "step"):
        builders[f"task-sessions/{mode}"] = partial(validation.build_run, mode=mode, version=VERSION, steps=10)
    for label, preset in snowball_online_eagle.PRESETS.items():
        builders[f"eagle/{label}"] = partial(snowball_online_eagle.online_eagle_step, preset)
    replay_choices = ((), *((mode,) for mode in mismatch.REPLAY_MODES), mismatch.REPLAY_MODES)
    probe_base = {"cache": "off", "replay": (), "updates": 0, "resume": False, "reuse": False}
    probe_axes = {
        "cache": ("off", "on", "both"),
        "replay": replay_choices,
        "updates": (0, 2),
        "resume": (False, True),
        "reuse": (False, True),
    }
    for options in _one_factor(probe_base, probe_axes):
        cache, replay, updates, resume, reuse = (options[name] for name in probe_base)
        key = f"mismatch/cache={cache}/replay={','.join(replay)}/updates={updates}/resume={resume}/reuse={reuse}"
        builders[key] = partial(
            _probe_step,
            artifact_root,
            seed=17,
            prompt_count=2,
            samples_per_prompt=2,
            updates=updates,
            keep_fraction=0.5,
            cache_mode=cache,
            reuse_probe=str(artifact_root / "probe-archive") if reuse else None,
            resume_path=str(artifact_root / "checkpoints") if resume else None,
            extra_trainer_modes=replay,
        )
    return builders


def _seed_dependency_metadata(step: ArtifactStep[SkyRLRun], artifact_root: str) -> None:
    for dependency in step.deps:
        if issubclass(dependency.artifact_type, SkyRLRun):
            write_artifact(
                {
                    "hf_model_uri": "s3://launch-census/policy",
                    "global_step": 2,
                    "tokenizer_uri": "Qwen/Qwen2.5-0.5B-Instruct",
                    "tokenizer_revision": "pinned",
                    "checkpoint_root": str(StoragePath(artifact_root) / "saved-checkpoints"),
                    "draft_checkpoint_root": None,
                    "terminal_manifest_uri": str(StoragePath(artifact_root) / "terminal.json"),
                    "iris_job_id": "/test/train",
                },
                dependency.path(str(artifact_root)),
            )
        elif issubclass(dependency.artifact_type, LevanterCheckpoint):
            location = StoragePath(dependency.path(str(artifact_root)))
            checkpoint = location / "hf"
            checkpoint.mkdirs()
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "tokenizer_config.json").write_text("{}")


def runtime_launch(tmp_path: Path):
    source_class = get_filesystem_class("s3")
    register_implementation("s3", _MemoryObjectStore, clobber=True)
    prefix = "s3://launcher-test/" + tmp_path.name

    def render(step):
        _seed_dependency_metadata(step, prefix)
        return materialized_config(step, prefix)

    try:
        yield render
    finally:
        register_implementation("s3", source_class, clobber=True)


def render_launch_census(artifact_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Render runtime documents using seeded artifact metadata without running steps."""
    builders = launch_builders(artifact_root)
    documents = {}
    failures = {}
    source_class = get_filesystem_class("s3")
    register_implementation("s3", _MemoryObjectStore, clobber=True)
    prefix = "s3://launch-census/" + artifact_root.name
    try:
        with build_context(BuildContext(versions=VersionCodex(default=VERSION))):
            for name, build in builders.items():
                try:
                    step = build()
                    _seed_dependency_metadata(step, prefix)
                    config = materialized_config(step, prefix)
                    documents[name] = config.launch_config_yaml
                except Exception as error:
                    failures[name] = f"{type(error).__name__}: {error}"
    finally:
        register_implementation("s3", source_class, clobber=True)
    return documents, failures
