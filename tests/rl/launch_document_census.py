# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import yaml
from marin.execution.artifact import write_artifact
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import async_rl, iceball_micro, skyrl_port_smoke, snowball_online_eagle
from experiments.post_training.cat_count_canary import launcher as canary
from experiments.post_training.curriculum_rl import launch as curriculum
from experiments.post_training.inventory import train_quick_loop as inventory
from experiments.post_training.mismatch_probe import launch as mismatch
from experiments.post_training.task_sessions import validation

VERSION = "2026.10.24"


@dataclass(frozen=True)
class LaunchDocument:
    case: str
    document: dict[str, Any]


@dataclass(frozen=True)
class BuildFailure:
    case: str
    error_type: str
    message: str


@dataclass(frozen=True)
class LaunchCensus:
    cases: tuple[str, ...]
    documents: tuple[LaunchDocument, ...]
    failures: tuple[BuildFailure, ...]


def _curriculum_step(scale, policy, arm) -> ArtifactStep[SkyRLRun]:
    return curriculum.build_arms(specs=(arm,), scale=scale, policy=policy, version=VERSION)[arm.name].rl


def _async_step(preset, settings=()) -> ArtifactStep[SkyRLRun]:
    return async_rl.build_run(curriculum.SNOWBALL_POLICY, preset, VERSION, settings=settings).rl


def _iceball_step() -> ArtifactStep[SkyRLRun]:
    return iceball_micro.build_workflow(version=VERSION).rl


def _probe_step(artifact_root: Path, settings: mismatch.ProbeSettings) -> ArtifactStep[SkyRLRun]:
    # The build-options wrapper displays/runs a graph. Its wrapped Click build body returns that graph.
    builder = inspect.unwrap(mismatch.main.callback)
    return builder(
        model_uri=str(artifact_root / "iceball-sft"),
        data_uri=str(artifact_root / "gsm8k"),
        input_version=VERSION,
        **asdict(settings),
    )


def _inventory_step(artifact_root: Path) -> ArtifactStep[SkyRLRun]:
    selection = ArtifactStep.adopt("rl-data/task-selection/census", VERSION, str(artifact_root / "selection"))
    return inventory.build_run(VERSION, selection=selection)


def _port_smoke_step(stage: str) -> ArtifactStep[SkyRLRun]:
    return getattr(skyrl_port_smoke.build_smoke(VERSION), stage)


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
    for label, preset in async_rl.PRESETS.items():
        builders[f"async/{label}"] = partial(_async_step, preset)
    default = async_rl.PRESETS["default"]
    builders["async/settings/checkpoint-interval"] = partial(
        _async_step, default, ("trainer.max_steps=3", "trainer.eval_interval=2")
    )
    builders["async/settings/sampling"] = partial(_async_step, default, ("generator.sampling_params.temperature=0.9",))
    canary_base = {"preset": "dry", "lane": "async", "model": canary.DEFAULT_MODEL, "checkpoint": False, "export": False}
    canary_axes = {
        "preset": canary.PRESETS,
        "lane": ("async", "sync"),
        "model": canary.MODELS,
        "checkpoint": (False, True),
        "export": (False, True),
    }
    canary_variants = [
        canary_base | {name: value}
        for name, choices in canary_axes.items()
        for value in choices
        if value != canary_base[name]
    ]
    for options in [canary_base, *canary_variants]:
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
    builders["inventory/train"] = partial(_inventory_step, artifact_root)
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
    probe_variants = [
        probe_base | {name: value}
        for name, choices in probe_axes.items()
        for value in choices
        if value != probe_base[name]
    ]
    for options in [probe_base, *probe_variants]:
        cache, replay, updates, resume, reuse = (options[name] for name in probe_base)
        settings = mismatch.ProbeSettings(
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
        key = f"mismatch/cache={cache}/replay={','.join(replay)}/updates={updates}/resume={resume}/reuse={reuse}"
        builders[key] = partial(_probe_step, artifact_root, settings)
    return builders


def _write_model_metadata(step: ArtifactStep[SkyRLRun], artifact_root: Path) -> None:
    for dependency in step.deps:
        if issubclass(dependency.artifact_type, SkyRLRun):
            write_artifact(
                {
                    "hf_model_uri": str(artifact_root / "policy"),
                    "global_step": 2,
                    "tokenizer_uri": "Qwen/Qwen2.5-0.5B-Instruct",
                    "tokenizer_revision": "pinned",
                    "checkpoint_root": str(artifact_root / "saved-checkpoints"),
                    "draft_checkpoint_root": None,
                    "terminal_manifest_uri": str(artifact_root / "terminal.json"),
                    "iris_job_id": "/test/train",
                },
                dependency.path(str(artifact_root)),
            )
        if not issubclass(dependency.artifact_type, LevanterCheckpoint):
            continue
        location = StoragePath(dependency.path(str(artifact_root)))
        if location.is_remote:
            continue
        checkpoint = Path(str(location)) / "hf"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "tokenizer_config.json").write_text("{}")


def render_launch_census(artifact_root: Path) -> LaunchCensus:
    """Render runtime documents using local metadata without running steps."""
    builders = launch_builders(artifact_root)
    documents = []
    failures = []
    with (
        build_context(BuildContext(versions=VersionCodex(default=VERSION))),
        patch("uuid.uuid4", return_value=UUID(int=17)),
    ):
        for name, build in builders.items():
            try:
                step = build()
                _write_model_metadata(step, artifact_root)
                config = materialized_config(step, str(artifact_root))
                documents.append(LaunchDocument(name, yaml.safe_load(config.launch_config_yaml)))
            except Exception as error:
                failures.append(BuildFailure(name, type(error).__name__, str(error)))
    return LaunchCensus(tuple(builders), tuple(documents), tuple(failures))
