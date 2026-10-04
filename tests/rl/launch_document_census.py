# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import async_rl, iceball_micro, snowball_online_eagle
from experiments.post_training.cat_count_canary import launcher as canary
from experiments.post_training.curriculum_rl import launch as curriculum
from experiments.post_training.mismatch_probe import launch as mismatch
from experiments.post_training.tasktrove import rl_smoke as tasktrove

VERSION = "2026.10.03"


@dataclass(frozen=True)
class LaunchDocument:
    case: str
    document: dict[str, Any]
    fingerprint: str
    launcher_requirement: str


@dataclass(frozen=True)
class BuildFailure:
    case: str
    error_type: str
    message: str


@dataclass(frozen=True)
class UnsupportedCase:
    case: str
    reason: str


@dataclass(frozen=True)
class LaunchCensus:
    cases: tuple[str, ...]
    documents: tuple[LaunchDocument, ...]
    failures: tuple[BuildFailure, ...]
    unsupported: tuple[UnsupportedCase, ...]


def _curriculum_step(scale, policy, arm) -> ArtifactStep[SkyRLRun]:
    return curriculum.build_arms(specs=(arm,), scale=scale, policy=policy, version=VERSION)[arm.name].rl


def _async_step(preset, settings=()) -> ArtifactStep[SkyRLRun]:
    return async_rl.build_run(curriculum.SNOWBALL_POLICY, preset, VERSION, settings=settings).rl


def _iceball_step() -> ArtifactStep[SkyRLRun]:
    return iceball_micro.build_workflow(version=VERSION).rl


def _tasktrove_step(artifact_root: Path) -> ArtifactStep[SkyRLRun]:
    release_path = artifact_root / "tasktrove"
    release_path.mkdir(parents=True, exist_ok=True)
    (release_path / "manifest.json").write_text(json.dumps({"verify_tool_ref": "verifyit==0.1.0"}))
    release = ArtifactStep.adopt("documents/launch-census-tasktrove", VERSION, str(release_path))
    return tasktrove.smoke_step(release)


def _probe_step(artifact_root: Path, settings: mismatch.ProbeSettings) -> ArtifactStep[SkyRLRun]:
    # The build-options wrapper displays/runs a graph. Its wrapped Click build body returns that graph.
    builder = inspect.unwrap(mismatch.main.callback)
    return builder(
        model_uri=str(artifact_root / "iceball-sft"),
        data_uri=str(artifact_root / "gsm8k"),
        input_version=VERSION,
        **asdict(settings),
    )


def launch_builders(
    artifact_root: Path,
) -> tuple[dict[str, Callable[[], ArtifactStep[SkyRLRun]]], tuple[UnsupportedCase, ...]]:
    """Enumerate current public presets and representative document-changing CLI options."""
    builders = {}
    unsupported = []
    for scale, policy, arm in product(curriculum.SCALES, curriculum.POLICIES.values(), curriculum.ARMS.values()):
        name = f"curriculum/{scale}/{policy.label}/{arm.name}"
        if (policy is curriculum.SNOWBALL_POLICY) != scale.startswith("snowball-"):
            unsupported.append(UnsupportedCase(name, f"scale {scale!r} belongs to a different policy family"))
            continue
        builders[name] = partial(_curriculum_step, scale, policy, arm)
    for label, preset in async_rl.PRESETS.items():
        builders[f"async/{label}"] = partial(_async_step, preset)
    default = async_rl.PRESETS["default"]
    builders["async/settings/checkpoint-interval"] = partial(
        _async_step, default, ("trainer.max_steps=3", "trainer.eval_interval=2")
    )
    builders["async/settings/sampling"] = partial(_async_step, default, ("generator.sampling_params.temperature=0.9",))
    for preset, lane, model, checkpoint, export in product(
        canary.PRESETS, ("async", "sync"), canary.MODELS, (False, True), (False, True)
    ):
        key = f"canary/{preset}/{lane}/{model}/checkpoint={checkpoint}/export={export}"
        builders[key] = partial(
            canary.build_run,
            preset=preset,
            lane=lane,
            model=model,
            checkpoint=checkpoint,
            export=export,
            version=VERSION,
        )
    builders["canary/settings/context-budget"] = partial(
        canary.build_run,
        preset="dry",
        version=VERSION,
        settings=("context_budget.request_window_tokens=256", "context_budget.max_new_tokens_per_turn=128"),
    )
    builders["canary/seed-identity"] = partial(canary.build_run, preset="dry", version=VERSION, seed=18)
    builders["iceball/workflow"] = _iceball_step
    builders["tasktrove/smoke"] = partial(_tasktrove_step, artifact_root)
    for label, preset in snowball_online_eagle.PRESETS.items():
        builders[f"eagle/{label}"] = partial(snowball_online_eagle.online_eagle_step, preset)
    replay_choices = ((), *((mode,) for mode in mismatch.REPLAY_MODES), mismatch.REPLAY_MODES)
    for cache, replay, updates, resume, reuse in product(
        ("off", "on", "both"), replay_choices, (0, 2), (False, True), (False, True)
    ):
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
    return builders, tuple(unsupported)


def _write_model_metadata(step: ArtifactStep[SkyRLRun], artifact_root: Path) -> None:
    for dependency in step.deps:
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
    builders, unsupported = launch_builders(artifact_root)
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
                documents.append(
                    LaunchDocument(
                        name, yaml.safe_load(config.launch_config_yaml), step.fingerprint(), config.launcher_requirement
                    )
                )
            except Exception as error:
                failures.append(BuildFailure(name, type(error).__name__, str(error)))
    return LaunchCensus(tuple(builders), tuple(documents), tuple(failures), unsupported)
