# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from collections.abc import Callable
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

from marin.execution.artifact import write_artifact
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import iceball_micro, skyrl_port_smoke, snowball_online_eagle, snowball_rl
from experiments.post_training import snowball_rl_presets as snowball
from experiments.post_training.cat_count_canary import launcher as canary
from experiments.post_training.curriculum_rl import launch as curriculum
from experiments.post_training.mismatch_probe import launch as mismatch
from experiments.post_training.task_sessions import validation

VERSION = "2026.10.24"


def _curriculum_step(scale, policy, arm) -> ArtifactStep[SkyRLRun]:
    return curriculum.build_arms(specs=(arm,), scale=scale, policy=policy, version=VERSION)[arm.name].rl


def _snowball_step(preset: snowball.Preset, settings: tuple[str, ...] = ()) -> ArtifactStep[SkyRLRun]:
    panel = snowball.selection_artifact(preset.eval_lite)
    selection = snowball.default_selection(preset)
    for source, full_rows in ((selection, 910), (panel, 1500)):
        summary = {
            "recipe": {
                "heldout": [],
                "exclusions": [
                    {"uri": str(StoragePath(panel.path()) / snowball.FULL_VALIDATION_FILENAME), "version": panel.version}
                ],
            },
            "training_rows": 100000,
            "validation_rows": 100,
            "full_validation_rows": full_rows,
            "grader_kind_counts": {"train": {}, "validation": {}, "full_validation": {}},
        }
        path = StoragePath(source.path()) / snowball.SUMMARY_FILENAME
        path.parent.mkdirs()
        path.write_text(json.dumps(summary))
    return snowball_rl.build_run(
        preset,
        tasks=selection if preset.focus == snowball.Focus.DATA else None,
        settings=settings,
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
    for label, preset in snowball.PRESETS.items():
        builders[f"snowball/{label}"] = partial(_snowball_step, preset)
    builders["snowball/settings/optimizer"] = partial(
        _snowball_step, snowball.PRESETS["learning-medium"], ("trainer.policy.optimizer_config.lr=2e-6",)
    )
    builders["snowball/settings/context-budget"] = partial(
        _snowball_step, snowball.PRESETS["train-long"], ("context_budget.request_window_tokens=32768",)
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


def _seed_dependency_metadata(step: ArtifactStep[SkyRLRun], artifact_root: Path) -> None:
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
        elif issubclass(dependency.artifact_type, LevanterCheckpoint):
            location = StoragePath(dependency.path(str(artifact_root)))
            if location.is_remote:
                continue
            checkpoint = Path(str(location)) / "hf"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "tokenizer_config.json").write_text("{}")


def render_launch_census(artifact_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Render runtime documents using local metadata without running steps."""
    builders = launch_builders(artifact_root)
    documents = {}
    failures = {}
    with build_context(BuildContext(versions=VersionCodex(default=VERSION))):
        for name, build in builders.items():
            try:
                step = build()
                _seed_dependency_metadata(step, artifact_root)
                config = materialized_config(step, str(artifact_root))
                documents[name] = config.launch_config_yaml
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
    return documents, failures
