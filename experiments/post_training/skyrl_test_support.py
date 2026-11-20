# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, materialized_config
from rigging.config_discovery import find_project_root
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training.rl_presets import ModelPresets
from experiments.post_training.task_curation.selection import TRAIN_FILENAME


def with_current_control_revisions(table: ModelPresets) -> ModelPresets:
    return replace(
        table,
        control_pins={
            name: replace(pin, revision=table.by_name[name].revision) for name, pin in table.control_pins.items()
        },
    )


def selection_summary(full_rows: int) -> dict:
    return {
        "recipe": {"heldout": [{"version": "2026.10.23.1", "rows": 910}]},
        "training_rows": 100000,
        "validation_rows": 100,
        "full_validation_rows": full_rows,
        "grader_kind_counts": {"train": {}, "validation": {}, "full_validation": {}},
    }


def write_training_families(root: StoragePath, counts: Mapping[str, int]) -> None:
    families = [family for family, count in counts.items() for _ in range(count)]
    with (root / TRAIN_FILENAME).open("wb") as stream:
        pq.write_table(pa.table({"family": families}), stream)


def native_json(*uv_options: str):
    def execute(script, paths):
        source = str(find_project_root(Path(__file__)) / "vendor/skyrl")
        program = gate._native_program(source, ("cloud.iris.launch_config", "skyrl_train"), script)
        try:
            return gate._native_json(*uv_options, "python", "-c", program, *paths, timeout=170)
        except gate.InvalidComparison as error:
            cause = error.__cause__
            if isinstance(cause, subprocess.CalledProcessError) and cause.stderr:
                raise AssertionError(cause.stderr.strip()) from error
            raise

    return execute


def assert_revisioned_native_configs(
    table: ModelPresets,
    revisions: Mapping[str, Mapping[str, str]],
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mixed_tasks: ArtifactStep | None = None,
) -> list[dict]:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    bindings = {binding for preset in table.presets for binding in (preset.tasks, preset.held_out_tasks) if binding}
    summaries = {
        binding: (Path(recipe.selection_artifact(binding).path()) / recipe.SUMMARY_FILENAME).read_text()
        for binding in bindings
    }
    namespace = table.namespace.removesuffix("-rl")
    monkeypatch.setattr(recipe, "SELECTION_ROOT", f"memory://{namespace}-calibration")
    for binding, summary in summaries.items():
        (StoragePath(recipe.selection_artifact(binding).path()) / recipe.SUMMARY_FILENAME).write_text(summary)
    paths = []
    for name in revisions:
        preset = table.by_name[name]
        spec = recipe.run_spec(
            preset,
            name="fixture-default-" + name,
            model_name=table.model,
            tasks=mixed_tasks if preset.batch_mix is not None else None,
        )
        step = recipe.run_step(preset, spec)
        path = tmp_path / f"{name}.yaml"
        path.write_text(materialized_config(step, f"/{namespace}-pin-prefix").launch_config_yaml)
        paths.append(str(path))
    configs = gate.compose_configs(paths)
    actual = {}
    for name, config in zip(revisions, configs, strict=True):
        canonical = json.dumps(gate.comparison_config(config), sort_keys=True)
        if mixed_tasks is not None:
            canonical = canonical.replace(mixed_tasks.path(), "<mixed-selection-fixture>")
            pinned = f"{mixed_tasks.name}@{mixed_tasks.version}"
            canonical = canonical.replace(f"{pinned}:{mixed_tasks.fingerprint()}", f"{pinned}:<fixture>")
        actual[name] = fingerprint_hash(canonical)
    assert actual == {name: values.get(table.by_name[name].revision) for name, values in revisions.items()}, actual
    for preset, config in zip((table.by_name[name] for name in revisions), configs, strict=True):
        staleness = preset.recipe.config.trainer.rollout_buffer.max_staleness_steps
        if staleness is not None:
            assert config["skyrl"]["trainer"]["rollout_buffer"]["max_staleness_steps"] == staleness
    return configs
