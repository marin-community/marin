# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi checkpoints through 1e22 FLOPs for the GSM8K training-seed sweeps."""

import posixpath
from dataclasses import dataclass

from marin.utils import get_directory_friendly_name
from rigging.filesystem.buckets import filesystem_for
from thalas.execution.types import ExecutorStep, this_output_path, versioned

from experiments.downstream_scaling import train_delphi_seeds
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS, DELPHI_SEED_HF_DOWNLOADS

GCS_SOURCE_PREFIX = "gs://marin-us-central2"
GCS_RUNS: tuple[tuple[str, int], ...] = (
    ("3e18", 42),
    ("3e18", 62746),
    ("9e18", 42),
    ("9e18", 62746),
    ("2e19", 42),
    ("2e19", 62746),
    ("3e19", 42),
    ("3e19", 62746),
    ("5e19", 0),
    ("5e19", 42),
    ("5e19", 62746),
    ("9e19", 42),
    ("9e19", 62746),
    ("2e20", 42),
    ("2e20", 62746),
    ("3e20", 42),
    ("3e20", 62746),
    ("6e20", 0),
    ("6e20", 42),
    ("6e20", 62746),
)

DELPHI_SEED_CHECKPOINTS: dict[str, str] = {
    f"{slug}_seed{seed}": f"{step.name}/{step.version}/hf/step-{train_delphi_seeds.MODELS[slug].num_train_steps - 1}"
    for (slug, seed), step in zip(GCS_RUNS, train_delphi_seeds.build(GCS_RUNS), strict=True)
}


@dataclass(frozen=True)
class CopyCheckpointConfig:
    source: str
    output_path: str


def copy_checkpoint(config: CopyCheckpointConfig) -> None:
    """Copy an HF export into a regional model directory with server-side GCS copies."""
    fs, source_path = filesystem_for(config.source)
    paths = fs.find(source_path)
    if not paths:
        raise FileNotFoundError(f"No files under {config.source}")
    for path in paths:
        relative_path = posixpath.relpath(path, source_path)
        fs.cp_file(path, f"{config.output_path.rstrip('/')}/{relative_path}")


def _copy_checkpoint_step(key: str) -> ExecutorStep:
    relative_path = DELPHI_SEED_CHECKPOINTS[key]
    run_name, version, _, final_step = relative_path.split("/")
    relative_output = f"models/{run_name}--{get_directory_friendly_name(version)}--{final_step}"
    return ExecutorStep(
        name=relative_output,
        fn=copy_checkpoint,
        config=CopyCheckpointConfig(
            source=versioned(f"{GCS_SOURCE_PREFIX}/{relative_path}"),
            output_path=this_output_path(),
        ),
        override_output_path=relative_output,
    )


MODEL_STEPS: dict[str, ExecutorStep] = {
    "3e18": DELPHI_HF_DOWNLOADS["3e18"],
    "3e18_seed42": _copy_checkpoint_step("3e18_seed42"),
    "3e18_seed62746": _copy_checkpoint_step("3e18_seed62746"),
    "9e18": DELPHI_HF_DOWNLOADS["9e18"],
    "9e18_seed42": _copy_checkpoint_step("9e18_seed42"),
    "9e18_seed62746": _copy_checkpoint_step("9e18_seed62746"),
    "2e19": DELPHI_HF_DOWNLOADS["2e19"],
    "2e19_seed42": _copy_checkpoint_step("2e19_seed42"),
    "2e19_seed62746": _copy_checkpoint_step("2e19_seed62746"),
    "3e19": DELPHI_HF_DOWNLOADS["3e19"],
    "3e19_seed42": _copy_checkpoint_step("3e19_seed42"),
    "3e19_seed62746": _copy_checkpoint_step("3e19_seed62746"),
    "5e19_seed0": _copy_checkpoint_step("5e19_seed0"),
    "5e19_seed42": _copy_checkpoint_step("5e19_seed42"),
    "5e19_seed62746": _copy_checkpoint_step("5e19_seed62746"),
    "9e19": DELPHI_HF_DOWNLOADS["9e19"],
    "9e19_seed42": _copy_checkpoint_step("9e19_seed42"),
    "9e19_seed62746": _copy_checkpoint_step("9e19_seed62746"),
    "2e20": DELPHI_HF_DOWNLOADS["2e20"],
    "2e20_seed42": _copy_checkpoint_step("2e20_seed42"),
    "2e20_seed62746": _copy_checkpoint_step("2e20_seed62746"),
    "3e20": DELPHI_HF_DOWNLOADS["3e20"],
    "3e20_seed42": _copy_checkpoint_step("3e20_seed42"),
    "3e20_seed62746": _copy_checkpoint_step("3e20_seed62746"),
    "6e20_seed0": _copy_checkpoint_step("6e20_seed0"),
    "6e20_seed42": _copy_checkpoint_step("6e20_seed42"),
    "6e20_seed62746": _copy_checkpoint_step("6e20_seed62746"),
    "1e21": DELPHI_HF_DOWNLOADS["1e21"],
    "1e21_seed42": DELPHI_SEED_HF_DOWNLOADS["1e21_seed42"],
    "1e21_seed62746": DELPHI_SEED_HF_DOWNLOADS["1e21_seed62746"],
    "1e22": DELPHI_HF_DOWNLOADS["1e22"],
    "1e22_seed42": DELPHI_SEED_HF_DOWNLOADS["1e22_seed42"],
    "1e22_seed62746": DELPHI_SEED_HF_DOWNLOADS["1e22_seed62746"],
}
