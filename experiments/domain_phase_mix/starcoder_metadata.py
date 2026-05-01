# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared StarCoder dataset metadata for offline analysis and nextgen loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec

DEFAULT_STARCODER_OBJECTIVE = "eval/paloma/dolma_100_programing_languages/bpb"
NEMOTRON_TOKENS = 5_729_908_864_777
STARCODER_TOKENS = 217_000_000_000
TARGET_BUDGET = NEMOTRON_TOKENS
DOMAIN_TOKEN_COUNTS = {
    "nemotron_full": NEMOTRON_TOKENS,
    "starcoder": STARCODER_TOKENS,
}
DOMAIN_NAMES = ("nemotron_full", "starcoder")
SMALL_DOMAIN_NAMES = ("starcoder",)


@dataclass(frozen=True)
class StarcoderDatasetDefinition:
    """Metadata for one StarCoder dataset topology."""

    name: str
    csv_path: Path
    phase_names: tuple[str, ...]
    phase_fracs: tuple[float, ...]


SCRIPT_DIR = Path(__file__).resolve().parent
EXPLORATORY_DIR = SCRIPT_DIR / "exploratory"

TWO_PHASE_STARCODER = StarcoderDatasetDefinition(
    name="two_phase_starcoder",
    csv_path=EXPLORATORY_DIR / "two_phase_starcoder_combined.csv",
    phase_names=("phase_0", "phase_1"),
    phase_fracs=(0.5, 0.5),
)

THREE_PHASE_STARCODER = StarcoderDatasetDefinition(
    name="three_phase_starcoder",
    csv_path=EXPLORATORY_DIR / "three_phase_starcoder.csv",
    phase_names=("phase_0", "phase_1", "phase_2"),
    phase_fracs=(0.33, 0.34, 0.33),
)

STARCODER_DATASETS = (TWO_PHASE_STARCODER, THREE_PHASE_STARCODER)


def is_starcoder_topology(phase_names: list[str] | tuple[str, ...], domain_names: list[str] | tuple[str, ...]) -> bool:
    """Return whether the provided topology matches the StarCoder setup."""
    return tuple(domain_names) == DOMAIN_NAMES and tuple(phase_names) in {
        TWO_PHASE_STARCODER.phase_names,
        THREE_PHASE_STARCODER.phase_names,
    }


def infer_starcoder_definition(
    phase_names: list[str] | tuple[str, ...],
    domain_names: list[str] | tuple[str, ...],
) -> StarcoderDatasetDefinition | None:
    """Infer the StarCoder definition for a phase/domain topology."""
    if tuple(domain_names) != DOMAIN_NAMES:
        return None
    phase_key = tuple(phase_names)
    for definition in STARCODER_DATASETS:
        if definition.phase_names == phase_key:
            return definition
    return None


def build_epoch_multipliers_for_definition(
    definition: StarcoderDatasetDefinition,
    domain_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Build real epoch multipliers for a StarCoder dataset definition."""
    return np.array(
        [
            [phase_frac * TARGET_BUDGET / DOMAIN_TOKEN_COUNTS[domain_name] for domain_name in domain_names]
            for phase_frac in definition.phase_fracs
        ],
        dtype=float,
    )


def infer_starcoder_metadata(
    phase_names: list[str] | tuple[str, ...],
    domain_names: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, list[int]] | None:
    """Infer epoch multipliers and small-domain indices for StarCoder layouts."""
    definition = infer_starcoder_definition(phase_names, domain_names)
    if definition is None:
        return None

    epoch_multipliers = build_epoch_multipliers_for_definition(definition, domain_names)
    small_domains = [idx for idx, domain_name in enumerate(domain_names) if domain_name in SMALL_DOMAIN_NAMES]
    return epoch_multipliers, small_domains


def _weights_from_frame(
    df: pd.DataFrame,
    phase_names: tuple[str, ...],
    domain_names: tuple[str, ...],
) -> np.ndarray:
    weights = np.zeros((len(df), len(phase_names), len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(phase_names):
        for domain_idx, domain_name in enumerate(domain_names):
            weights[:, phase_idx, domain_idx] = df[f"{phase_name}_{domain_name}"].to_numpy(dtype=float)
    return weights


def load_starcoder_dataset(
    definition: StarcoderDatasetDefinition,
    *,
    target_col: str = DEFAULT_STARCODER_OBJECTIVE,
) -> tuple[DatasetSpec, pd.DataFrame]:
    """Load one StarCoder CSV and construct a DatasetSpec with real epoching."""
    df = pd.read_csv(definition.csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "completed"].reset_index(drop=True)

    weights = _weights_from_frame(df, definition.phase_names, DOMAIN_NAMES)
    y = df[target_col].to_numpy(dtype=float)
    valid = np.isfinite(y)
    if not valid.all():
        weights = weights[valid]
        y = y[valid]
        df = df[valid].reset_index(drop=True)

    epoch_multipliers = build_epoch_multipliers_for_definition(definition, DOMAIN_NAMES)
    small_domains = [idx for idx, domain_name in enumerate(DOMAIN_NAMES) if domain_name in SMALL_DOMAIN_NAMES]
    spec = DatasetSpec(
        weights=weights,
        y=y,
        epoch_multipliers=epoch_multipliers,
        domain_names=list(DOMAIN_NAMES),
        phase_names=list(definition.phase_names),
        small_domains=small_domains,
        name=definition.name,
    )
    return spec, df


def load_two_phase_starcoder_dataset(
    *,
    target_col: str = DEFAULT_STARCODER_OBJECTIVE,
) -> tuple[DatasetSpec, pd.DataFrame]:
    """Load the 2-phase StarCoder dataset."""
    return load_starcoder_dataset(TWO_PHASE_STARCODER, target_col=target_col)


def load_three_phase_starcoder_dataset(
    *,
    target_col: str = DEFAULT_STARCODER_OBJECTIVE,
) -> tuple[DatasetSpec, pd.DataFrame]:
    """Load the 3-phase StarCoder dataset."""
    return load_starcoder_dataset(THREE_PHASE_STARCODER, target_col=target_col)


def iter_starcoder_datasets(
    *,
    target_col: str = DEFAULT_STARCODER_OBJECTIVE,
) -> list[tuple[StarcoderDatasetDefinition, DatasetSpec, pd.DataFrame]]:
    """Load both StarCoder datasets for batch analysis."""
    return [
        (definition, *load_starcoder_dataset(definition, target_col=target_col)) for definition in STARCODER_DATASETS
    ]
