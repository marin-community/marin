# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate external Parquet records into validated two-phase swarm arrays."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypedDict, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PHASE_COUNT = 2


class ArtifactReference(TypedDict):
    path: str
    sha256: str


class ObservationArtifactReference(ArtifactReference):
    schema: str


class MixtureComponentMetadata(TypedDict):
    cell: str
    domain: str
    quality: int
    available_tokens: float


class MixtureComponentRecord(TypedDict):
    cells: list[MixtureComponentMetadata]


class ContentRecord(TypedDict):
    basis_id: str
    cells: list[str]
    matrix: list[list[float]]
    provenance: dict[str, Any]


class SwarmObservations(NamedTuple):
    observation_ids: list[str]
    run_names: list[str]
    groups: list[str]
    mixture_components: list[str]
    component_metadata: list[MixtureComponentMetadata]
    available_tokens: np.ndarray
    weights: np.ndarray
    labels: list[str]
    outcomes: np.ndarray


@dataclass(frozen=True)
class SwarmProvenance:
    store_uri: str
    store_artifact_uri: str
    training_recipe: str
    tokenizer: str
    evaluation_suite: str
    model_active_parameters: int
    model_total_parameters: int
    physical_training_tokens: int
    simulated_training_tokens: int
    content_provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if not all(
            (
                self.store_uri,
                self.store_artifact_uri,
                self.training_recipe,
                self.tokenizer,
                self.evaluation_suite,
            )
        ):
            raise ValueError("Every swarm needs complete provenance")
        model_context = (
            self.model_active_parameters,
            self.model_total_parameters,
            self.physical_training_tokens,
            self.simulated_training_tokens,
        )
        if any(value <= 0 for value in model_context):
            raise ValueError("Every fixed context feature must be positive")
        if self.model_active_parameters > self.model_total_parameters:
            raise ValueError("Active parameters cannot exceed total parameters")


@dataclass(frozen=True)
class Swarm:
    swarm_id: str
    data: SwarmObservations
    phase_budgets: np.ndarray
    content_basis_id: str
    content_matrix: np.ndarray
    provenance: SwarmProvenance

    def __post_init__(self) -> None:
        budgets = np.asarray(self.phase_budgets, dtype=np.float64)
        content = np.asarray(self.content_matrix, dtype=np.float64)
        if not self.swarm_id:
            raise ValueError("Swarm ID is required")
        if budgets.shape != (PHASE_COUNT,) or self.data.weights.shape[1] != PHASE_COUNT:
            raise ValueError(f"A swarm must contain exactly {PHASE_COUNT} training phases")
        if np.any(budgets <= 0) or not np.isfinite(budgets).all():
            raise ValueError("Phase budgets must be finite and positive")
        if not np.isclose(budgets.sum(), self.provenance.simulated_training_tokens, rtol=1e-12):
            raise ValueError("Phase budgets must sum to simulated training tokens")
        if content.ndim != 2 or content.shape[0] != len(self.data.mixture_components):
            raise ValueError("Content matrix must have one row per mixture component")
        if np.any(content < 0) or not np.allclose(content.sum(axis=1), 1.0):
            raise ValueError("Every content row must be a probability distribution")
        if not self.content_basis_id:
            raise ValueError("Content basis ID is required")
        object.__setattr__(self, "phase_budgets", budgets)
        object.__setattr__(self, "content_matrix", content)

    @property
    def exposure_multipliers(self) -> np.ndarray:
        return self.phase_budgets[:, None] / self.data.available_tokens[None, :]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path: Path) -> dict[str, Any]:
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Expected one record in {path}, found {len(rows)}")
    return rows[0]


def write_record(path: Path, record: dict[str, Any]) -> None:
    pq.write_table(pa.Table.from_pylist([record]), path, compression="zstd")


def record_sha256(record: dict[str, Any]) -> str:
    table = pa.Table.from_pylist([record])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def load_observations(parquet_path: Path, components_path: Path, swarm_id: str) -> SwarmObservations:
    frame = pd.read_parquet(parquet_path)
    required = {
        "observation_id",
        "swarm_id",
        "run_name",
        "group",
        "phase0_weights",
        "phase1_weights",
        "grouped_bpb",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Observation artifact is missing columns: {missing}")
    observation_ids = frame.observation_id.tolist()
    id_prefix = f"{swarm_id}:"
    if (
        not all(
            isinstance(value, str) and value.startswith(id_prefix) and len(value) > len(id_prefix)
            for value in observation_ids
        )
        or not frame.observation_id.is_unique
    ):
        raise ValueError("Observation IDs must be unique, non-empty, and swarm-qualified")
    observed_swarms = set(frame.swarm_id)
    if observed_swarms != {swarm_id}:
        raise ValueError(f"Swarm {swarm_id} contains observations from {sorted(observed_swarms)}")

    component_record = cast(MixtureComponentRecord, read_record(components_path))
    metadata = component_record["cells"]
    required_metadata = {"cell", "domain", "quality", "available_tokens"}
    for row in metadata:
        missing_metadata = sorted(required_metadata - set(row))
        if missing_metadata:
            raise ValueError(f"Mixture component metadata is missing fields: {missing_metadata}")
        if not isinstance(row["domain"], str) or not row["domain"]:
            raise ValueError("Every mixture component must have a domain")
        if not isinstance(row["quality"], int) or row["quality"] < 0:
            raise ValueError("Every mixture component must have a non-negative integer quality tier")
    component_names = [row["cell"] for row in metadata]
    if len(component_names) != len(set(component_names)):
        raise ValueError("Mixture component names must be unique")
    available_tokens = np.asarray([row["available_tokens"] for row in metadata], dtype=np.float64)
    if np.any(available_tokens <= 0) or not np.isfinite(available_tokens).all():
        raise ValueError("Every mixture component must have a positive token count")

    weights = np.empty((len(frame), 2, len(component_names)), dtype=np.float64)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        phase0 = _weight_map(row.phase0_weights, component_names, row.observation_id, 0)
        phase1 = _weight_map(row.phase1_weights, component_names, row.observation_id, 1)
        weights[row_index, 0] = [phase0[name] for name in component_names]
        weights[row_index, 1] = [phase1[name] for name in component_names]
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Observation weights must be finite and non-negative")
    if not np.allclose(weights.sum(axis=-1), 1.0, atol=1e-9):
        raise ValueError("Every observation phase must be a simplex vector")

    grouped = frame.grouped_bpb.tolist()
    if not all(isinstance(row, dict) for row in grouped):
        raise TypeError("Canonical BPB values must be mappings")
    label_sets = {frozenset(row) for row in grouped}
    if len(label_sets) != 1:
        raise ValueError("Every observation must contain the same BPB labels")
    labels = sorted(label_sets.pop())
    if not labels:
        raise ValueError("Observation BPB maps must be non-empty")
    outcomes = np.asarray([[row[label] for label in labels] for row in grouped])
    if not np.isfinite(outcomes).all():
        raise ValueError("Observation BPBs must be finite and complete")
    return SwarmObservations(
        observation_ids,
        frame.run_name.tolist(),
        frame.group.tolist(),
        component_names,
        metadata,
        available_tokens,
        weights,
        labels,
        outcomes,
    )


def _weight_map(value: object, component_names: list[str], observation_id: str, phase: int) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"Observation {observation_id} phase {phase} weights must be a mapping")
    if set(value) != set(component_names):
        raise ValueError(f"Observation {observation_id} phase {phase} weights do not match its mixture components")
    return value
