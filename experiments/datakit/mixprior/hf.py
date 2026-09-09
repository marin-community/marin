# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The named Parquet format in marin-community/grug-moe-mix-swarm."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

HF_DATASET = "marin-community/grug-moe-mix-swarm"


@dataclass
class Data:
    name: str
    components: list[str]
    domains: list[str]
    available_tokens: np.ndarray
    phase_budgets: np.ndarray
    weights: np.ndarray
    labels: list[str]
    outcomes: np.ndarray
    groups: list[str]
    observation_ids: list[str]

    @property
    def exposure(self) -> np.ndarray:
        return self.phase_budgets[:, None] / self.available_tokens


def read_record(path: Path) -> dict:
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Expected one record in {path}")
    return rows[0]


def load_data(root: Path, swarm: str) -> Data:
    """Read a swarm directory; align every weight by cell name."""
    directory = root / "registry/v1/swarms" / swarm
    spec = read_record(directory / "swarm.parquet")
    cells = read_record(directory / "buckets.parquet")["cells"]
    rows = pq.read_table(directory / "observations.parquet").to_pylist()
    if not cells or not rows:
        raise ValueError("A swarm needs components and completed observations")
    names = [cell["cell"] for cell in cells]
    if len(names) != len(set(names)) or spec["swarm_id"] != swarm:
        raise ValueError("Swarm and component names must be unique and match the requested data")
    ids = [row["observation_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(row["swarm_id"] != swarm for row in rows):
        raise ValueError("Observation IDs must be unique and belong to this swarm")
    weights = []
    outcomes = []
    for row in rows:
        phases = [row["phase0_weights"], row["phase1_weights"]]
        if any(set(phase) != set(names) for phase in phases):
            raise ValueError("Observation weights must name every component exactly once")
        weights.append([[phase[name] for name in names] for phase in phases])
        evaluation = row["grouped_bpb"]
        training = row.get("training_eval_metrics", {})
        if evaluation.keys() & training.keys():
            raise ValueError("Training and grouped evaluation metrics must have distinct names")
        metrics = evaluation | training
        for group in ("include", "belebele"):
            leaves = [
                value for name, value in metrics.items() if name.startswith(group + "_") and name != group + "_mean"
            ]
            if leaves:
                metrics[group + "_mean"] = float(np.mean(leaves))
        outcomes.append(metrics)
    labels = sorted(outcomes[0])
    if any(set(row) != set(labels) for row in outcomes):
        raise ValueError("Every observation must contain the same evaluation metrics")
    weights = np.asarray(weights, dtype=np.float64)
    available = np.asarray([cell["available_tokens"] for cell in cells], dtype=np.float64)
    budgets = np.asarray(spec["phase_budgets"], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(axis=-1), 1):
        raise ValueError("Each observed phase must be a finite simplex")
    if (
        budgets.shape != (2,)
        or not np.isfinite(budgets).all()
        or not np.isfinite(available).all()
        or np.any(budgets <= 0)
        or np.any(available <= 0)
    ):
        raise ValueError("Token budgets and component availability must be positive")
    return Data(
        name=swarm,
        components=names,
        domains=[cell["domain"] for cell in cells],
        available_tokens=available,
        phase_budgets=budgets,
        weights=weights,
        labels=labels,
        outcomes=np.asarray([[row[label] for label in labels] for row in outcomes], dtype=np.float64),
        groups=[row["group"] for row in rows],
        observation_ids=ids,
    )


def load_hf(revision: str, swarm: str) -> Data:
    """Download only the three named files needed for one swarm."""
    path = snapshot_download(
        HF_DATASET,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"registry/v1/swarms/{swarm}/{name}.parquet" for name in ("swarm", "observations", "buckets")],
    )
    return load_data(Path(path), swarm)


def write_candidates(path: Path, weights: np.ndarray, swarm_id: str, components: list[str], revision: str) -> None:
    """Write candidate phases with the same cell names as the HF observations."""
    if weights.shape[1:] != (2, len(components)) or not np.isfinite(weights).all():
        raise ValueError("Candidate weights must match the named phase/component axes")
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=-1), 1):
        raise ValueError("Candidate phases must be simplexes")
    rows = [
        {
            "swarm_id": swarm_id,
            "hf_revision": revision,
            "phase0_weights": dict(zip(components, phases[0].tolist(), strict=True)),
            "phase1_weights": dict(zip(components, phases[1].tolist(), strict=True)),
        }
        for phases in weights
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
