# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.0", "numpy==2.3.5"]
# ///

"""Freeze dense WSD80 surfaces across token horizon and StarCoder support."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import jax
import numpy as np

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DOMAIN_PHASE_MIX_DIR = SCRIPT_DIR.parents[1]
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
OBSERVATIONS_PATH = PANEL_DIR / "stage3_dense_surface_results_20260802" / "combined_discovery_observations.csv"
OUTPUT_PATH = DOMAIN_PHASE_MIX_DIR / "starcoder_wsd80_dense_support_surface_design_20260808.json"
ARTIFACT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_dense_support_surface_design_20260808"
MANIFEST_CSV_PATH = ARTIFACT_DIR / "run_manifest.csv"
REPORT_PATH = ARTIFACT_DIR / "report.md"

DESIGN_VERSION = "2026-08-08-v5"
PHASE_0_FRACTION = Fraction(4, 5)
PHASE_1_FRACTION = Fraction(1, 5)
REFERENCE_SEED = 20_260_711
REPEAT_SEEDS = (20_260_811, 20_260_812, 20_260_813)
STARCODER_SOURCE_TOKENS = 216_567_300_822
STARCODER_SOURCE_TOKEN_PROVENANCE = (
    "experiments/domain_phase_mix/domains.py:DOLMA_TOKENS; central1 tokenized cache queried 2025-01-28"
)
STARCODER_CACHE_RELATIVE_PATH = "tokenized/dolma/starcoder-8b6089"
STARCODER_CACHE_DOCUMENTS = 206_640_114
STARCODER_CACHE_SHARDS = 49
STARCODER_CACHE_LAYOUT = "consolidated"
STARCODER_CACHE_TOKENIZER_METADATA = {
    "append_bos": False,
    "append_eos": True,
    "max_length": 131_072,
    "padding": False,
    "return_attention_mask": False,
    "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
    "vocab_size": 128_256,
}
DESIGN_JAX_VERSION = "0.11.0"
DESIGN_NUMPY_VERSION = "2.3.5"
DESIGN_JAX_DEFAULT_PRNG_IMPL = "threefry2x32"
DESIGN_JAX_ENABLE_X64 = False
TRAINING_JAX_VERSION = "0.10.1"
TRAINING_NUMPY_VERSION = "2.3.5"
TRAINING_JAX_DEFAULT_PRNG_IMPL = "threefry2x32"
TRAINING_JAX_ENABLE_X64 = False
UV_LOCK_SHA256 = "d6a6a17fda4dd7d6c3733efcbee87151bb74971219e7a17c6ed05ba7a788086d"
NEMOTRON_SOURCE_TOKENS = base.TARGET_BUDGET
TOKENS_PER_BATCH = base.BATCH_SIZE * base.SEQ_LEN
EXPECTED_CELL_COUNT = 4
EXPECTED_COORDINATES_PER_CELL = 125
EXPECTED_SUPPORT_COUNT = 7
EXPECTED_COMPLETE_PRIMARY_ROWS = EXPECTED_CELL_COUNT * EXPECTED_SUPPORT_COUNT * EXPECTED_COORDINATES_PER_CELL
CALIBRATION_COORDINATES = (
    (0.0364194347695976, 0.0364194347695976),
    (0.10, 0.10),
    (0.18, 0.18),
    (0.35, 0.35),
    (0.70, 0.70),
    (0.02, 0.82),
    (0.10, 0.50),
    (0.39, 0.19),
)
EXPECTED_COMPLETE_REPEAT_ROWS = (
    EXPECTED_CELL_COUNT * EXPECTED_SUPPORT_COUNT * len(CALIBRATION_COORDINATES) * len(REPEAT_SEEDS)
)
EXPECTED_ALIAS_ROWS = 504
EXPECTED_PRIMARY_RUNS = 3_104
EXPECTED_REPEAT_RUNS = 564
EXPECTED_RUN_COUNT = EXPECTED_PRIMARY_RUNS + EXPECTED_REPEAT_RUNS
EXPECTED_COMPLETE_ROWS = EXPECTED_COMPLETE_PRIMARY_ROWS + EXPECTED_COMPLETE_REPEAT_ROWS


@dataclass(frozen=True)
class SupportSpec:
    """One StarCoder-only unique-support intervention."""

    support_id: str
    epoch_multiplier_numerator: int | None
    epoch_multiplier_denominator: int | None
    role: str

    @property
    def epoch_multiplier(self) -> Fraction | None:
        if self.epoch_multiplier_numerator is None:
            return None
        assert self.epoch_multiplier_denominator is not None
        return Fraction(self.epoch_multiplier_numerator, self.epoch_multiplier_denominator)


SUPPORT_SPECS = (
    SupportSpec("full", None, None, "complete_physical_starcoder_cache"),
    SupportSpec("m0125", 1, 8, "one_eighth_historical_starcoder_epoch_burden"),
    SupportSpec("m025", 1, 4, "one_quarter_historical_starcoder_epoch_burden"),
    SupportSpec("m050", 1, 2, "one_half_historical_starcoder_epoch_burden"),
    SupportSpec("m100", 1, 1, "historical_starcoder_epoch_burden"),
    SupportSpec("m200", 2, 1, "twice_historical_starcoder_epoch_burden"),
    SupportSpec("m400", 4, 1, "four_times_historical_starcoder_epoch_burden"),
)


def canonical_sha256(value: Any) -> str:
    """Return a stable hash for a JSON-compatible value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coordinate_key(phase_0: float, phase_1: float) -> tuple[float, float]:
    return round(float(phase_0), 6), round(float(phase_1), 6)


def _coordinate_role(phase_0: float, phase_1: float) -> str:
    if abs(phase_0 - phase_1) <= 1e-12:
        return "tied"
    if min(phase_0, phase_1) <= 1e-12 or max(phase_0, phase_1) >= 1.0 - 1e-12:
        return "boundary_untied"
    return "interior_untied"


def _cell_slug(cell_id: str) -> str:
    rung = cell_id.split("_", maxsplit=1)[0]
    steps = cell_id.rsplit("_s", maxsplit=1)[1]
    return f"{rung}d{steps}"


def _load_fixed_n_cells() -> tuple[dict[str, Any], ...]:
    rows_by_cell: dict[str, list[dict[str, str]]] = {}
    with OBSERVATIONS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["hidden_size"]) != 640:
                continue
            tracks = row["track_memberships"]
            if "increase_d" not in tracks:
                continue
            rows_by_cell.setdefault(row["cell_id"], []).append(row)

    cells: list[dict[str, Any]] = []
    for cell_id, rows in rows_by_cell.items():
        first = rows[0]
        cells.append(
            {
                "cell_id": cell_id,
                "cell_slug": _cell_slug(cell_id),
                "rung": int(first["rung"]),
                "hidden_size": int(first["hidden_size"]),
                "total_steps": int(first["total_steps"]),
                "boundary_step": int(first["boundary_step"]),
                "materialized_tokens": int(first["materialized_tokens"]),
                "total_parameters": int(first["total_parameters"]),
                "non_embedding_parameters": int(first["non_embedding_parameters"]),
                "prior_rows": rows,
            }
        )
    cells.sort(key=lambda item: item["rung"])
    if len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(f"Expected {EXPECTED_CELL_COUNT} fixed-N cells, got {len(cells)}")
    return tuple(cells)


def _common_coordinates(cells: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Choose one coordinate-identical grid spanning all prior fixed-N surfaces."""
    points: dict[tuple[float, float], dict[str, Any]] = {}

    def add_point(phase_0: float, phase_1: float, source: str) -> None:
        key = _coordinate_key(phase_0, phase_1)
        if key not in points:
            points[key] = {
                "phase_0_starcoder": float(phase_0),
                "phase_1_starcoder": float(phase_1),
                "sources": [],
            }
        points[key]["sources"].append(source)

    for phase_0, phase_1 in base.SURFACE_COORDINATES:
        add_point(phase_0, phase_1, "original_64_surface")
    for phase_0, phase_1 in CALIBRATION_COORDINATES:
        add_point(phase_0, phase_1, "calibration_control")
    for cell in cells:
        for row in cell["prior_rows"]:
            add_point(
                float(row["phase_0_starcoder"]),
                float(row["phase_1_starcoder"]),
                f"{cell['cell_id']}:{row['source_stage']}",
            )

    selected = {_coordinate_key(*point) for point in base.SURFACE_COORDINATES}
    selected.update(_coordinate_key(*point) for point in CALIBRATION_COORDINATES)
    for cell in cells:
        prior = tuple(cell["prior_rows"])
        best_tied = min(
            (row for row in prior if row["policy_class"] == "tied"),
            key=lambda row: float(row["starcoder_bpb"]),
        )
        best_untied = min(
            (row for row in prior if row["policy_class"] == "untied"),
            key=lambda row: float(row["starcoder_bpb"]),
        )
        # Resolve the two policy classes symmetrically around their prior minima.
        for anchor, neighbor_count in ((best_tied, 8), (best_untied, 8)):
            anchor_0 = float(anchor["phase_0_starcoder"])
            anchor_1 = float(anchor["phase_1_starcoder"])
            nearest = sorted(
                prior,
                key=lambda row: (
                    (float(row["phase_0_starcoder"]) - anchor_0) ** 2
                    + (float(row["phase_1_starcoder"]) - anchor_1) ** 2,
                    float(row["phase_0_starcoder"]),
                    float(row["phase_1_starcoder"]),
                ),
            )[: neighbor_count + 1]
            selected.update(
                _coordinate_key(float(row["phase_0_starcoder"]), float(row["phase_1_starcoder"])) for row in nearest
            )

    if len(selected) > EXPECTED_COORDINATES_PER_CELL:
        raise ValueError(f"Forced common grid exceeds {EXPECTED_COORDINATES_PER_CELL} coordinates: {len(selected)}")
    remaining = set(points) - selected
    while len(selected) < EXPECTED_COORDINATES_PER_CELL:
        if not remaining:
            raise ValueError("Candidate union cannot fill the common coordinate grid")
        candidate = max(
            remaining,
            key=lambda point: (
                min((point[0] - chosen[0]) ** 2 + (point[1] - chosen[1]) ** 2 for chosen in selected),
                -point[0],
                -point[1],
            ),
        )
        selected.add(candidate)
        remaining.remove(candidate)

    ordered = [points[key] for key in sorted(selected)]
    coordinates: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        phase_0 = float(item["phase_0_starcoder"])
        phase_1 = float(item["phase_1_starcoder"])
        coordinates.append(
            {
                "coordinate_id": f"c{index:03d}",
                "phase_0_starcoder": phase_0,
                "phase_1_starcoder": phase_1,
                "aggregate_starcoder": float(PHASE_0_FRACTION) * phase_0 + float(PHASE_1_FRACTION) * phase_1,
                "phase_contrast": phase_1 - phase_0,
                "policy_role": _coordinate_role(phase_0, phase_1),
                "sources": sorted(set(item["sources"])),
            }
        )
    return tuple(coordinates)


def _support_row(cell: dict[str, Any], support: SupportSpec) -> dict[str, Any]:
    multiplier = support.epoch_multiplier
    if multiplier is None:
        support_batches = None
        realized_support_tokens = STARCODER_SOURCE_TOKENS
    else:
        numerator = STARCODER_SOURCE_TOKENS * cell["materialized_tokens"] * multiplier.denominator
        denominator = base.TARGET_BUDGET * multiplier.numerator * TOKENS_PER_BATCH
        support_batches = max(1, numerator // denominator)
        realized_support_tokens = support_batches * TOKENS_PER_BATCH
        if realized_support_tokens >= STARCODER_SOURCE_TOKENS:
            raise ValueError(f"{cell['cell_id']}, {support.support_id}: finite support unexpectedly reaches full cache")
    return {
        "cell_slug": cell["cell_slug"],
        **asdict(support),
        "epoch_multiplier": None if multiplier is None else float(multiplier),
        "starcoder_support_batches": support_batches,
        "starcoder_realized_support_tokens": realized_support_tokens,
        "starcoder_support_fraction": realized_support_tokens / STARCODER_SOURCE_TOKENS,
    }


def _phase_component_weights(starcoder_weight: float) -> tuple[float, ...]:
    total_nemotron_tokens = sum(base.NEMOTRON_TOKEN_COUNTS.values())
    broad_weight = 1.0 - starcoder_weight
    broad_weights = tuple(
        broad_weight * token_count / total_nemotron_tokens for token_count in base.NEMOTRON_TOKEN_COUNTS.values()
    )
    return (*broad_weights, starcoder_weight)


def _realized_component_counts_per_block(
    phase_weights: tuple[float, ...],
    active_indices: tuple[int, ...],
) -> np.ndarray:
    """Match MixtureDataset normalization, truncation, and remainder allocation."""
    active_weights = np.asarray([phase_weights[index] for index in active_indices], dtype=np.float64)
    active_weights /= active_weights.sum()
    counts = np.asarray(active_weights * base.MIXTURE_BLOCK_SIZE, dtype=np.int32)
    counts[int(np.argmax(counts))] += base.MIXTURE_BLOCK_SIZE - int(counts.sum())
    return counts


def _partial_block_starcoder_count(
    *,
    counts: np.ndarray,
    starcoder_active_index: int | None,
    block_id: int,
    partial_size: int,
    mix_key: jax.Array,
) -> int:
    if partial_size == 0 or starcoder_active_index is None:
        return 0
    base_ids = np.empty(base.MIXTURE_BLOCK_SIZE, dtype=np.int64)
    start = 0
    for dataset_id, count in enumerate(counts):
        stop = start + int(count)
        base_ids[start:stop] = (dataset_id << 16) + np.arange(int(count), dtype=np.int64)
        start = stop
    permutation_key = jax.random.fold_in(mix_key, block_id)
    permuted_ids = np.asarray(jax.random.permutation(permutation_key, base_ids))
    dataset_ids = permuted_ids[:partial_size] >> 16
    return int(np.count_nonzero(dataset_ids == starcoder_active_index))


def _realized_starcoder_sequences(
    *,
    cell: dict[str, Any],
    phase_0: float,
    phase_1: float,
    data_seed: int,
) -> tuple[int, int]:
    """Count exact StarCoder draws, including the randomized trailing partial block."""
    phase_weights = (_phase_component_weights(phase_0), _phase_component_weights(phase_1))
    active_indices = tuple(
        index for index in range(len(phase_weights[0])) if any(weights[index] > 0 for weights in phase_weights)
    )
    starcoder_global_index = len(phase_weights[0]) - 1
    starcoder_active_index = (
        active_indices.index(starcoder_global_index) if starcoder_global_index in active_indices else None
    )
    phase_counts = tuple(_realized_component_counts_per_block(weights, active_indices) for weights in phase_weights)
    starcoder_per_block = tuple(
        0 if starcoder_active_index is None else int(counts[starcoder_active_index]) for counts in phase_counts
    )

    phase_0_sequences = cell["boundary_step"] * base.BATCH_SIZE
    if phase_0_sequences % base.MIXTURE_BLOCK_SIZE != 0:
        raise ValueError(f"{cell['cell_id']}: phase boundary is not mixture-block aligned")
    phase_0_blocks = phase_0_sequences // base.MIXTURE_BLOCK_SIZE
    phase_1_sequences = (cell["total_steps"] - cell["boundary_step"]) * base.BATCH_SIZE
    phase_1_blocks, phase_1_remainder = divmod(phase_1_sequences, base.MIXTURE_BLOCK_SIZE)
    mix_key, _ = jax.random.split(jax.random.PRNGKey(data_seed))
    partial_count = _partial_block_starcoder_count(
        counts=phase_counts[1],
        starcoder_active_index=starcoder_active_index,
        block_id=phase_0_blocks + phase_1_blocks,
        partial_size=phase_1_remainder,
        mix_key=mix_key,
    )
    return (
        phase_0_blocks * starcoder_per_block[0],
        phase_1_blocks * starcoder_per_block[1] + partial_count,
    )


def _run_row(
    *,
    cell: dict[str, Any],
    support: dict[str, Any],
    coordinate: dict[str, Any],
    seed: int,
    replicate_kind: str,
) -> dict[str, Any]:
    phase_0 = float(coordinate["phase_0_starcoder"])
    phase_1 = float(coordinate["phase_1_starcoder"])
    support_tokens = int(support["starcoder_realized_support_tokens"])
    phase_0_sequences, phase_1_sequences = _realized_starcoder_sequences(
        cell=cell,
        phase_0=phase_0,
        phase_1=phase_1,
        data_seed=seed,
    )
    support_sequences = support_tokens // base.SEQ_LEN
    total_starcoder_sequences = phase_0_sequences + phase_1_sequences
    seed_slug = str(seed)[-4:]
    run_name = f"dss_{cell['cell_slug']}_{support['support_id']}_{coordinate['coordinate_id']}_s{seed_slug}"
    return {
        "run_name": run_name,
        "cell_id": cell["cell_id"],
        "cell_slug": cell["cell_slug"],
        "rung": cell["rung"],
        "hidden_size": cell["hidden_size"],
        "total_steps": cell["total_steps"],
        "boundary_step": cell["boundary_step"],
        "materialized_tokens": cell["materialized_tokens"],
        "total_parameters": cell["total_parameters"],
        "non_embedding_parameters": cell["non_embedding_parameters"],
        "support_id": support["support_id"],
        "support_role": support["role"],
        "epoch_multiplier": support["epoch_multiplier"],
        "starcoder_support_batches": support["starcoder_support_batches"],
        "starcoder_realized_support_tokens": support_tokens,
        "starcoder_support_fraction": support["starcoder_support_fraction"],
        "coordinate_id": coordinate["coordinate_id"],
        "policy_role": coordinate["policy_role"],
        "coordinate_sources": coordinate["sources"],
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder": coordinate["aggregate_starcoder"],
        "phase_contrast": coordinate["phase_contrast"],
        "starcoder_phase_0_sequences": phase_0_sequences,
        "starcoder_phase_1_sequences": phase_1_sequences,
        "starcoder_total_sequences": total_starcoder_sequences,
        "starcoder_phase_0_epochs": phase_0_sequences / support_sequences,
        "starcoder_phase_1_epochs": phase_1_sequences / support_sequences,
        "starcoder_support_wraps": total_starcoder_sequences > support_sequences,
        "nemotron_max_total_epochs": cell["materialized_tokens"] / NEMOTRON_SOURCE_TOKENS,
        "data_seed": seed,
        "replicate_kind": replicate_kind,
    }


def build_payload() -> dict[str, Any]:
    """Build and validate the complete immutable design."""
    if jax.__version__ != DESIGN_JAX_VERSION or np.__version__ != DESIGN_NUMPY_VERSION:
        raise ValueError(f"Design dependency drift: jax={jax.__version__}, numpy={np.__version__}")
    if jax.config.jax_default_prng_impl != DESIGN_JAX_DEFAULT_PRNG_IMPL:
        raise ValueError(f"Design PRNG drift: {jax.config.jax_default_prng_impl}")
    if bool(jax.config.jax_enable_x64) != DESIGN_JAX_ENABLE_X64:
        raise ValueError(f"Design x64 mode drift: {jax.config.jax_enable_x64}")
    observed_uv_lock_sha256 = file_sha256(SCRIPT_DIR.parents[3] / "uv.lock")
    if observed_uv_lock_sha256 != UV_LOCK_SHA256:
        logger.warning(
            "uv.lock drifted from the frozen design provenance: %s != %s",
            observed_uv_lock_sha256,
            UV_LOCK_SHA256,
        )
    cells = _load_fixed_n_cells()
    cell_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    complete_rows: list[dict[str, Any]] = []
    coordinates = _common_coordinates(cells)
    coordinate_rows.extend(coordinates)
    coordinate_by_key = {
        _coordinate_key(coordinate["phase_0_starcoder"], coordinate["phase_1_starcoder"]): coordinate
        for coordinate in coordinates
    }
    missing_calibration = [
        point for point in CALIBRATION_COORDINATES if _coordinate_key(*point) not in coordinate_by_key
    ]
    if missing_calibration:
        raise ValueError(f"Common grid is missing calibration coordinates {missing_calibration}")

    for cell in cells:
        public_cell = {key: value for key, value in cell.items() if key != "prior_rows"}
        cell_rows.append(public_cell)
        supports = tuple(_support_row(cell, support_spec) for support_spec in SUPPORT_SPECS)
        support_rows.extend({"cell_id": cell["cell_id"], **support} for support in supports)
        for coordinate in coordinates:
            for support in supports:
                complete_rows.append(
                    _run_row(
                        cell=cell,
                        support=support,
                        coordinate=coordinate,
                        seed=REFERENCE_SEED,
                        replicate_kind="coverage",
                    )
                )
        for support in supports:
            for point in CALIBRATION_COORDINATES:
                coordinate = coordinate_by_key[_coordinate_key(*point)]
                for seed in REPEAT_SEEDS:
                    complete_rows.append(
                        _run_row(
                            cell=cell,
                            support=support,
                            coordinate=coordinate,
                            seed=seed,
                            replicate_kind="calibration_repeat",
                        )
                    )

    if len(complete_rows) != EXPECTED_COMPLETE_ROWS:
        raise ValueError(f"Expected {EXPECTED_COMPLETE_ROWS} complete rows, got {len(complete_rows)}")
    full_by_identity = {
        (row["cell_id"], row["coordinate_id"], row["data_seed"], row["replicate_kind"]): row
        for row in complete_rows
        if row["support_id"] == "full"
    }
    runs: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for row in complete_rows:
        if row["support_id"] != "full" and not row["starcoder_support_wraps"]:
            identity = (row["cell_id"], row["coordinate_id"], row["data_seed"], row["replicate_kind"])
            source = full_by_identity[identity]
            aliases.append(
                {
                    **row,
                    "alias_of_run_name": source["run_name"],
                    "alias_reason": "finite_support_not_exhausted_same_shuffled_prefix",
                }
            )
        else:
            runs.append(row)

    support_order = {support.support_id: index for index, support in enumerate(SUPPORT_SPECS)}
    replicate_order = {"coverage": 0, "calibration_repeat": 1}

    def row_order(row: dict[str, Any]) -> tuple[int, int, int, str, int]:
        return (
            replicate_order[row["replicate_kind"]],
            row["rung"],
            support_order[row["support_id"]],
            row["coordinate_id"],
            row["data_seed"],
        )

    runs.sort(key=row_order)
    aliases.sort(key=row_order)

    if len(runs) != EXPECTED_RUN_COUNT or len(aliases) != EXPECTED_ALIAS_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_RUN_COUNT} launched runs and {EXPECTED_ALIAS_ROWS} aliases, "
            f"got {len(runs)} and {len(aliases)}"
        )
    if len({row["run_name"] for row in runs}) != len(runs):
        raise ValueError("Run names are not unique")
    if sum(row["replicate_kind"] == "coverage" for row in runs) != EXPECTED_PRIMARY_RUNS:
        raise ValueError("Coverage rows drifted")
    if sum(row["replicate_kind"] == "calibration_repeat" for row in runs) != EXPECTED_REPEAT_RUNS:
        raise ValueError("Calibration-repeat rows drifted")
    for cell in cells:
        cell_aliases = sum(row["cell_id"] == cell["cell_id"] for row in aliases)
        if cell_aliases * EXPECTED_CELL_COUNT != EXPECTED_ALIAS_ROWS:
            raise ValueError(f"{cell['cell_id']}: aliases are not balanced across token horizons")

    alias_names = {row["run_name"] for row in aliases}
    all_seeds = (REFERENCE_SEED, *REPEAT_SEEDS)
    calibration_alias_groups = 0
    for cell in cells:
        for support in SUPPORT_SPECS:
            for point in CALIBRATION_COORDINATES:
                coordinate = coordinate_by_key[_coordinate_key(*point)]
                statuses = []
                for seed in all_seeds:
                    seed_slug = str(seed)[-4:]
                    run_name = f"dss_{cell['cell_slug']}_{support.support_id}_{coordinate['coordinate_id']}_s{seed_slug}"
                    statuses.append(run_name in alias_names)
                if len(set(statuses)) != 1:
                    raise ValueError(
                        f"Calibration alias status differs across aligned seeds: "
                        f"{cell['cell_id']}, {support.support_id}, {coordinate['coordinate_id']}"
                    )
                calibration_alias_groups += int(statuses[0])
    complete_calibration_groups = EXPECTED_CELL_COUNT * EXPECTED_SUPPORT_COUNT * len(CALIBRATION_COORDINATES)
    unique_calibration_groups = complete_calibration_groups - calibration_alias_groups

    wrap_active_coverage_counts = [
        {
            "cell_id": cell["cell_id"],
            "support_id": support.support_id,
            "wrap_active": sum(
                row["cell_id"] == cell["cell_id"]
                and row["support_id"] == support.support_id
                and row["replicate_kind"] == "coverage"
                and row["starcoder_support_wraps"]
                for row in complete_rows
            ),
            "structural_zero": sum(
                row["cell_id"] == cell["cell_id"]
                and row["support_id"] == support.support_id
                and row["replicate_kind"] == "coverage"
                and not row["starcoder_support_wraps"]
                for row in complete_rows
            ),
        }
        for cell in cells
        for support in SUPPORT_SPECS
        if support.support_id != "full"
    ]

    return {
        "design_version": DESIGN_VERSION,
        "description": "Dense fixed-N WSD80 surfaces crossed with StarCoder-only unique-support interventions.",
        "source_observations": str(OBSERVATIONS_PATH.relative_to(SCRIPT_DIR.parents[3])),
        "source_observations_sha256": file_sha256(OBSERVATIONS_PATH),
        "phase_0_fraction": float(PHASE_0_FRACTION),
        "sequence_length": base.SEQ_LEN,
        "batch_size": base.BATCH_SIZE,
        "tokens_per_batch": TOKENS_PER_BATCH,
        "starcoder_source_tokens": STARCODER_SOURCE_TOKENS,
        "starcoder_source_token_provenance": STARCODER_SOURCE_TOKEN_PROVENANCE,
        "runtime_cache_contract": {
            "relative_path": STARCODER_CACHE_RELATIVE_PATH,
            "document_count": STARCODER_CACHE_DOCUMENTS,
            "shard_count": STARCODER_CACHE_SHARDS,
            "layout": STARCODER_CACHE_LAYOUT,
            "tokenizer_metadata": STARCODER_CACHE_TOKENIZER_METADATA,
            "legacy_token_count_policy": (
                "the pinned cache predates train/.stats.json; validate exact path, completion, document count, "
                "shards, layout, and tokenizer metadata at parent startup; retain the frozen token count from "
                "the domain registry as provenance rather than reconstructing it from document count"
            ),
        },
        "design_environment": {
            "jax_version": DESIGN_JAX_VERSION,
            "numpy_version": DESIGN_NUMPY_VERSION,
            "jax_default_prng_impl": DESIGN_JAX_DEFAULT_PRNG_IMPL,
            "jax_enable_x64": DESIGN_JAX_ENABLE_X64,
            "uv_lock_sha256": UV_LOCK_SHA256,
        },
        "training_environment": {
            "jax_version": TRAINING_JAX_VERSION,
            "numpy_version": TRAINING_NUMPY_VERSION,
            "jax_default_prng_impl": TRAINING_JAX_DEFAULT_PRNG_IMPL,
            "jax_enable_x64": TRAINING_JAX_ENABLE_X64,
        },
        "nemotron_source_tokens": NEMOTRON_SOURCE_TOKENS,
        "historical_target_budget": base.TARGET_BUDGET,
        "reference_seed": REFERENCE_SEED,
        "repeat_seeds": list(REPEAT_SEEDS),
        "cell_count": len(cell_rows),
        "support_count": len(SUPPORT_SPECS),
        "coordinates_per_cell": EXPECTED_COORDINATES_PER_CELL,
        "primary_run_count": EXPECTED_PRIMARY_RUNS,
        "repeat_run_count": EXPECTED_REPEAT_RUNS,
        "complete_primary_row_count": EXPECTED_COMPLETE_PRIMARY_ROWS,
        "complete_repeat_row_count": EXPECTED_COMPLETE_REPEAT_ROWS,
        "deterministic_alias_count": len(aliases),
        "calibration_alias_group_count": calibration_alias_groups,
        "unique_calibration_variance_group_count": unique_calibration_groups,
        "expected_run_count": len(runs),
        "cells": cell_rows,
        "coordinates": coordinate_rows,
        "supports": support_rows,
        "wrap_active_coverage_counts": wrap_active_coverage_counts,
        "runs": runs,
        "deterministic_aliases": aliases,
        "analysis_contract": {
            "primary_target": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
            "primary_metric_read_point": (
                "forced final evaluation at num_train_steps; select the metric record whose training step equals "
                "the row's num_train_steps, never the W&B summary or merely latest record; duplicate exact-step "
                "records must agree numerically or the row is unresolved"
            ),
            "missing_metric_policy": {
                "collection": "retry every launched row until the exact-step primary metric is durable or irrecoverable",
                "imputation": "none",
                "observed_definition": (
                    "a coordinate is observed in a block when its launched row has an exact-step primary metric, or "
                    "when it is a deterministic_aliases row whose alias_of_run_name source has that metric; exact "
                    "mechanism aliases are not statistical imputations"
                ),
                "completeness_denominator": (
                    "all_125_primary_coordinates_after_resolving_deterministic_aliases; aliases count toward "
                    "completeness but remain excluded from finite-support replay slopes and interactions"
                ),
                "within_block_surface": (
                    "fit only observed primary coverage coordinates in that block and report every missing coordinate"
                ),
                "coordinate_matched_contrasts": (
                    "use the global intersection of coordinates observed in every included block"
                ),
                "minimum_block_completeness": "120_of_125_primary_coverage_coordinates",
                "minimum_global_matched_coordinates": 100,
                "below_threshold": (
                    "report the block but exclude it from pooled slopes, interactions, optimization claims, and the "
                    "calibration variance model"
                ),
                "global_below_threshold": (
                    "report the achieved global coordinate intersection, mark coordinate-matched contrasts as "
                    "underpowered, and issue no optimization claims from those contrasts"
                ),
                "calibration_group_minimum": (
                    "at least three exact-step seed metrics per distinct cell-support-coordinate group; deduplicate "
                    "finite-support aliases to their full-pool source before fitting the variance model"
                ),
            },
            "primary_estimands": [
                "policy_class_gap_within_cell",
                "token_horizon_effect_at_fixed_finite_epoch_multiplier",
                "epoch_multiplier_effect_at_fixed_token_horizon",
                "token_horizon_by_log_epoch_multiplier_interaction",
                "physical_full_pool_path",
            ],
            "raw_minima_are_descriptive": True,
            "surface_estimator": {
                "basis": "all aggregate_and_raw_contrast_monomials_through_total_degree_four",
                "aggregate_normalization": "x=(aggregate-0.5)/0.5",
                "ridge_grid": [10 ** (-6 + index / 2) for index in range(17)],
                "intercept_penalized": False,
                "ridge_selection": "deterministic_five_fold_spatial_cv_minimum_weighted_rmse",
                "spatial_fold": "(floor(5*p0)+2*floor(5*p1)) mod 5",
                "fit_weights": "inverse_variance_from_frozen_calibration_variance_model",
                "weight_ratio_cap": 100.0,
            },
            "optimization": {
                "domain": "closed_empirical_convex_hull_including_boundary",
                "axis_grid_size": 197,
                "policy_interior_bound": 0.0,
                "minimum_untied_absolute_contrast": 0.04,
                "tied_comparator": "same_fitted_surface_restricted_to_diagonal",
                "report_argmin_nearest_design_coordinate_distance": True,
            },
            "uncertainty": {
                "method": "leverage_corrected_rademacher_wild_bootstrap_with_frozen_selected_ridge_and_weights",
                "replicates": 500,
                "seed": "20260808 + first_8_hex_sha256(cell_id:support_id)",
                "variance_model": {
                    "observations": (
                        "sample_variance_across_four_aligned_seeds at eight calibration coordinates; structural "
                        "aliases are deduplicated to 188 distinct cell-support-coordinate groups before fitting"
                    ),
                    "formula": "log_variance=block_intercept+b1*aggregate+b2*aggregate^2+b3*abs(contrast)+b4*contrast^2",
                    "shape_ridge": 1.0,
                    "block_intercepts_penalized": False,
                    "variance_floor": 1e-8,
                    "fit_scope": (
                        f"{unique_calibration_groups}_distinct_cell_support_coordinate_groups_pooled_after_"
                        "deduplicating_structural_aliases"
                    ),
                },
                "cross_block_seed_component": {
                    "estimator": "paired_aligned_seed_contrasts_averaged_over_the_eight_calibration_coordinates",
                    "seed_count": 4,
                    "use": "add_empirical_block_level_seed_uncertainty_to_cross_D_and_cross_support_estimands",
                    "within_block_policy_gap": "unchanged_by_block_level_seed_offset",
                },
            },
            "fresh_confirmation_gate": {
                "minimum_fitted_gain_bpb": 0.005,
                "minimum_positive_gain_probability": 0.8,
            },
            "selection_confirmation_required": True,
            "calibration_repeats_may_not_select_models_or_optima_or_estimate_mean_response": True,
            "calibration_repeats_may_only_estimate_variance_and_seed_nuisance": True,
            "cross_treatment_analysis": {
                "D_effect_coordinates": "all_125_common_coordinates",
                "finite_support_primary_region": (
                    "realized_wrap_active_rows_only; for each contrast, require StarCoder consumption to exceed "
                    "the smaller compared finite support using integer MixtureDataset block allocation"
                ),
                "finite_support_structural_zero_region": (
                    "report no-wrap counts and exact zero differences separately; exclude these aliases from "
                    "epoch-multiplier slopes and D-by-epoch-multiplier interaction estimates"
                ),
                "report_live_coordinate_counts_per_cell_support": True,
                "cross_support_surface_comparisons": (
                    "report the exact shared-coordinate fraction and compare surface-derived quantities only on "
                    "wrap-active coordinates common to the support pair"
                ),
                "finite_support_basis": "log_materialized_tokens_and_log_epoch_multiplier",
                "full_pool_path_analyzed_separately": True,
                "report": [
                    "coordinate_matched_D_contrasts_at_fixed_epoch_multiplier",
                    "coordinate_matched_epoch_multiplier_contrasts_at_fixed_D",
                    "D_by_epoch_multiplier_interaction",
                    "surface_selected_policy_class_gap_per_cell",
                ],
                "identifiability_boundary": (
                    "At finite support, unique support, replay epochs, and D obey support*epochs proportional to D; "
                    "the experiment identifies the chosen log-D/log-m basis, not three independent effects."
                ),
                "scope_boundary": (
                    "Only StarCoder support is capped; Nemotron always uses complete physical caches. A null effect "
                    "therefore does not exclude replay effects in the broad pool or joint-source replay effects."
                ),
                "schedule_boundary": (
                    "MuonH learning-rate hyperparameters are fixed across the 7.4x D range; within-cell policy gaps "
                    "remain primary, while cross-D level effects include this fixed-schedule condition."
                ),
            },
            "checkpoint_retention": (
                "launcher disables permanent interval checkpoints and retains one forced final checkpoint per row; "
                "temporary preemption checkpoints remain region-local and rolling. Retain final checkpoints until "
                "primary metrics, manifests, and analysis are durably sealed; cleanup is separately reviewed"
            ),
            "execution_stages": (
                "stage_1 launches coverage only across all four cells; stage_2 calibration repeats are rejected by "
                "the launcher until every full-manifest coverage artifact reports SUCCESS"
            ),
        },
    }


def _write_artifacts(payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(payload["runs"][0])
    with MANIFEST_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["runs"]:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})

    support_lines = []
    for support_id in [item.support_id for item in SUPPORT_SPECS]:
        rows = [row for row in payload["supports"] if row["support_id"] == support_id]
        fractions = ", ".join(f"{row['cell_slug']}: {row['starcoder_support_fraction']:.6f}" for row in rows)
        support_lines.append(f"- `{support_id}`: {fractions}")
    wrap_lines = []
    for support_id in [item.support_id for item in SUPPORT_SPECS if item.support_id != "full"]:
        rows = [row for row in payload["wrap_active_coverage_counts"] if row["support_id"] == support_id]
        counts = ", ".join(
            f"{row['cell_id']}: {row['wrap_active']} live / {row['structural_zero']} structural-zero" for row in rows
        )
        wrap_lines.append(f"- `{support_id}`: {counts}")
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Dense WSD80 StarCoder-support surface design",
                "",
                f"Frozen design: `{payload['design_version']}`.",
                "",
                f"- Cells: `{payload['cell_count']}` fixed-N token horizons.",
                f"- Support regimes: `{payload['support_count']}`.",
                f"- Unique policy coordinates per cell: `{payload['coordinates_per_cell']}`.",
                f"- Coverage runs: `{payload['primary_run_count']}`.",
                f"- Calibration repeats: `{payload['repeat_run_count']}`.",
                f"- Deterministic no-wrap aliases: `{payload['deterministic_alias_count']}`.",
                f"- New checkpoints: `{payload['expected_run_count']}`.",
                "- Complete analysis rows: "
                f"`{payload['complete_primary_row_count'] + payload['complete_repeat_row_count']}`.",
                "- Nemotron always uses its complete physical caches; only StarCoder support is capped.",
                "- A finite-support row is aliased to its same-seed full-pool row only when total StarCoder "
                "consumption does not exceed one support epoch; both configurations then traverse the same shuffled "
                "prefix exactly.",
                "- Coverage is frozen before outcomes; calibration repeats cannot select optima.",
                "- Design coordinates were materialized with JAX 0.11.0; CPU/TPU execution is pinned to JAX 0.10.1. "
                "The full sequence-count identity audit must pass in the Iris training environment before launch.",
                "- The historical StarCoder cache predates `train/.stats.json`; parent startup validates its exact "
                "path, completed ledger, 49 shards, 206,640,114 documents, consolidated layout, and tokenizer "
                "metadata. The frozen 216,567,300,822-token count remains registry provenance, not a value inferred "
                "from document count.",
                "- Finite-support effects are estimated only where realized StarCoder consumption wraps the compared "
                "support. Exact no-wrap aliases are reported as structural-zero controls rather than averaged into "
                "the replay effect.",
                "- Heteroskedastic surface fits use calibration-derived inverse-variance weights and a "
                "leverage-corrected Rademacher wild bootstrap; aligned repeat seeds additionally quantify "
                "cross-block seed uncertainty.",
                "- Coverage and calibration are separate enforced launch stages; calibration cannot be built until "
                "every coverage artifact reports success.",
                "- Final checkpoints are retained until metrics and analysis are durably sealed; cleanup requires a "
                "separate reviewed operation.",
                "",
                "## StarCoder support fractions",
                "",
                *support_lines,
                "",
                "## Wrap-active coverage counts",
                "",
                *wrap_lines,
                "",
                "The physical-full regime is a separate D-dependent path. Finite-m regimes hold StarCoder epoch "
                "burden approximately invariant across D up to integral-batch rounding.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload()
    payload["design_sha256"] = canonical_sha256(payload)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_artifacts(payload)
    print(
        json.dumps(
            {
                "output_path": str(OUTPUT_PATH),
                "design_sha256": payload["design_sha256"],
                "expected_run_count": payload["expected_run_count"],
                "primary_run_count": payload["primary_run_count"],
                "repeat_run_count": payload["repeat_run_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
