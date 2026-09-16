# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "lightgbm==4.7.0"]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Freeze Figure 10 midpoint predictions before their prospective training runs.

Uses only the original 280-mixture fitting panel. Every refitted model must reproduce
all 61 positions in the archived figure, including its endpoints and exact midpoint.
Fitted components are checkpointed locally under their input/dependency fingerprint.
The script exits if materialized candidate files are absent; it never submits jobs.

Run one process per objective with the repository environment:
  uv run --offline --no-sync --with lightgbm==4.7.0 python -m \
    experiments.domain_phase_mix.exploratory.two_phase_many.predict_delphi_path_midpoints_20260912 \
    --target uncheatable
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_comparator_proposal_diagnostics_20260909 as paths,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_path_midpoints_3e18_20260912"
DEFAULT_FIGURE_CSV = Path(
    "/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/"
    "data_mixing_paper_one_phase/revision_notes/20260912_full_outline_revision/"
    "baseline_calibration/data/path_predictions.csv"
)
PARITY_ATOL = 1e-8
PARITY_RTOL = 1e-8
SAMPLER_BLOCK_SIZE = 2048
LOG = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def array_hash(array: np.ndarray) -> dict:
    values = np.ascontiguousarray(array)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def dependencies(panel: paths.bench.BenchPanel, candidates: Path, exact: Path, figure: Path) -> dict:
    modules = (
        paths,
        paths.bench,
        paths.bench.canonical,
        paths.bench.olmix_benchmark,
        paths.bench.single_phase,
        paths.proposals,
        paths.proposals.models,
        paths.registry,
        paths.proposals.ms,
    )
    sources = [Path(module.__file__).resolve() for module in modules]
    sources.append(Path(__file__).resolve())
    artifacts = [candidates, exact, figure, paths.PROPOSAL_DIR / "solutions.csv", paths.OLMIX_DIR / "solutions.csv"]
    artifacts.extend(paths.OLMIX_DIR / f"laws_{target}.json" for target, _ in paths.TARGETS)
    artifacts.extend(
        SCRIPT_DIR
        / "reference_outputs"
        / "delphi_frozen_procedure_validation_3e18_20260908"
        / "fits"
        / f"fit_{target}.json"
        for target, _ in paths.TARGETS
    )
    artifacts.extend(sorted(paths.proposals.ms.DATA.glob("*.csv")))
    return {
        "schema_version": 1,
        "source_sha256": {str(path): file_hash(path) for path in sources},
        "artifact_sha256": {str(path): file_hash(path) for path in artifacts},
        "panel_input_hashes": panel.input_hashes,
        "panel": {
            "name": panel.name,
            "runs": list(panel.runs),
            "buckets": list(panel.buckets),
            "weights": array_hash(panel.features.weights),
            "inventory": array_hash(panel.features.inventory),
            "outcomes": {target: array_hash(panel.group(target).outcomes) for target, _ in paths.TARGETS},
        },
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "lightgbm")
        },
        "python": sys.version,
        "parity_absolute_tolerance": PARITY_ATOL,
        "parity_relative_tolerance": PARITY_RTOL,
        "sampler_block_size": SAMPLER_BLOCK_SIZE,
    }


def cached_fit(path: Path, fit):
    if path.exists():
        LOG.info("Reusing %s", path.name)
        with path.open("rb") as stream:
            return pickle.load(stream)
    LOG.info("Fitting %s", path.name)
    result = fit()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return result


def fit_predictors(panel: paths.bench.BenchPanel, swarm, target: str, directory: Path) -> dict:
    inner = paths.bench.heldout_inner_folds(panel)
    predictors = {}
    for key, _, model_id in paths.PATH_COMPARATORS:
        if model_id == paths.OLMIX_MODEL:
            predictors[key] = paths.olmix_predictor(panel, target)
            continue
        fits = [
            cached_fit(
                directory / f"{target}_{key}_{index:02}.pkl",
                lambda index=index, model_id=model_id: paths.proposals.fit_component(
                    panel, model_id, target, index, inner
                ),
            )
            for index in range(len(panel.group(target).components))
        ]
        predictors[key] = paths.proposals.ObservatorySurrogate(panel, model_id, target, fits).predict
    reference_path = (
        SCRIPT_DIR
        / "reference_outputs"
        / "delphi_frozen_procedure_validation_3e18_20260908"
        / "fits"
        / f"fit_{target}.json"
    )
    reference = paths.proposals.ms.ObjectiveFit.from_json(json.loads(reference_path.read_text()))
    write_json(directory / f"{target}_mariner.json", reference.to_json())
    to_panel = np.asarray([swarm.buckets.index(bucket) for bucket in panel.buckets])
    from_panel = np.argsort(to_panel)
    predictors["mariner"] = lambda rows: reference.predict(np.atleast_2d(rows)[:, from_panel])
    return predictors


def candidate_mixtures(exact: pd.DataFrame, panel: paths.bench.BenchPanel, weights: dict, target: str) -> list[dict]:
    rows = []
    selected = exact.loc[exact.target.eq(target)]
    expected = {key for key, _, _ in paths.PATH_COMPARATORS}
    if set(selected.baseline) != expected or selected.candidate_id.nunique() != len(expected):
        raise ValueError(f"{target}: expected exactly the five Figure 10 columns")
    for candidate_id, block in selected.groupby("candidate_id", sort=True):
        if len(block) != len(panel.buckets) or block.domain.duplicated().any():
            raise ValueError(f"{candidate_id}: each bucket must occur exactly once")
        block = block.set_index("domain").loc[list(panel.buckets)]
        key = str(block.baseline.iloc[0])
        start = weights[target]["mariner"]
        end = weights[target][key]
        if not np.allclose(block.mariner_weight, start, atol=1e-14, rtol=0):
            raise ValueError(f"{candidate_id}: MARINER endpoint changed")
        if not np.allclose(block.baseline_weight, end, atol=1e-14, rtol=0):
            raise ValueError(f"{candidate_id}: baseline endpoint changed")
        exact_w = block.exact_weight.to_numpy(float)
        runtime_w = block.runtime_weight.to_numpy(float)
        if not np.allclose(exact_w, 0.5 * (start + end), atol=1e-14, rtol=0):
            raise ValueError(f"{candidate_id}: weights are not the exact 50% midpoint")
        for mixture in (exact_w, runtime_w):
            if np.any(mixture < 0) or abs(mixture.sum() - 1) > 1e-12:
                raise ValueError(f"{candidate_id}: invalid simplex mixture")
        if not np.allclose(runtime_w * SAMPLER_BLOCK_SIZE, np.rint(runtime_w * SAMPLER_BLOCK_SIZE), atol=1e-12, rtol=0):
            raise ValueError(f"{candidate_id}: runtime mixture is not on the original sampler grid")
        rows.append({"candidate_id": candidate_id, "baseline": key, "exact": exact_w, "runtime": runtime_w})
    return rows


def freeze_predictions(target: str, output: Path, figure: Path) -> None:
    candidate_path = output / "candidate_weights.csv"
    exact_path = output / "exact_midpoints.csv"
    for path in (candidate_path, exact_path, figure):
        if not path.is_file():
            raise FileNotFoundError(f"Required materialized input does not exist: {path}")
    if importlib.metadata.version("lightgbm") != "4.7.0":
        raise ValueError("Use the audited LightGBM 4.7.0 environment")
    panel = paths.bench.load_panel(paths.proposals.PANEL)
    if panel.rows != 280:
        raise ValueError(f"Expected the original 280-mixture panel, got {panel.rows}")
    weights = paths.proposal_weights(tuple(panel.buckets))
    candidates = candidate_mixtures(pd.read_csv(exact_path), panel, weights, target)
    materialized = pd.read_csv(candidate_path)
    for candidate in candidates:
        block = materialized.loc[materialized.candidate_id.eq(candidate["candidate_id"])].set_index("domain")
        block = block.loc[list(panel.buckets)]
        if not np.array_equal(block.weight.to_numpy(float), candidate["runtime"]):
            raise ValueError(f"{candidate['candidate_id']}: candidate CSV differs from frozen runtime weights")
        if not np.array_equal(
            block.runtime_count.to_numpy(int), np.rint(candidate["runtime"] * SAMPLER_BLOCK_SIZE).astype(int)
        ):
            raise ValueError(f"{candidate['candidate_id']}: candidate runtime counts differ")
    provenance = dependencies(panel, candidate_path, exact_path, figure)
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    directory = output / "prediction_freeze" / fingerprint / target
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "dependencies.json", provenance)
    LOG.info("Prediction fingerprint: %s", fingerprint)
    swarm, _ = paths.proposals.reference_swarm_in_panel_order(panel)
    predictors = fit_predictors(panel, swarm, target, directory)
    archived = pd.read_csv(figure)
    archived = archived.loc[archived.target.eq(target)]
    parity_rows = []
    predictions = []
    for candidate in candidates:
        key = candidate["baseline"]
        positions, mixtures = paths.path_mixtures(weights[target]["mariner"], weights[target][key])
        if len(positions) != 61 or not np.any(np.isclose(positions, 0.5, rtol=0, atol=1e-15)):
            raise ValueError("Expected 61 path points including the midpoint")
        for predictor in ("mariner", key):
            original = archived.loc[archived.comparator.eq(key) & archived.predictor.eq(predictor)].sort_values(
                "position"
            )
            if len(original) != 61 or not np.allclose(original.position, positions, atol=1e-14, rtol=0):
                raise ValueError(f"{target}/{key}/{predictor}: archived path positions differ")
            values = predictors[predictor](mixtures)
            for position, predicted, frozen in zip(positions, values, original.prediction, strict=True):
                parity_rows.append(
                    {
                        "target": target,
                        "baseline": key,
                        "predictor": predictor,
                        "position": float(position),
                        "frozen_prediction_bpb": float(frozen),
                        "refit_prediction_bpb": float(predicted),
                        "absolute_difference_bpb": abs(float(predicted - frozen)),
                        "pass": bool(np.isclose(predicted, frozen, atol=PARITY_ATOL, rtol=PARITY_RTOL)),
                    }
                )
            exact_value = float(predictors[predictor](candidate["exact"][None])[0])
            runtime_value = float(predictors[predictor](candidate["runtime"][None])[0])
            for coordinate, value in (("exact", exact_value), ("runtime", runtime_value)):
                predictions.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "target": target,
                        "baseline": key,
                        "predictor": predictor,
                        "coordinate": coordinate,
                        "prediction_bpb": value,
                        "rounding_change_bpb": runtime_value - exact_value,
                        "rounding_total_variation": float(0.5 * np.abs(candidate["runtime"] - candidate["exact"]).sum()),
                    }
                )
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(directory / "figure_path_parity.csv", index=False)
    pd.DataFrame(predictions).to_csv(directory / "midpoint_predictions.csv", index=False)
    passed = bool(parity["pass"].all())
    receipt = {
        "target": target,
        "fingerprint": fingerprint,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "pass": passed,
        "figure_path_points_checked": len(parity),
        "max_figure_path_difference_bpb": float(parity.absolute_difference_bpb.max()),
        "prediction_rows": len(predictions),
        "checkpoint_sha256": {path.name: file_hash(path) for path in sorted(directory.glob("*.pkl"))},
        "midpoint_predictions_sha256": file_hash(directory / "midpoint_predictions.csv"),
        "figure_path_parity_sha256": file_hash(directory / "figure_path_parity.csv"),
        "dependencies_sha256": file_hash(directory / "dependencies.json"),
        "training_data": "Original frozen 280-mixture swarm only; no midpoint outcomes read.",
    }
    write_json(directory / "validation.json", receipt)
    if not passed:
        raise ValueError(f"Frozen Figure 10 parity failed; inspect {directory / 'figure_path_parity.csv'}")
    write_json(output / f"prediction_freeze_{target}.json", {**receipt, "directory": str(directory)})
    LOG.info(
        "PASS: %s, %d path values, maximum difference %.3g BPB",
        target,
        len(parity),
        receipt["max_figure_path_difference_bpb"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=[target for target, _ in paths.TARGETS], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-csv", type=Path, default=DEFAULT_FIGURE_CSV)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    freeze_predictions(args.target, args.output_dir.resolve(), args.figure_csv.resolve())


if __name__ == "__main__":
    main()
