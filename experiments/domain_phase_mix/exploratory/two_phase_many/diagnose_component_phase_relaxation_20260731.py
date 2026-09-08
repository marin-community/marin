# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Test a component-observed phase-local relaxation law.

The 300M paired panel logs seven Uncheatable component BPBs throughout
training. For each aggregate-matched asymmetric/tied pair, this diagnostic
uses only the pre-switch component difference and component trajectories ending
by step 21,000 to fit a shared phase-local relaxation law. Step 22,000, the
final endpoint, aggregate Uncheatable BPB, Table-9, and WSD80 do not select any
parameter.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "component_phase_relaxation_20260731"
SOURCE_HISTORY = REFERENCE_OUTPUTS / "tied_two_phase_trajectory_audit_20260726" / "wandb_histories.csv"
WANDB_PATH = "marin-community/marin"

CANDIDATE_ID = "WSD80-SUR-068"
VERSION = "component-phase-relaxation-v3"
HISTORY_CACHE_NAME = "component_histories_scan.csv"
AVAILABILITY_MANIFEST_NAME = "availability_manifest.csv"
EXPECTED_PAIRS = 238
MIN_FIT_PAIRS = 180
MIN_STEP_22000_PAIRS = 180
MIN_FINAL_PAIRS = 200
PHASE_BOUNDARY_STEP = 18_310
FINAL_STEP = 22_887
PRE_SWITCH_STEPS = (17_000, 18_000)
FIT_STEPS = (19_000, 20_000, 21_000)
HOLDOUT_STEPS = (22_000, FINAL_STEP)
PHASE_RATIO = 4.0
COMPONENT_KEYS = (
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)
AGGREGATE_KEY = "eval/uncheatable_eval/bpb"
ALL_TARGET_KEYS = (*COMPONENT_KEYS, AGGREGATE_KEY)
GAMMA_BOUNDS = (0.0, 4.0)
RATE_BOUNDS = (0.05, 20.0)
OUTER_FOLDS = 5
FOLD_SEED = 7316801
BOOTSTRAP_DRAWS = 4_000
BOOTSTRAP_SEED = 7316811
HPR_FINAL_RMSE = 0.007850

GATES = {
    "fit_oof_zero_improvement_min": 0.20,
    "step22000_component_zero_improvement_min": 0.20,
    "step22000_component_persistence_improvement_min": 0.20,
    "step22000_component_spearman_min": 0.50,
    "step22000_component_sign_accuracy_min": 0.65,
    "step22000_component_slope_min": 0.50,
    "step22000_component_slope_max": 1.50,
    "step22000_aggregate_zero_improvement_min": 0.20,
    "step22000_aggregate_improvement_ci_low_min": 0.0,
    "final_aggregate_rmse_max": 1.25 * HPR_FINAL_RMSE,
    "final_aggregate_zero_improvement_min": 0.30,
    "final_aggregate_improvement_ci_low_min": 0.0,
    "parameter_fold_cv_max": 0.50,
    "leave_component_parameter_cv_max": 0.50,
}


@dataclass(frozen=True)
class Parameters:
    """Shared phase-local response parameters."""

    gamma: float
    rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze-availability", "prepare", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_payload(output_dir: Path) -> dict[str, object]:
    history_path = output_dir / HISTORY_CACHE_NAME
    availability_path = output_dir / AVAILABILITY_MANIFEST_NAME
    if not history_path.exists() or not availability_path.exists():
        raise RuntimeError("Run --mode freeze-availability before preparing the protocol")
    availability = pd.read_csv(availability_path)
    availability_counts = {
        column: int(availability[column].sum()) for column in ("fit_complete", "step22000_complete", "final_complete")
    }
    payload: dict[str, object] = {
        "candidate_id": CANDIDATE_ID,
        "version": VERSION,
        "equation": "d(s)=d1+(d0-d1)exp(-lambda*s); d1=-4*gamma*d0",
        "state": "component BPB difference between aggregate-matched asymmetric and tied policies",
        "fit_outcomes": "seven component BPB trajectories at steps 19000, 20000, and 21000 only",
        "heldout_outcomes": "component and aggregate BPB at step 22000 and 22887; aggregate BPB at all steps",
        "pre_switch_extrapolation": "linear 17000-to-18000 extrapolation to the exact step-18310 boundary",
        "phase_boundary_step": PHASE_BOUNDARY_STEP,
        "final_step": FINAL_STEP,
        "fit_steps": FIT_STEPS,
        "holdout_steps": HOLDOUT_STEPS,
        "phase_ratio": PHASE_RATIO,
        "component_keys": COMPONENT_KEYS,
        "aggregate_key": AGGREGATE_KEY,
        "parameter_bounds": {"gamma": GAMMA_BOUNDS, "rate": RATE_BOUNDS},
        "outer_folds": {"count": OUTER_FOLDS, "group": "pair_id", "seed": FOLD_SEED},
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "unit": "pair_id", "seed": BOOTSTRAP_SEED},
        "gates": GATES,
        "decision": (
            "Every gate must pass. A pass identifies only the transition law and does not promote a full surrogate."
        ),
        "coverage_requirement": (
            "All 238 pairs must have every component and aggregate outcome at both pre-switch, all fit, "
            "and both holdout steps."
        ),
        "history_collection": {
            "method": "wandb.scan_history",
            "cache": HISTORY_CACHE_NAME,
        },
        "complete_case_rule": {
            "fit": "all seven components present at steps 17000, 18000, 19000, 20000, and 21000",
            "step22000": "all seven components and aggregate present at steps 17000, 18000, and 22000",
            "final": "all seven components and aggregate present at steps 17000, 18000, and 22887",
            "counts": availability_counts,
            "minimum_counts": {
                "fit_complete": MIN_FIT_PAIRS,
                "step22000_complete": MIN_STEP_22000_PAIRS,
                "final_complete": MIN_FINAL_PAIRS,
            },
        },
        "source_hashes": {
            str(Path(__file__).resolve().relative_to(REPO_ROOT)): sha256_file(Path(__file__).resolve()),
            str(SOURCE_HISTORY.relative_to(REPO_ROOT)): sha256_file(SOURCE_HISTORY),
            str(history_path.resolve().relative_to(REPO_ROOT)): sha256_file(history_path),
            str(availability_path.resolve().relative_to(REPO_ROOT)): sha256_file(availability_path),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def manifest() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_HISTORY)
    source = source.loc[source["scale_key"].eq("300m")].copy()
    columns = ["pair_id", "policy_class", "wandb_run_id", "wandb_run_name", "wandb_data_seed"]
    frame = source[columns].drop_duplicates().reset_index(drop=True)
    counts = frame.groupby(["pair_id", "policy_class"]).size()
    expected_runs = 2 * EXPECTED_PAIRS
    if len(frame) != expected_runs or len(counts) != expected_runs or not counts.eq(1).all():
        raise ValueError("Expected exactly one one-phase and one two-phase run for each of 238 pairs")
    if set(frame["policy_class"]) != {"one_phase", "two_phase"}:
        raise ValueError("Unexpected policy class")
    return frame.sort_values(["pair_id", "policy_class"]).reset_index(drop=True)


def preflight_payload(frame: pd.DataFrame, protocol: dict[str, object]) -> dict[str, object]:
    complete_case_rule = protocol["complete_case_rule"]
    if not isinstance(complete_case_rule, dict):
        raise TypeError("Protocol complete-case rule must be a mapping")
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "pairs": int(frame["pair_id"].nunique()),
        "runs": len(frame),
        "component_targets": len(COMPONENT_KEYS),
        "fit_uses_aggregate_outcomes": False,
        "fit_uses_step22000": False,
        "fit_uses_final_endpoint": False,
        "fit_uses_table9": False,
        "fit_uses_wsd80": False,
        "nominal_parameter_count": 2,
        "complete_case_counts": complete_case_rule["counts"],
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = manifest()
    protocol = protocol_payload(output_dir)
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "preflight.json", preflight_payload(frame, protocol))
    frame.to_csv(output_dir / "run_manifest.csv", index=False)
    print(json.dumps(preflight_payload(frame, protocol), indent=2, sort_keys=True))


def verify_protocol(output_dir: Path) -> dict[str, object]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise RuntimeError("Run --mode prepare before evaluating outcomes")
    frozen = json.loads(path.read_text())
    current = protocol_payload(output_dir)
    frozen_canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
    current_canonical = json.dumps(current, sort_keys=True, separators=(",", ":"))
    if frozen_canonical != current_canonical:
        raise RuntimeError("Frozen protocol or a hashed source changed; declare an erratum instead of evaluating")
    return frozen


def fetch_history(row: object) -> pd.DataFrame:
    api = wandb.Api(timeout=90)
    run = api.run(f"{WANDB_PATH}/{row.wandb_run_id}")
    records = list(run.scan_history(keys=["global_step", *ALL_TARGET_KEYS], page_size=1_000))
    history = pd.DataFrame.from_records(records)
    missing = [key for key in ("global_step", *ALL_TARGET_KEYS) if key not in history]
    if missing:
        raise RuntimeError(f"Run {run.id} lacks history keys: {missing}")
    history = history[["global_step", *ALL_TARGET_KEYS]].copy()
    history = history.loc[history[list(ALL_TARGET_KEYS)].notna().any(axis=1)].copy()
    history["global_step"] = history["global_step"].astype(int)
    # W&B can store metrics logged at one training step in multiple sparse rows.
    # last retains the final non-null value of each target rather than dropping split rows.
    history = history.groupby("global_step", as_index=False, sort=True)[list(ALL_TARGET_KEYS)].last()
    history["pair_id"] = row.pair_id
    history["policy_class"] = row.policy_class
    history["wandb_run_id"] = run.id
    return history


def collect_histories(
    frame: pd.DataFrame,
    path: Path,
    *,
    refresh: bool,
    max_workers: int,
) -> pd.DataFrame:
    if path.exists() and not refresh:
        cached = pd.read_csv(path)
    else:
        cached = pd.DataFrame()
    completed = set(cached.get("wandb_run_id", pd.Series(dtype=str)).astype(str))
    pending_rows = [row for row in frame.itertuples(index=False) if str(row.wandb_run_id) not in completed]
    blocks = [cached] if not cached.empty else []
    errors: list[str] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {executor.submit(fetch_history, row): row for row in pending_rows}
        batch: list[pd.DataFrame] = []
        for index, future in enumerate(futures.as_completed(pending), start=1):
            row = pending[future]
            try:
                batch.append(future.result())
            except Exception as error:  # Re-run safely after transient W&B failures.
                errors.append(f"{row.wandb_run_id}: {error}")
            if index % 25 == 0 or index == len(pending):
                if batch:
                    blocks.extend(batch)
                    batch = []
                    combined = pd.concat(blocks, ignore_index=True).drop_duplicates(
                        ["wandb_run_id", "global_step"], keep="last"
                    )
                    combined.to_csv(path, index=False)
                print(f"Fetched {index}/{len(pending)} pending histories", flush=True)
    if errors:
        raise RuntimeError("History collection incomplete:\n" + "\n".join(errors))
    result = pd.read_csv(path)
    duplicate_rows = result.duplicated(["wandb_run_id", "global_step"], keep=False)
    if duplicate_rows.any():
        raise RuntimeError(f"History cache contains {int(duplicate_rows.sum())} duplicate run-step rows")
    observed_runs = set(result["wandb_run_id"].astype(str))
    expected_runs = set(frame["wandb_run_id"].astype(str))
    if observed_runs != expected_runs:
        raise RuntimeError(f"History cache covers {len(observed_runs)}/{len(expected_runs)} runs")
    return result


def pair_differences(histories: pd.DataFrame) -> pd.DataFrame:
    index = ["pair_id", "global_step"]
    one = histories.loc[histories["policy_class"].eq("one_phase"), [*index, *ALL_TARGET_KEYS]].copy()
    two = histories.loc[histories["policy_class"].eq("two_phase"), [*index, *ALL_TARGET_KEYS]].copy()
    merged = two.merge(one, on=index, suffixes=("__two", "__one"), validate="one_to_one")
    output = merged[index].copy()
    for key in ALL_TARGET_KEYS:
        output[key] = merged[f"{key}__two"] - merged[f"{key}__one"]
    return output.sort_values(index).reset_index(drop=True)


def boundary_difference(block: pd.DataFrame, key: str) -> pd.Series:
    pivot = block.pivot(index="pair_id", columns="global_step", values=key)
    missing = [step for step in PRE_SWITCH_STEPS if step not in pivot]
    if missing:
        raise RuntimeError(f"Missing pre-switch steps for {key}: {missing}")
    fraction = (PHASE_BOUNDARY_STEP - PRE_SWITCH_STEPS[1]) / (PRE_SWITCH_STEPS[1] - PRE_SWITCH_STEPS[0])
    return pivot[PRE_SWITCH_STEPS[1]] + fraction * (pivot[PRE_SWITCH_STEPS[1]] - pivot[PRE_SWITCH_STEPS[0]])


def phase_progress(step: int | np.ndarray) -> np.ndarray:
    return (np.asarray(step, dtype=float) - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)


def relaxation_factor(progress: np.ndarray, parameters: Parameters) -> np.ndarray:
    return (1.0 + PHASE_RATIO * parameters.gamma) * np.exp(-parameters.rate * progress) - (
        PHASE_RATIO * parameters.gamma
    )


def long_component_frame(differences: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key in COMPONENT_KEYS:
        d0 = boundary_difference(differences, key).rename("d0")
        block = differences.loc[
            differences["global_step"].isin((*FIT_STEPS, *HOLDOUT_STEPS)),
            [
                "pair_id",
                "global_step",
                key,
            ],
        ].rename(columns={key: "observed"})
        block = block.merge(d0, left_on="pair_id", right_index=True, how="inner")
        block["component"] = key.removeprefix("eval/uncheatable_eval/").removesuffix("/bpb")
        block["progress"] = phase_progress(block["global_step"].to_numpy())
        rows.append(block)
    return pd.concat(rows, ignore_index=True).dropna().reset_index(drop=True)


def aggregate_frame(differences: pd.DataFrame) -> pd.DataFrame:
    d0 = boundary_difference(differences, AGGREGATE_KEY).rename("d0")
    block = differences.loc[
        differences["global_step"].isin(HOLDOUT_STEPS),
        [
            "pair_id",
            "global_step",
            AGGREGATE_KEY,
        ],
    ].rename(columns={AGGREGATE_KEY: "observed"})
    block = block.merge(d0, left_on="pair_id", right_index=True, how="inner")
    block["component"] = "aggregate_uncheatable"
    block["progress"] = phase_progress(block["global_step"].to_numpy())
    return block.dropna().reset_index(drop=True)


def complete_pair_mask(
    differences: pd.DataFrame,
    *,
    steps: tuple[int, ...],
    keys: tuple[str, ...],
) -> pd.Series:
    block = differences.loc[differences["global_step"].isin(steps), ["pair_id", "global_step", *keys]].copy()
    block["complete"] = block[list(keys)].notna().all(axis=1)
    counts = block.loc[block["complete"]].groupby("pair_id")["global_step"].nunique()
    pair_ids = pd.Index(sorted(differences["pair_id"].unique()), name="pair_id")
    return counts.reindex(pair_ids, fill_value=0).eq(len(steps))


def availability_frame(differences: pd.DataFrame) -> pd.DataFrame:
    pair_ids = pd.Index(sorted(differences["pair_id"].unique()), name="pair_id")
    if len(pair_ids) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} paired trajectories, found {len(pair_ids)}")
    result = pd.DataFrame(index=pair_ids)
    result["fit_complete"] = complete_pair_mask(
        differences,
        steps=(*PRE_SWITCH_STEPS, *FIT_STEPS),
        keys=COMPONENT_KEYS,
    )
    result["step22000_complete"] = complete_pair_mask(
        differences,
        steps=(*PRE_SWITCH_STEPS, 22_000),
        keys=ALL_TARGET_KEYS,
    )
    result["final_complete"] = complete_pair_mask(
        differences,
        steps=(*PRE_SWITCH_STEPS, FINAL_STEP),
        keys=ALL_TARGET_KEYS,
    )
    return result.reset_index()


def freeze_availability(output_dir: Path) -> None:
    history_path = output_dir / HISTORY_CACHE_NAME
    if not history_path.exists():
        raise RuntimeError(f"Missing exhaustive history cache: {history_path}")
    histories = pd.read_csv(history_path)
    differences = pair_differences(histories)
    availability = availability_frame(differences)
    counts = {
        column: int(availability[column].sum()) for column in ("fit_complete", "step22000_complete", "final_complete")
    }
    minimums = {
        "fit_complete": MIN_FIT_PAIRS,
        "step22000_complete": MIN_STEP_22000_PAIRS,
        "final_complete": MIN_FINAL_PAIRS,
    }
    failed = {key: value for key, value in counts.items() if value < minimums[key]}
    if failed:
        raise RuntimeError(f"Insufficient complete-case trajectory coverage: {failed}")
    availability.to_csv(output_dir / AVAILABILITY_MANIFEST_NAME, index=False)
    write_json(
        output_dir / "availability_summary.json",
        {
            "candidate_id": CANDIDATE_ID,
            "counts": counts,
            "minimums": minimums,
            "selection_uses_outcome_values": False,
        },
    )
    print(json.dumps({"counts": counts, "minimums": minimums}, indent=2, sort_keys=True))


def verify_availability(differences: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    path = output_dir / AVAILABILITY_MANIFEST_NAME
    frozen = pd.read_csv(path).sort_values("pair_id").reset_index(drop=True)
    current = availability_frame(differences).sort_values("pair_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(frozen, current, check_dtype=False)
    return frozen


def component_scales(frame: pd.DataFrame) -> dict[str, float]:
    scales = frame.groupby("component")["d0"].apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
    return {str(key): max(float(value), 1e-4) for key, value in scales.items()}


def fit_parameters(frame: pd.DataFrame) -> Parameters:
    scales = component_scales(frame)
    scale = frame["component"].map(scales).to_numpy(dtype=float)
    progress = frame["progress"].to_numpy(dtype=float)
    d0 = frame["d0"].to_numpy(dtype=float)
    observed = frame["observed"].to_numpy(dtype=float)

    def residual(raw: np.ndarray) -> np.ndarray:
        parameters = Parameters(gamma=float(raw[0]), rate=float(raw[1]))
        predicted = d0 * relaxation_factor(progress, parameters)
        return (predicted - observed) / scale

    starts = ((0.25, 0.5), (0.5, 2.0), (1.0, 5.0), (2.0, 10.0), (3.0, 15.0))
    fits = [
        least_squares(
            residual,
            x0=np.asarray(start, dtype=float),
            bounds=(np.asarray([GAMMA_BOUNDS[0], RATE_BOUNDS[0]]), np.asarray([GAMMA_BOUNDS[1], RATE_BOUNDS[1]])),
            max_nfev=20_000,
        )
        for start in starts
    ]
    best = min(fits, key=lambda fit: float(np.dot(fit.fun, fit.fun)))
    return Parameters(gamma=float(best.x[0]), rate=float(best.x[1]))


def predict(frame: pd.DataFrame, parameters: Parameters) -> np.ndarray:
    return frame["d0"].to_numpy(dtype=float) * relaxation_factor(frame["progress"].to_numpy(dtype=float), parameters)


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    value = spearmanr(observed, predicted).statistic
    return float(value) if np.isfinite(value) else 0.0


def summary_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    denominator = float(np.dot(predicted - predicted.mean(), predicted - predicted.mean()))
    slope = (
        float(np.dot(predicted - predicted.mean(), observed - observed.mean()) / denominator)
        if denominator > 1e-16
        else 0.0
    )
    observed_rms = float(np.sqrt(np.mean(np.square(observed))))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
        "spearman": safe_spearman(observed, predicted),
        "observed_on_predicted_slope": slope,
        "amplitude_ratio": float(np.sqrt(np.mean(np.square(predicted))) / max(observed_rms, 1e-12)),
        "sign_accuracy": float(np.mean(np.signbit(predicted) == np.signbit(observed))),
    }


def relative_improvement(candidate_rmse: float, baseline_rmse: float) -> float:
    return 1.0 - candidate_rmse / max(baseline_rmse, 1e-12)


def grouped_oof(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    pairs = np.asarray(sorted(frame["pair_id"].unique()), dtype=object)
    splitter = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=FOLD_SEED)
    prediction = np.full(len(frame), np.nan)
    parameter_rows: list[dict[str, float | int]] = []
    pair_values = frame["pair_id"].to_numpy(dtype=object)
    for fold, (train_index, test_index) in enumerate(splitter.split(pairs)):
        train_pairs = set(pairs[train_index])
        test_pairs = set(pairs[test_index])
        train_mask = np.asarray([value in train_pairs for value in pair_values])
        test_mask = np.asarray([value in test_pairs for value in pair_values])
        parameters = fit_parameters(frame.loc[train_mask])
        prediction[test_mask] = predict(frame.loc[test_mask], parameters)
        parameter_rows.append({"fold": fold, "gamma": parameters.gamma, "rate": parameters.rate})
    if not np.isfinite(prediction).all():
        raise RuntimeError("OOF predictions are incomplete")
    return prediction, pd.DataFrame(parameter_rows)


def leave_component_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for component in sorted(frame["component"].unique()):
        parameters = fit_parameters(frame.loc[~frame["component"].eq(component)])
        rows.append({"excluded_component": component, "gamma": parameters.gamma, "rate": parameters.rate})
    return pd.DataFrame(rows)


def coefficient_of_variation(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.std(array, ddof=1) / max(abs(float(np.mean(array))), 1e-12))


def bootstrap_improvement(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    working = frame[["pair_id", "observed"]].copy()
    working["predicted"] = predicted
    working["baseline"] = baseline
    groups = {pair: block.index.to_numpy() for pair, block in working.groupby("pair_id", sort=True)}
    pair_ids = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        indices = np.concatenate([groups[pair] for pair in sampled])
        observed = working.loc[indices, "observed"].to_numpy(dtype=float)
        candidate = working.loc[indices, "predicted"].to_numpy(dtype=float)
        null = working.loc[indices, "baseline"].to_numpy(dtype=float)
        candidate_rmse = float(np.sqrt(np.mean(np.square(candidate - observed))))
        null_rmse = float(np.sqrt(np.mean(np.square(null - observed))))
        values[draw] = relative_improvement(candidate_rmse, null_rmse)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_scope(
    frame: pd.DataFrame,
    parameters: Parameters,
    *,
    scope: str,
    step: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    block = frame.loc[frame["global_step"].eq(step)].copy().reset_index(drop=True)
    predicted = predict(block, parameters)
    observed = block["observed"].to_numpy(dtype=float)
    zero = np.zeros(len(block), dtype=float)
    persistence = block["d0"].to_numpy(dtype=float)
    metrics: dict[str, object] = {"scope": scope, "step": step, **summary_metrics(observed, predicted)}
    zero_rmse = summary_metrics(observed, zero)["rmse"]
    persistence_rmse = summary_metrics(observed, persistence)["rmse"]
    metrics["zero_rmse"] = zero_rmse
    metrics["persistence_rmse"] = persistence_rmse
    metrics["zero_improvement"] = relative_improvement(float(metrics["rmse"]), zero_rmse)
    metrics["persistence_improvement"] = relative_improvement(float(metrics["rmse"]), persistence_rmse)
    zero_low, zero_high = bootstrap_improvement(block, predicted, zero, seed=BOOTSTRAP_SEED + step)
    metrics["zero_improvement_ci_low"] = zero_low
    metrics["zero_improvement_ci_high"] = zero_high
    block["predicted"] = predicted
    block["zero_prediction"] = zero
    block["persistence_prediction"] = persistence
    block["scope"] = scope
    return metrics, block


def component_breakdown(frame: pd.DataFrame, parameters: Parameters, step: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component, block in frame.loc[frame["global_step"].eq(step)].groupby("component", sort=True):
        predicted = predict(block, parameters)
        observed = block["observed"].to_numpy(dtype=float)
        zero_rmse = summary_metrics(observed, np.zeros(len(block)))["rmse"]
        row: dict[str, object] = {"component": component, "step": step, **summary_metrics(observed, predicted)}
        row["zero_improvement"] = relative_improvement(float(row["rmse"]), zero_rmse)
        rows.append(row)
    return pd.DataFrame(rows)


def render_plot(
    component_predictions: pd.DataFrame,
    aggregate_predictions: pd.DataFrame,
    parameters: Parameters,
    path: Path,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Shared relaxation factor", "Step 22,000 components", "Final aggregate"),
    )
    progress = np.linspace(0.0, 1.0, 200)
    factor = relaxation_factor(progress, parameters)
    figure.add_trace(
        go.Scatter(x=progress, y=factor, mode="lines", line={"color": "#2166ac", "width": 3}, name="g(s)"),
        row=1,
        col=1,
    )
    step_block = component_predictions.loc[component_predictions["global_step"].eq(22_000)]
    figure.add_trace(
        go.Scatter(
            x=step_block["observed"],
            y=step_block["predicted"],
            mode="markers",
            marker={"color": step_block["observed"], "colorscale": "RdYlGn_r", "size": 6, "opacity": 0.65},
            text=step_block["component"],
            name="components",
        ),
        row=1,
        col=2,
    )
    final_block = aggregate_predictions.loc[aggregate_predictions["global_step"].eq(FINAL_STEP)]
    figure.add_trace(
        go.Scatter(
            x=final_block["observed"],
            y=final_block["predicted"],
            mode="markers",
            marker={"color": final_block["observed"], "colorscale": "RdYlGn_r", "size": 8},
            name="aggregate final",
        ),
        row=1,
        col=3,
    )
    for column, block in ((2, step_block), (3, final_block)):
        lower = float(min(block["observed"].min(), block["predicted"].min()))
        upper = float(max(block["observed"].max(), block["predicted"].max()))
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line={"dash": "dash", "color": "#555"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title=f"Component-observed phase-local relaxation: gamma={parameters.gamma:.3f}, lambda={parameters.rate:.3f}",
        template="plotly_white",
        width=1500,
        height=520,
    )
    figure.write_html(path, include_plotlyjs="cdn")


def render_report(
    decision: dict[str, object],
    parameters: Parameters,
    metrics: pd.DataFrame,
    fold_parameters: pd.DataFrame,
    component_metrics: pd.DataFrame,
) -> str:
    return f"""# Component-observed phase-local relaxation audit

**Decision: {decision['decision']}**

The shared parameters were selected only from seven component BPB trajectories
at steps 19,000--21,000. Step 22,000, final endpoints, aggregate Uncheatable,
Table-9, and WSD80 did not select the form or parameters.

- late equilibrium multiplier `gamma`: `{parameters.gamma:.6f}`
- relaxation rate `lambda`: `{parameters.rate:.6f}` per normalized phase-1 duration

## Scope metrics

{metrics.to_markdown(index=False)}

## Parameter stability

{fold_parameters.to_markdown(index=False)}

## Component holdouts

{component_metrics.to_markdown(index=False)}

A pass identifies only the phase-local relaxation law. It does not identify a
deployment policy-to-equilibrium map and does not repair a failed aggregate
spine.
"""


def evaluate(output_dir: Path, max_workers: int, refresh: bool) -> None:
    protocol = verify_protocol(output_dir)
    run_manifest = manifest()
    histories = collect_histories(
        run_manifest,
        output_dir / HISTORY_CACHE_NAME,
        refresh=refresh,
        max_workers=max_workers,
    )
    differences = pair_differences(histories)
    differences.to_csv(output_dir / "pair_differences.csv", index=False)
    availability = verify_availability(differences, output_dir)
    component = long_component_frame(differences)
    aggregate = aggregate_frame(differences)

    fit_pairs = set(availability.loc[availability["fit_complete"], "pair_id"])
    fit_frame = component.loc[component["global_step"].isin(FIT_STEPS) & component["pair_id"].isin(fit_pairs)].copy()
    fit_frame = fit_frame.reset_index(drop=True)
    expected_fit_rows = len(fit_pairs) * len(COMPONENT_KEYS) * len(FIT_STEPS)
    if len(fit_frame) != expected_fit_rows:
        raise RuntimeError(f"Expected {expected_fit_rows} complete fit rows, found {len(fit_frame)}")
    oof_prediction, fold_parameters = grouped_oof(fit_frame)
    full_parameters = fit_parameters(fit_frame)
    leave_component_parameters = leave_component_out(fit_frame)
    fold_parameters.to_csv(output_dir / "fold_parameters.csv", index=False)
    leave_component_parameters.to_csv(output_dir / "leave_component_parameters.csv", index=False)

    fit_observed = fit_frame["observed"].to_numpy(dtype=float)
    fit_metrics = {"scope": "component_oof_fit_steps", "step": -1, **summary_metrics(fit_observed, oof_prediction)}
    fit_zero_rmse = summary_metrics(fit_observed, np.zeros(len(fit_frame)))["rmse"]
    fit_metrics["zero_rmse"] = fit_zero_rmse
    fit_metrics["persistence_rmse"] = float("nan")
    fit_metrics["zero_improvement"] = relative_improvement(float(fit_metrics["rmse"]), fit_zero_rmse)
    fit_metrics["persistence_improvement"] = float("nan")
    fit_metrics["zero_improvement_ci_low"] = float("nan")
    fit_metrics["zero_improvement_ci_high"] = float("nan")

    metric_rows: list[dict[str, object]] = [fit_metrics]
    prediction_blocks: list[pd.DataFrame] = []
    component_metric_blocks: list[pd.DataFrame] = []
    for step in HOLDOUT_STEPS:
        availability_column = "step22000_complete" if step == 22_000 else "final_complete"
        holdout_pairs = set(availability.loc[availability[availability_column], "pair_id"])
        component_holdout = component.loc[component["pair_id"].isin(holdout_pairs)]
        aggregate_holdout = aggregate.loc[aggregate["pair_id"].isin(holdout_pairs)]
        component_metrics, component_prediction = evaluate_scope(
            component_holdout, full_parameters, scope="components", step=step
        )
        aggregate_metrics, aggregate_prediction = evaluate_scope(
            aggregate_holdout, full_parameters, scope="aggregate", step=step
        )
        metric_rows.extend((component_metrics, aggregate_metrics))
        prediction_blocks.extend((component_prediction, aggregate_prediction))
        component_metric_blocks.append(component_breakdown(component_holdout, full_parameters, step))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    component_metrics = pd.concat(component_metric_blocks, ignore_index=True)
    component_metrics.to_csv(output_dir / "component_metrics.csv", index=False)

    metric_index = metrics.set_index(["scope", "step"])
    fit_row = metric_index.loc[("component_oof_fit_steps", -1)]
    component_22000 = metric_index.loc[("components", 22_000)]
    aggregate_22000 = metric_index.loc[("aggregate", 22_000)]
    aggregate_final = metric_index.loc[("aggregate", FINAL_STEP)]
    fold_gamma_cv = coefficient_of_variation(fold_parameters["gamma"])
    fold_rate_cv = coefficient_of_variation(fold_parameters["rate"])
    component_gamma_cv = coefficient_of_variation(leave_component_parameters["gamma"])
    component_rate_cv = coefficient_of_variation(leave_component_parameters["rate"])
    interior = (
        GAMMA_BOUNDS[0] + 1e-3 < full_parameters.gamma < GAMMA_BOUNDS[1] - 1e-3
        and RATE_BOUNDS[0] + 1e-3 < full_parameters.rate < RATE_BOUNDS[1] - 1e-3
    )
    checks = {
        "fit_oof_zero_improvement": float(fit_row["zero_improvement"]) >= GATES["fit_oof_zero_improvement_min"],
        "step22000_component_zero_improvement": (
            float(component_22000["zero_improvement"]) >= GATES["step22000_component_zero_improvement_min"]
        ),
        "step22000_component_persistence_improvement": (
            float(component_22000["persistence_improvement"]) >= GATES["step22000_component_persistence_improvement_min"]
        ),
        "step22000_component_spearman": float(component_22000["spearman"]) >= GATES["step22000_component_spearman_min"],
        "step22000_component_sign": (
            float(component_22000["sign_accuracy"]) >= GATES["step22000_component_sign_accuracy_min"]
        ),
        "step22000_component_slope": (
            GATES["step22000_component_slope_min"]
            <= float(component_22000["observed_on_predicted_slope"])
            <= GATES["step22000_component_slope_max"]
        ),
        "step22000_aggregate_zero_improvement": (
            float(aggregate_22000["zero_improvement"]) >= GATES["step22000_aggregate_zero_improvement_min"]
        ),
        "step22000_aggregate_uncertainty": (
            float(aggregate_22000["zero_improvement_ci_low"]) >= GATES["step22000_aggregate_improvement_ci_low_min"]
        ),
        "final_aggregate_rmse": float(aggregate_final["rmse"]) <= GATES["final_aggregate_rmse_max"],
        "final_aggregate_zero_improvement": (
            float(aggregate_final["zero_improvement"]) >= GATES["final_aggregate_zero_improvement_min"]
        ),
        "final_aggregate_uncertainty": (
            float(aggregate_final["zero_improvement_ci_low"]) >= GATES["final_aggregate_improvement_ci_low_min"]
        ),
        "parameters_interior": interior,
        "fold_parameter_stability": max(fold_gamma_cv, fold_rate_cv) <= GATES["parameter_fold_cv_max"],
        "component_parameter_stability": (
            max(component_gamma_cv, component_rate_cv) <= GATES["leave_component_parameter_cv_max"]
        ),
    }
    passed = all(checks.values())
    decision = {
        "candidate_id": CANDIDATE_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "passed": passed,
        "decision": "PASS: relaxation law identified" if passed else "FAIL: relaxation law rejected",
        "parameters": {"gamma": full_parameters.gamma, "rate": full_parameters.rate},
        "stability": {
            "fold_gamma_cv": fold_gamma_cv,
            "fold_rate_cv": fold_rate_cv,
            "leave_component_gamma_cv": component_gamma_cv,
            "leave_component_rate_cv": component_rate_cv,
        },
        "checks": checks,
    }
    write_json(output_dir / "decision.json", decision)
    render_plot(
        predictions.loc[predictions["scope"].eq("components")],
        predictions.loc[predictions["scope"].eq("aggregate")],
        full_parameters,
        output_dir / "phase_local_relaxation.html",
    )
    (output_dir / "report.md").write_text(
        render_report(decision, full_parameters, metrics, fold_parameters, component_metrics)
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "freeze-availability":
        freeze_availability(args.output_dir)
        return
    if args.mode == "prepare":
        prepare(args.output_dir)
        return
    evaluate(args.output_dir, args.max_workers, args.refresh)


if __name__ == "__main__":
    main()
