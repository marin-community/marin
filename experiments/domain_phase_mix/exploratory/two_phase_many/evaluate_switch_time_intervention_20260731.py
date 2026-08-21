# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Fit the preregistered WSD80 switch-time intervention without endpoint leakage.

The workflow is intentionally staged:

1. ``materialize-transition`` may read only evaluations through step 6400.
2. ``freeze-transition`` selects the temporal mechanism on the primary target,
   freezes its transition and target-specific zero-intercept response heads,
   and records source, data, and prediction hashes.
3. ``materialize-final`` refuses to access the full post-6400 cosine-decay
   trajectory unless the transition gates passed and every frozen hash still
   matches.
4. ``evaluate-final`` applies the frozen model without refitting anything.

The aggregate-potential response is retained only as a null comparator. A
signed response whose utility varies linearly with aggregate and its
aggregate-invariant ablation permit phase gains inside a tied-optimal
uncertainty region, as required by the replicated 2B WSD80 counterexample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    design_switch_time_intervention_20260731 as design,
)

DESIGN_DIR = SCRIPT_DIR / "reference_outputs" / "switch_time_intervention_design_20260731"
PROTOCOL_PATH = DESIGN_DIR / "protocol.json"
MANIFEST_PATH = DESIGN_DIR / "manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "switch_time_intervention_evaluation_20260731"

WANDB_PATH = "marin-community/marin"
TRANSITION_MAX_STEP = design.TRANSITION_MAX_STEP
FINAL_STEP = int(design.wsd80._schedule_summary(design.EXPERIMENT_BUDGET)["total_steps"]) - 1
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260731
MAX_TWO_STATE_CONDITION = 1_000.0
FETCH_ATTEMPTS = 4
SPINE_CENTER = 0.35

MEMORYLESS_FAMILIES = {
    "token_dose_null",
    "lr_mass_dose_null",
    "terminal_level_null",
    "phase_local_repetition",
    "static_switch_control_null",
}
DYNAMIC_FAMILIES = {
    "token_clock_acquisition_forgetting",
    "lr_clock_acquisition_forgetting",
}
CONTROL_POTENTIAL_FAMILIES = {*DYNAMIC_FAMILIES, "static_switch_control_null"}
CONTROL_RESPONSE_MODES = (
    "potential_constrained",
    "aggregate_linear_signed",
    "unconditioned_signed",
)


@dataclass(frozen=True)
class Candidate:
    """One frozen temporal-state candidate."""

    family: str
    clock: str = "none"
    acquisition_rate: float | None = None
    forgetting_ratio: float | None = None
    repetition_power: int | None = None
    response_mode: str = "potential_constrained"

    @property
    def candidate_id(self) -> str:
        parts = [self.family]
        if self.clock != "none":
            parts.append(self.clock)
        if self.acquisition_rate is not None:
            parts.append(f"ka{self.acquisition_rate:g}")
        if self.forgetting_ratio is not None:
            parts.append(f"gamma{self.forgetting_ratio:g}")
        if self.repetition_power is not None:
            parts.append(f"h{self.repetition_power}")
        if self.family in CONTROL_POTENTIAL_FAMILIES:
            parts.append(self.response_mode)
        return "__".join(parts)

    @property
    def feature_dimension(self) -> int:
        if self.family not in CONTROL_POTENTIAL_FAMILIES:
            return 1
        return 3 if self.response_mode == "aggregate_linear_signed" else 2

    @property
    def coefficient_lower_bounds(self) -> tuple[float, ...]:
        """Return mechanistic sign constraints for the zero-intercept head."""
        if self.family == "phase_local_repetition":
            return (0.0,)
        if self.family not in CONTROL_POTENTIAL_FAMILIES:
            return (-math.inf,) * self.feature_dimension
        if self.response_mode == "potential_constrained":
            return (0.0, 0.0)
        return (-math.inf,) * self.feature_dimension


@dataclass(frozen=True)
class HeadFit:
    """One zero-intercept response head."""

    coefficients: tuple[float, ...]
    condition_number: float


@dataclass(frozen=True)
class SpineFit:
    """One tied-only local quadratic aggregate response at one target and step."""

    intercept: float
    slope: float
    curvature: float
    center: float
    condition_number: float


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def endpoint_unseal_path(output_dir: Path) -> Path:
    return output_dir / "endpoint_unsealed.json"


def final_artifacts(output_dir: Path) -> tuple[Path, ...]:
    return (
        endpoint_unseal_path(output_dir),
        output_dir / "final_runs",
        output_dir / "final_observations.csv",
        output_dir / "final_materialization.json",
        output_dir / "final_predictions.csv",
        output_dir / "final_metrics.csv",
        output_dir / "final_evaluation.json",
        output_dir / "final_report.md",
    )


def assert_endpoint_still_sealed(output_dir: Path) -> None:
    existing = [str(path.name) for path in final_artifacts(output_dir) if path.exists()]
    if existing:
        raise PermissionError(f"Transition cannot be refrozen after endpoint unseal: {existing}")


def mark_endpoint_unsealed(output_dir: Path, frozen: dict[str, Any]) -> None:
    """Create an irreversible local marker before any final-endpoint request."""
    marker = endpoint_unseal_path(output_dir)
    payload = {
        "candidate_id": frozen["candidate_id"],
        "protocol_sha256": frozen["protocol_sha256"],
        "frozen_transition_sha256": sha256_path(output_dir / "frozen_transition.json"),
        "unsealed_at_unix_seconds": time.time(),
        "reason": "final endpoint materialization began; transition refreeze is permanently prohibited",
    }
    if marker.exists():
        existing = json.loads(marker.read_text())
        for key in ("candidate_id", "protocol_sha256", "frozen_transition_sha256"):
            if str(existing[key]) != str(payload[key]):
                raise ValueError(f"Endpoint-unseal marker drift at {key}")
        return
    write_json(marker, payload)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_protocol() -> dict[str, Any]:
    """Load the protocol and verify its self-hash and all frozen sources."""
    protocol = json.loads(PROTOCOL_PATH.read_text())
    claimed_hash = str(protocol.pop("protocol_sha256"))
    observed_hash = sha256_json(protocol)
    if observed_hash != claimed_hash:
        raise ValueError(f"Protocol self-hash mismatch: {observed_hash} != {claimed_hash}")
    protocol["protocol_sha256"] = claimed_hash

    source_paths = {
        "design_script_sha256": Path(design.__file__),
        "launcher_sha256": design.LAUNCHER_PATH,
        "canonical_wsd80_launcher_sha256": design.BASE_LAUNCHER_PATH,
        "evaluator_sha256": Path(__file__),
        "fiber_counterexample_report_sha256": design.FIBER_COUNTEREXAMPLE_REPORT_PATH,
    }
    for key, path in source_paths.items():
        observed = sha256_path(path)
        expected = str(protocol["sources"][key])
        if observed != expected:
            raise ValueError(f"Frozen source drift at {key}: {observed} != {expected}")
    return protocol


def load_manifest(protocol: dict[str, Any]) -> pd.DataFrame:
    """Load the observation manifest and verify its frozen string-level hash."""
    rows = csv_rows(MANIFEST_PATH)
    observed_hash = sha256_json(rows)
    expected_hash = str(protocol["panel"]["manifest_sha256"])
    if observed_hash != expected_hash:
        raise ValueError(f"Manifest hash mismatch: {observed_hash} != {expected_hash}")
    frame = pd.DataFrame.from_records(rows)
    numeric_columns = (
        "switch_step",
        "run_seed",
        "phase_0_code_weight",
        "phase_1_code_weight",
        "aggregate_code_weight",
        "code_subset_tokens",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column])
    if len(frame) != int(protocol["panel"]["observations"]):
        raise ValueError("Manifest observation count drifted")
    return frame


def target_columns(protocol: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(protocol["targets"]["primary"]),
        *(str(value) for value in protocol["targets"]["code_transfer"]),
        *(str(value) for value in protocol["targets"]["broad_text_negative_controls"]),
    )


def transition_steps(protocol: dict[str, Any]) -> tuple[int, ...]:
    interval = int(protocol["schedule"]["eval_interval_steps"])
    return tuple(range(interval, TRANSITION_MAX_STEP + 1, interval))


def sealed_decay_steps(protocol: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(step) for step in protocol["schedule"]["sealed_decay_steps"])


def verify_stage_steps(frame: pd.DataFrame, expected_steps: Sequence[int], *, stage: str) -> None:
    observed = set(frame["global_step"].astype(int))
    expected = set(expected_steps)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{stage} step mismatch; missing={missing[:10]}, extra={extra[:10]}")


def fetch_history(
    observation_id: str,
    *,
    targets: Sequence[str],
    minimum_step: int,
    maximum_step: int,
    wandb_path: str,
) -> pd.DataFrame:
    """Fetch one bounded W&B history range and return a long target table."""
    records: list[dict[str, Any]] | None = None
    last_error: Exception | None = None
    keys = ["_step", "global_step", *targets]
    for attempt in range(FETCH_ATTEMPTS):
        try:
            run = wandb.Api(timeout=120).run(f"{wandb_path}/{observation_id}")
            records = list(
                run.scan_history(
                    keys=keys,
                    page_size=1_000,
                    min_step=minimum_step,
                    max_step=maximum_step + 1,
                )
            )
            break
        except Exception as error:
            last_error = error
            if attempt + 1 < FETCH_ATTEMPTS:
                time.sleep(2**attempt)
    if records is None:
        raise RuntimeError(f"W&B history fetch exhausted retries for {observation_id}") from last_error

    history = pd.DataFrame.from_records(records)
    required = {"global_step", *targets}
    if not required.issubset(history.columns):
        missing = sorted(required - set(history.columns))
        raise ValueError(f"Run {observation_id} lacks history columns: {missing}")
    history = history.loc[history["global_step"].notna()].copy()
    history["global_step"] = history["global_step"].astype(int)
    if "_step" not in history or history["_step"].isna().any():
        raise ValueError(f"Run {observation_id} lacks a W&B step for at least one target row")
    history["_step"] = history["_step"].astype(int)
    if not np.array_equal(history["_step"].to_numpy(int), history["global_step"].to_numpy(int)):
        mismatches = history.loc[history["_step"].ne(history["global_step"]), ["_step", "global_step"]].head()
        raise ValueError(f"W&B and logged global steps differ for {observation_id}:\n{mismatches}")
    internal_steps = history["_step"]
    if int(internal_steps.min()) < minimum_step or int(internal_steps.max()) > maximum_step:
        raise ValueError(f"W&B internal-step bound violated for {observation_id}")
    if int(history["global_step"].min()) < minimum_step or int(history["global_step"].max()) > maximum_step:
        raise ValueError(f"Logged global-step bound violated for {observation_id}")
    history = history.groupby("global_step", as_index=False, sort=True)[list(targets)].last()
    long = history.melt(id_vars="global_step", var_name="target", value_name="value")
    long = long.loc[long["value"].notna()].copy()
    long["observation_id"] = observation_id
    return long[["observation_id", "global_step", "target", "value"]]


def materialize_stage(
    *,
    output_dir: Path,
    stage: str,
    wandb_path: str,
    refresh: bool,
    max_workers: int,
) -> Path:
    """Materialize one explicitly bounded W&B stage with per-run resume caches."""
    protocol = load_protocol()
    manifest = load_manifest(protocol)
    targets = target_columns(protocol)
    if stage == "transition":
        minimum_step = 0
        maximum_step = TRANSITION_MAX_STEP
        expected_steps = transition_steps(protocol)
    elif stage == "final":
        frozen = verify_frozen_transition(output_dir, require_license=True)
        mark_endpoint_unsealed(output_dir, frozen)
        expected_steps = sealed_decay_steps(protocol)
        minimum_step = min(expected_steps)
        maximum_step = FINAL_STEP
    else:
        raise ValueError(f"Unknown materialization stage: {stage}")

    cache_dir = output_dir / f"{stage}_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def obtain(observation_id: str) -> pd.DataFrame:
        cache_path = cache_dir / f"{observation_id}.csv"
        if cache_path.exists() and not refresh:
            cached = pd.read_csv(cache_path)
            verify_stage_steps(cached, expected_steps, stage=f"cached {stage} run {observation_id}")
            return cached
        fetched = fetch_history(
            observation_id,
            targets=targets,
            minimum_step=minimum_step,
            maximum_step=maximum_step,
            wandb_path=wandb_path,
        )
        fetched = fetched.loc[fetched["global_step"].isin(expected_steps)].copy()
        verify_stage_steps(fetched, expected_steps, stage=f"fetched {stage} run {observation_id}")
        expected_rows = len(expected_steps) * len(targets)
        if len(fetched) != expected_rows:
            raise ValueError(f"Run {observation_id} has {len(fetched)} {stage} rows, expected {expected_rows}")
        fetched.to_csv(cache_path, index=False)
        return fetched

    histories: list[pd.DataFrame] = []
    failures: list[str] = []
    observation_ids = manifest["observation_id"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(obtain, observation_id): observation_id for observation_id in observation_ids}
        for future in as_completed(futures):
            observation_id = futures[future]
            try:
                histories.append(future.result())
            except Exception as error:
                failures.append(f"{observation_id}: {error}")
    if failures:
        raise RuntimeError(f"{stage} materialization incomplete ({len(failures)} runs):\n" + "\n".join(failures))

    combined = pd.concat(histories, ignore_index=True).sort_values(["observation_id", "global_step", "target"])
    expected_total = len(observation_ids) * len(expected_steps) * len(targets)
    if len(combined) != expected_total:
        raise ValueError(f"Combined {stage} rows {len(combined)} != {expected_total}")
    output_path = output_dir / f"{stage}_observations.csv"
    combined.to_csv(output_path, index=False)
    write_json(
        output_dir / f"{stage}_materialization.json",
        {
            "stage": stage,
            "protocol_sha256": protocol["protocol_sha256"],
            "evaluator_sha256": sha256_path(Path(__file__)),
            "observations_sha256": sha256_path(output_path),
            "rows": len(combined),
            "minimum_step": minimum_step,
            "maximum_step": maximum_step,
            "outcomes_accessed": f"only {stage} steps {minimum_step} through {maximum_step}",
        },
    )
    return output_path


def residualize_with_observed_tied_controls(
    raw: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Define sealed deltas from same-anchor, same-seed tied observations without refitting."""
    metadata_columns = [
        "observation_id",
        "coordinate_id",
        "anchor_id",
        "design_arm",
        "role",
        "pair_id",
        "switch_step",
        "run_seed",
        "phase_0_code_weight",
        "phase_1_code_weight",
        "aggregate_code_weight",
        "signed_contrast",
        "code_subset_tokens",
    ]
    metadata = manifest[metadata_columns].copy()
    asymmetric = metadata.loc[~metadata["role"].eq("spine_tied_control")]
    residualized = asymmetric.merge(raw, on="observation_id", how="left", validate="one_to_many")
    tied_metadata = metadata.loc[
        metadata["role"].eq("spine_tied_control"),
        ["observation_id", "anchor_id", "run_seed"],
    ].rename(columns={"observation_id": "tied_observation_id"})
    tied_values = tied_metadata.merge(
        raw.rename(columns={"observation_id": "tied_observation_id", "value": "tied_value"}),
        on="tied_observation_id",
        how="left",
        validate="one_to_many",
    )
    tied_values = tied_values[["anchor_id", "run_seed", "global_step", "target", "tied_observation_id", "tied_value"]]
    residualized = residualized.merge(
        tied_values,
        on=["anchor_id", "run_seed", "global_step", "target"],
        how="left",
        validate="many_to_one",
    )
    if residualized[["value", "tied_value"]].isna().any().any():
        raise ValueError("Sealed trajectories lack a same-anchor, same-seed tied control")
    residualized["observed_delta"] = residualized["value"].astype(float) - residualized["tied_value"].astype(float)
    return residualized.sort_values(["target", "switch_step", "coordinate_id", "run_seed", "global_step"]).reset_index(
        drop=True
    )


def collapsed_tied_values(raw: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return one tied target value per aggregate, seed, and evaluation step."""
    tied_metadata = manifest.loc[
        manifest["role"].astype(str).eq("spine_tied_control"),
        ["observation_id", "aggregate_code_weight", "run_seed"],
    ].copy()
    tied = tied_metadata.merge(raw, on="observation_id", how="left", validate="one_to_many")
    if tied[["value", "aggregate_code_weight"]].isna().any().any():
        raise ValueError("Tied aggregate-spine rows contain missing values")
    return tied.groupby(
        ["target", "global_step", "aggregate_code_weight", "run_seed"],
        as_index=False,
    ).agg(value=("value", "mean"))


def quadratic_spine_fit(group: pd.DataFrame) -> SpineFit:
    """Fit an unconstrained local quadratic to tied policies only."""
    means = group.groupby("aggregate_code_weight", as_index=False)["value"].mean()
    x = means["aggregate_code_weight"].to_numpy(float) - SPINE_CENTER
    design_matrix = np.column_stack((np.ones(len(means)), x, x**2))
    coefficients, _, rank, _ = np.linalg.lstsq(design_matrix, means["value"].to_numpy(float), rcond=None)
    if rank != design_matrix.shape[1] or not np.isfinite(coefficients).all():
        raise RuntimeError("Tied-spine quadratic fit failed")
    return SpineFit(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        curvature=float(coefficients[2]),
        center=SPINE_CENTER,
        condition_number=float(np.linalg.cond(design_matrix)),
    )


def aggregate_spine_diagnostics(
    raw: pd.DataFrame,
    manifest: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit tied-spine interpolation and raw-optimum stability independently."""
    tied = collapsed_tied_values(raw, manifest)
    rows: list[dict[str, Any]] = []
    for (target_name, global_step), group in tied.groupby(["target", "global_step"], sort=True):
        fit = quadratic_spine_fit(group)
        lolo_errors: list[float] = []
        for omitted_aggregate in sorted(group["aggregate_code_weight"].astype(float).unique()):
            held_in = group.loc[~np.isclose(group["aggregate_code_weight"].astype(float), omitted_aggregate)]
            held_out = group.loc[np.isclose(group["aggregate_code_weight"].astype(float), omitted_aggregate)]
            held_in_fit = quadratic_spine_fit(held_in)
            offset = omitted_aggregate - held_in_fit.center
            prediction = held_in_fit.intercept + held_in_fit.slope * offset + held_in_fit.curvature * offset**2
            lolo_errors.append(prediction - float(held_out["value"].mean()))
        optimum = math.nan
        if fit.curvature > 0.0:
            optimum = fit.center - fit.slope / (2.0 * fit.curvature)
        rows.append(
            {
                "target": str(target_name),
                "global_step": int(global_step),
                "curvature": fit.curvature,
                "tied_optimum": optimum,
                "leave_one_level_out_rmse": float(np.sqrt(np.mean(np.square(lolo_errors)))),
                "leave_one_level_out_bias": float(np.mean(lolo_errors)),
                "condition_number": fit.condition_number,
            }
        )
    diagnostics = pd.DataFrame.from_records(rows)

    primary_target = str(protocol["targets"]["primary"])
    terminal = tied.loc[tied["target"].eq(primary_target) & tied["global_step"].eq(TRANSITION_MAX_STEP)].copy()
    aggregates = sorted(terminal["aggregate_code_weight"].astype(float).unique())
    seeds = sorted(terminal["run_seed"].astype(int).unique())
    pivot = terminal.pivot(index="aggregate_code_weight", columns="run_seed", values="value").reindex(
        index=aggregates,
        columns=seeds,
    )
    if pivot.isna().any().any() or len(aggregates) != 5 or len(seeds) != len(design.SPINE_SEED_VALUES):
        raise ValueError("Aggregate-spine bootstrap requires five complete tied levels by six common seeds")
    x = np.asarray(aggregates, dtype=float) - SPINE_CENTER
    design_matrix = np.column_stack((np.ones(len(x)), x, x**2))
    values = pivot.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_curvatures = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    bootstrap_optima = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(seeds), size=len(seeds))
        coefficients, _, rank, _ = np.linalg.lstsq(
            design_matrix,
            values[:, sampled].mean(axis=1),
            rcond=None,
        )
        if rank != design_matrix.shape[1] or not np.isfinite(coefficients).all():
            raise RuntimeError("Bootstrapped tied-spine quadratic failed")
        curvature = float(coefficients[2])
        bootstrap_curvatures[replicate] = curvature
        if curvature > 1e-12:
            bootstrap_optima[replicate] = SPINE_CENTER - float(coefficients[1]) / (2.0 * curvature)
    finite_optima = bootstrap_optima[np.isfinite(bootstrap_optima)]
    hull_min = min(aggregates)
    hull_max = max(aggregates)
    optimum_inside_hull_probability = float(
        np.mean(np.isfinite(bootstrap_optima) & (bootstrap_optima >= hull_min) & (bootstrap_optima <= hull_max))
    )
    terminal_diagnostic = diagnostics.loc[
        diagnostics["target"].eq(primary_target) & diagnostics["global_step"].eq(TRANSITION_MAX_STEP)
    ]
    if len(terminal_diagnostic) != 1:
        raise ValueError("Aggregate-spine diagnostic lacks the primary terminal transition step")
    terminal_lolo_rmse = float(terminal_diagnostic.iloc[0]["leave_one_level_out_rmse"])
    passed = terminal_lolo_rmse <= design.MAX_SPINE_LOLO_RMSE_BPB
    summary = {
        "terminal_step": TRANSITION_MAX_STEP,
        "terminal_leave_one_level_out_rmse": terminal_lolo_rmse,
        "maximum_leave_one_level_out_rmse": design.MAX_SPINE_LOLO_RMSE_BPB,
        "terminal_curvature_bootstrap_q05": float(np.quantile(bootstrap_curvatures, 0.05)),
        "terminal_curvature_bootstrap_median": float(np.median(bootstrap_curvatures)),
        "terminal_tied_optimum_bootstrap_inside_hull_probability": optimum_inside_hull_probability,
        "terminal_tied_optimum_stability_is_diagnostic_only": True,
        "terminal_tied_optimum_bootstrap_q05": (
            float(np.quantile(finite_optima, 0.05)) if len(finite_optima) else math.nan
        ),
        "terminal_tied_optimum_bootstrap_median": float(np.median(finite_optima)) if len(finite_optima) else math.nan,
        "terminal_tied_optimum_bootstrap_q95": (
            float(np.quantile(finite_optima, 0.95)) if len(finite_optima) else math.nan
        ),
        "aggregate_hull": [hull_min, hull_max],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "passed": passed,
    }
    return diagnostics, summary


def fit_tied_spines(
    raw: pd.DataFrame,
    manifest: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    excluded_seed: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit target- and step-specific local quadratics from tied policies only."""
    tied = collapsed_tied_values(raw, manifest)
    if excluded_seed is not None:
        tied = tied.loc[tied["run_seed"].astype(int).ne(excluded_seed)]
    collapsed = tied.groupby(
        ["target", "global_step", "aggregate_code_weight"],
        as_index=False,
    )["value"].mean()
    expected_aggregates = sorted(float(anchor["aggregate_code_weight"]) for anchor in protocol["anchors"])
    spines: dict[str, dict[str, dict[str, float]]] = {}
    for (target_name, global_step), group in collapsed.groupby(["target", "global_step"], sort=True):
        aggregate_values = sorted(group["aggregate_code_weight"].astype(float).tolist())
        if len(aggregate_values) != len(expected_aggregates) or not np.allclose(
            aggregate_values,
            expected_aggregates,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Tied spine lacks the five frozen aggregate levels for {target_name} step={global_step}: "
                f"{aggregate_values}"
            )
        spine = quadratic_spine_fit(group)
        spines.setdefault(str(target_name), {})[str(int(global_step))] = asdict(spine)
    return spines


def fit_tied_spine_bundle(
    raw: pd.DataFrame,
    manifest: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Fit full and leave-seed-out tied spines before reading asymmetric outcomes."""
    asymmetric_seeds = sorted(
        manifest.loc[~manifest["role"].astype(str).eq("spine_tied_control"), "run_seed"].astype(int).unique()
    )
    return {
        "full": fit_tied_spines(raw, manifest, protocol),
        "exclude_seed": {
            str(seed): fit_tied_spines(raw, manifest, protocol, excluded_seed=seed) for seed in asymmetric_seeds
        },
    }


def spine_fit_for_row(row: pd.Series, spines: dict[str, Any]) -> SpineFit:
    selected_spines = spines
    if "full" in spines:
        seed = str(int(row["run_seed"]))
        selected_spines = spines["exclude_seed"].get(seed, spines["full"])
    target_name = str(row["target"])
    requested_step = int(row["global_step"])
    available = selected_spines[target_name]
    step = requested_step if str(requested_step) in available else max(int(value) for value in available)
    if requested_step != step and requested_step <= TRANSITION_MAX_STEP:
        raise ValueError(f"Missing tied spine at non-final step {requested_step}")
    payload = available[str(step)]
    return SpineFit(**{key: float(value) for key, value in payload.items()})


def aggregate_gradient(frame: pd.DataFrame, spines: dict[str, Any]) -> np.ndarray:
    """Evaluate the cross-fitted tied-spine derivative once per row group."""
    gradients = np.empty(len(frame), dtype=float)
    groups = frame.groupby(["target", "global_step", "run_seed"], sort=False).indices
    aggregate = frame["aggregate_code_weight"].to_numpy(float)
    for positions in groups.values():
        indices = np.asarray(positions, dtype=int)
        fit = spine_fit_for_row(frame.iloc[int(indices[0])], spines)
        gradients[indices] = fit.slope + 2.0 * fit.curvature * (aggregate[indices] - fit.center)
    if not np.isfinite(gradients).all():
        raise ValueError("Aggregate-spine gradients are non-finite")
    return gradients


def candidate_grid(protocol: dict[str, Any]) -> tuple[Candidate, ...]:
    """Materialize the complete frozen mechanism and rate grid."""
    family = protocol["candidate_family"]
    candidates = [
        Candidate("token_dose_null"),
        Candidate("lr_mass_dose_null"),
        Candidate("terminal_level_null"),
    ]
    candidates.extend(
        Candidate("static_switch_control_null", response_mode=response_mode) for response_mode in CONTROL_RESPONSE_MODES
    )
    candidates.extend(
        Candidate("phase_local_repetition", repetition_power=int(power)) for power in family["repetition_powers"]
    )
    for clock in ("token", "lr_mass"):
        prefix = "token" if clock == "token" else "lr"
        for acquisition_rate in family["acquisition_rate_grid"]:
            for forgetting_ratio in family["forgetting_ratio_grid"]:
                candidates.extend(
                    Candidate(
                        f"{prefix}_clock_acquisition_forgetting",
                        clock=clock,
                        acquisition_rate=float(acquisition_rate),
                        forgetting_ratio=float(forgetting_ratio),
                        response_mode=response_mode,
                    )
                    for response_mode in CONTROL_RESPONSE_MODES
                )
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs are not unique")
    return tuple(candidates)


def lr_mass(protocol: dict[str, Any]) -> np.ndarray:
    schedule = protocol["schedule"]
    return np.asarray(
        design.normalized_lr_mass_by_step(
            total_steps=int(schedule["total_steps"]),
            warmup_steps=int(schedule["warmup_steps"]),
            decay_start_step=int(schedule["boundary_step"]),
        ),
        dtype=float,
    )


def segment_durations(
    *,
    eval_step: int,
    switch_step: int,
    clock: str,
    total_steps: int,
    cumulative_lr_mass: np.ndarray,
) -> tuple[float, float]:
    """Return completed phase-0 and phase-1 durations in the selected clock."""
    completed_steps = eval_step + 1
    phase_0_end = min(completed_steps, switch_step)
    if clock == "token":
        return phase_0_end / total_steps, max(completed_steps - switch_step, 0) / total_steps
    if clock == "lr_mass":
        phase_0 = float(cumulative_lr_mass[phase_0_end])
        phase_1 = 0.0
        if completed_steps > switch_step:
            phase_1 = float(cumulative_lr_mass[completed_steps] - cumulative_lr_mass[switch_step])
        return phase_0, phase_1
    raise ValueError(f"Unknown clock: {clock}")


def advance_acquisition(
    state: float,
    *,
    code_weight: float,
    duration: float,
    acquisition_rate: float,
    forgetting_ratio: float,
    maximum_code_epochs: float,
) -> tuple[float, float]:
    """Advance bounded acquisition and return the state plus its time integral."""
    if duration <= 0.0:
        return state, 0.0
    total_rate = acquisition_rate * maximum_code_epochs * (code_weight + forgetting_ratio * (1.0 - code_weight))
    if total_rate <= 0.0:
        return state, state * duration
    equilibrium = code_weight / (code_weight + forgetting_ratio * (1.0 - code_weight))
    decay = math.exp(-total_rate * duration)
    next_state = equilibrium + (state - equilibrium) * decay
    integral = equilibrium * duration + (state - equilibrium) * (1.0 - decay) / total_rate
    return next_state, integral


def dynamic_state(
    *,
    phase_0_code_weight: float,
    phase_1_code_weight: float,
    phase_0_duration: float,
    phase_1_duration: float,
    acquisition_rate: float,
    forgetting_ratio: float,
    maximum_code_epochs: float,
) -> float:
    """Evaluate the exact piecewise-constant bounded state transition."""
    x = 0.0
    for code_weight, duration in (
        (phase_0_code_weight, phase_0_duration),
        (phase_1_code_weight, phase_1_duration),
    ):
        x, _ = advance_acquisition(
            x,
            code_weight=code_weight,
            duration=duration,
            acquisition_rate=acquisition_rate,
            forgetting_ratio=forgetting_ratio,
            maximum_code_epochs=maximum_code_epochs,
        )
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"State invariant violated: x={x}")
    return x


def timescale_identification(
    frame: pd.DataFrame,
    candidate: Candidate,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Audit whether phase-0 memory and post-switch relaxation are resolved."""
    if candidate.family not in DYNAMIC_FAMILIES:
        return {"applicable": False, "passed": False, "reason": "selected candidate is not dynamic"}
    if candidate.acquisition_rate is None or candidate.forgetting_ratio is None:
        raise ValueError(f"Dynamic candidate lacks rates: {candidate.candidate_id}")
    total_steps = int(protocol["schedule"]["total_steps"])
    cumulative_lr_mass = lr_mass(protocol)
    coordinates = frame[
        [
            "coordinate_id",
            "switch_step",
            "phase_0_code_weight",
            "phase_1_code_weight",
            "code_subset_tokens",
        ]
    ].drop_duplicates()
    rows: list[dict[str, float | int | str]] = []
    for item in coordinates.itertuples(index=False):
        switch_step = int(item.switch_step)
        maximum_code_epochs = float(protocol["training_configuration"]["materialized_tokens"]) / float(
            item.code_subset_tokens
        )
        phase_0_duration = (
            switch_step / total_steps if candidate.clock == "token" else float(cumulative_lr_mass[switch_step])
        )
        phase_0_rate = (
            candidate.acquisition_rate
            * maximum_code_epochs
            * (float(item.phase_0_code_weight) + candidate.forgetting_ratio * (1.0 - float(item.phase_0_code_weight)))
        )
        initial_condition_memory = math.exp(-phase_0_rate * phase_0_duration)
        phase_1_rate = (
            candidate.acquisition_rate
            * maximum_code_epochs
            * (float(item.phase_1_code_weight) + candidate.forgetting_ratio * (1.0 - float(item.phase_1_code_weight)))
        )
        if phase_1_rate <= 0.0:
            relaxation_steps = math.inf
        elif candidate.clock == "token":
            relaxation_steps = total_steps / phase_1_rate
        else:
            target_mass = float(cumulative_lr_mass[switch_step]) + 1.0 / phase_1_rate
            relaxation_end = int(np.searchsorted(cumulative_lr_mass, target_mass, side="left"))
            relaxation_steps = math.inf if relaxation_end >= len(cumulative_lr_mass) else relaxation_end - switch_step
        rows.append(
            {
                "coordinate_id": str(item.coordinate_id),
                "switch_step": switch_step,
                "initial_condition_memory": initial_condition_memory,
                "relaxation_steps": float(relaxation_steps),
            }
        )
    audit = pd.DataFrame.from_records(rows)
    post_switch_step_counts = (
        frame.loc[frame["global_step"].astype(int).ge(frame["switch_step"].astype(int))]
        .groupby("switch_step")["global_step"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    folds: dict[str, Any] = {}
    informative_folds = 0
    for switch_step, group in audit.groupby("switch_step", sort=True):
        median_memory = float(group["initial_condition_memory"].median())
        median_relaxation = float(group["relaxation_steps"].median())
        post_switch_steps = int(post_switch_step_counts.get(int(switch_step), 0))
        eligible = post_switch_steps >= 2
        informative = (
            eligible
            and median_memory >= design.MIN_SWITCH_MEMORY_FRACTION
            and median_relaxation >= design.MIN_RELAXATION_STEPS
        )
        informative_folds += int(informative)
        folds[str(int(switch_step))] = {
            "coordinates": len(group),
            "post_switch_transition_steps": post_switch_steps,
            "eligible_for_timescale_identification": eligible,
            "median_initial_condition_memory": median_memory,
            "median_relaxation_steps": median_relaxation,
            "informative": informative,
        }
    return {
        "applicable": True,
        "clock": candidate.clock,
        "acquisition_rate": candidate.acquisition_rate,
        "forgetting_ratio": candidate.forgetting_ratio,
        "required_minimum_initial_condition_memory": design.MIN_SWITCH_MEMORY_FRACTION,
        "required_minimum_relaxation_steps": design.MIN_RELAXATION_STEPS,
        "required_informative_switch_folds": design.MIN_MEMORY_SWITCH_FOLDS,
        "informative_switch_folds": informative_folds,
        "folds": folds,
        "passed": informative_folds >= design.MIN_MEMORY_SWITCH_FOLDS,
    }


def control_potential_columns(
    displacement: np.ndarray,
    aggregate_derivative: np.ndarray,
    aggregate: np.ndarray,
    candidate: Candidate,
) -> np.ndarray:
    """Return one of the frozen signed or constrained temporal response bases."""
    if candidate.response_mode == "potential_constrained":
        return np.column_stack((aggregate_derivative * displacement, displacement**2))
    if candidate.response_mode == "aggregate_linear_signed":
        return np.column_stack(
            (
                displacement,
                (aggregate - SPINE_CENTER) * displacement,
                displacement**2,
            )
        )
    elif candidate.response_mode == "unconditioned_signed":
        return np.column_stack((displacement, displacement**2))
    else:
        raise ValueError(f"Unknown temporal response mode: {candidate.response_mode}")


def feature_matrix(
    frame: pd.DataFrame,
    candidate: Candidate,
    protocol: dict[str, Any],
    spines: dict[str, Any],
    *,
    aggregate_derivative: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized counterfactually tied-centered features for one candidate."""
    cumulative_lr_mass = lr_mass(protocol)
    eval_steps = frame["global_step"].to_numpy(int)
    switch_steps = frame["switch_step"].to_numpy(int)
    completed_steps = eval_steps + 1
    total_steps = int(protocol["schedule"]["total_steps"])
    phase_0 = frame["phase_0_code_weight"].to_numpy(float)
    phase_1 = frame["phase_1_code_weight"].to_numpy(float)
    aggregate = frame["aggregate_code_weight"].to_numpy(float)
    maximum_code_epochs = float(protocol["training_configuration"]["materialized_tokens"]) / frame[
        "code_subset_tokens"
    ].to_numpy(float)

    def tied_spine_derivative() -> np.ndarray:
        if aggregate_derivative is None:
            return aggregate_gradient(frame, spines)
        if aggregate_derivative.shape != (len(frame),):
            raise ValueError(f"Aggregate derivative shape mismatch: {aggregate_derivative.shape}")
        return aggregate_derivative

    def durations(clock: str) -> tuple[np.ndarray, np.ndarray]:
        phase_0_end = np.minimum(completed_steps, switch_steps)
        if clock == "token":
            return phase_0_end / total_steps, np.maximum(completed_steps - switch_steps, 0) / total_steps
        if clock == "lr_mass":
            first = cumulative_lr_mass[phase_0_end]
            second = np.where(
                completed_steps > switch_steps,
                cumulative_lr_mass[completed_steps] - cumulative_lr_mass[switch_steps],
                0.0,
            )
            return first, second
        raise ValueError(f"Unknown clock: {clock}")

    if candidate.family in {"token_dose_null", "lr_mass_dose_null"}:
        clock = "token" if candidate.family == "token_dose_null" else "lr_mass"
        d0, d1 = durations(clock)
        value = maximum_code_epochs * (phase_0 * d0 + phase_1 * d1 - aggregate * (d0 + d1))
        matrix = value[:, None]
    elif candidate.family == "terminal_level_null":
        current = np.where(completed_steps <= switch_steps, phase_0, phase_1)
        matrix = (current - aggregate)[:, None]
    elif candidate.family == "phase_local_repetition":
        d0, d1 = durations("token")
        power = int(candidate.repetition_power or 1)
        asymmetric = np.maximum(maximum_code_epochs * phase_0 * d0 - design.REPETITION_ONSET_EPOCHS, 0.0) ** power
        asymmetric += np.maximum(maximum_code_epochs * phase_1 * d1 - design.REPETITION_ONSET_EPOCHS, 0.0) ** power
        tied = np.maximum(maximum_code_epochs * aggregate * d0 - design.REPETITION_ONSET_EPOCHS, 0.0) ** power
        tied += np.maximum(maximum_code_epochs * aggregate * d1 - design.REPETITION_ONSET_EPOCHS, 0.0) ** power
        matrix = (asymmetric - tied)[:, None]
    elif candidate.family == "static_switch_control_null":
        lr_at_switch = np.asarray(
            [
                design.learning_rate_multiplier(
                    int(step),
                    total_steps=total_steps,
                    warmup_steps=int(protocol["schedule"]["warmup_steps"]),
                    decay_start_step=int(protocol["schedule"]["boundary_step"]),
                )
                for step in switch_steps
            ],
            dtype=float,
        )
        displacement = np.where(eval_steps >= switch_steps, lr_at_switch * (phase_1 - phase_0), 0.0)
        matrix = control_potential_columns(displacement, tied_spine_derivative(), aggregate, candidate)
    elif candidate.family in DYNAMIC_FAMILIES:
        if candidate.acquisition_rate is None or candidate.forgetting_ratio is None:
            raise ValueError(f"Dynamic candidate lacks rates: {candidate}")
        d0, d1 = durations(candidate.clock)

        def states(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
            x = np.zeros(len(frame), dtype=float)
            for code_weight, duration in ((q0, d0), (q1, d1)):
                total_rate = (
                    candidate.acquisition_rate
                    * maximum_code_epochs
                    * (code_weight + candidate.forgetting_ratio * (1.0 - code_weight))
                )
                equilibrium = np.divide(
                    code_weight,
                    code_weight + candidate.forgetting_ratio * (1.0 - code_weight),
                    out=np.zeros_like(code_weight),
                    where=total_rate > 0.0,
                )
                decay = np.exp(-total_rate * duration)
                next_x = equilibrium + (x - equilibrium) * decay
                x = next_x
            return x

        displacement = states(phase_0, phase_1) - states(aggregate, aggregate)
        matrix = control_potential_columns(displacement, tied_spine_derivative(), aggregate, candidate)
    else:
        raise ValueError(f"Unknown candidate family: {candidate.family}")
    if matrix.shape != (len(frame), candidate.feature_dimension):
        raise ValueError(f"Feature shape mismatch for {candidate.candidate_id}: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite feature for {candidate.candidate_id}")
    return matrix


def control_state_key(candidate: Candidate) -> tuple[str, str, float | None, float | None]:
    """Identify candidates that share the same latent temporal displacement."""
    if candidate.family not in CONTROL_POTENTIAL_FAMILIES:
        raise ValueError(f"Candidate has no control-potential state: {candidate.candidate_id}")
    return (
        candidate.family,
        candidate.clock,
        candidate.acquisition_rate,
        candidate.forgetting_ratio,
    )


def unconditioned_state_candidate(candidate: Candidate) -> Candidate:
    """Return the state-equivalent candidate whose first feature is displacement."""
    return Candidate(
        family=candidate.family,
        clock=candidate.clock,
        acquisition_rate=candidate.acquisition_rate,
        forgetting_ratio=candidate.forgetting_ratio,
        repetition_power=candidate.repetition_power,
        response_mode="unconditioned_signed",
    )


def feature_cache_for_candidates(
    frame: pd.DataFrame,
    candidates: Sequence[Candidate],
    protocol: dict[str, Any],
    spines: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build features while evaluating each latent transition only once."""
    candidates = tuple(candidates)
    has_control_state = any(candidate.family in CONTROL_POTENTIAL_FAMILIES for candidate in candidates)
    derivative = aggregate_gradient(frame, spines) if has_control_state else None
    aggregate = frame["aggregate_code_weight"].to_numpy(float)
    displacement_cache: dict[tuple[str, str, float | None, float | None], np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    for candidate in candidates:
        if candidate.family not in CONTROL_POTENTIAL_FAMILIES:
            features[candidate.candidate_id] = feature_matrix(
                frame,
                candidate,
                protocol,
                spines,
                aggregate_derivative=derivative,
            )
            continue
        key = control_state_key(candidate)
        if key not in displacement_cache:
            state_features = feature_matrix(
                frame,
                unconditioned_state_candidate(candidate),
                protocol,
                spines,
                aggregate_derivative=derivative,
            )
            displacement_cache[key] = state_features[:, 0]
        if derivative is None:
            raise AssertionError("Control-potential feature cache lacks the tied-spine derivative")
        features[candidate.candidate_id] = control_potential_columns(
            displacement_cache[key],
            derivative,
            aggregate,
            candidate,
        )
    if len(features) != len(candidates):
        raise ValueError("Feature cache lost a candidate or contains duplicate candidate IDs")
    return features


def feature_subspace_separation(
    dynamic_features: np.ndarray,
    memoryless_features: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Measure dynamic feature energy outside the joint memoryless-null span."""
    if len(dynamic_features) != len(memoryless_features) or dynamic_features.ndim != 2 or dynamic_features.shape[1] < 2:
        raise ValueError("Feature-separation audit received incompatible row counts")

    def normalized_weighted_columns(features: np.ndarray) -> np.ndarray:
        weighted = np.sqrt(weights)[:, None] * features
        norms = np.linalg.norm(weighted, axis=0)
        retained = norms > 1e-12
        if not retained.any():
            return np.empty((len(features), 0), dtype=float)
        return weighted[:, retained] / norms[retained]

    dynamic = normalized_weighted_columns(dynamic_features)
    memoryless = normalized_weighted_columns(memoryless_features)
    if dynamic.shape[1] == 0 or memoryless.shape[1] == 0:
        return {
            "projection_residual": 0.0,
            "minimum_principal_angle_degrees": 0.0,
            "maximum_principal_angle_degrees": 0.0,
        }
    dynamic_u, dynamic_s, _ = np.linalg.svd(dynamic, full_matrices=False)
    memoryless_u, memoryless_s, _ = np.linalg.svd(memoryless, full_matrices=False)
    dynamic_rank = int(np.sum(dynamic_s > 1e-10 * dynamic_s[0]))
    memoryless_rank = int(np.sum(memoryless_s > 1e-10 * memoryless_s[0]))
    dynamic_basis = dynamic_u[:, :dynamic_rank]
    memoryless_basis = memoryless_u[:, :memoryless_rank]
    residual = dynamic - memoryless_basis @ (memoryless_basis.T @ dynamic)
    projection_residual = float(np.linalg.norm(residual) / np.linalg.norm(dynamic))
    singular_values = np.linalg.svd(memoryless_basis.T @ dynamic_basis, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(singular_values, -1.0, 1.0)))
    return {
        "projection_residual": projection_residual,
        "minimum_principal_angle_degrees": float(np.min(angles)),
        "maximum_principal_angle_degrees": float(np.max(angles)),
    }


def dynamic_static_separation_table(
    frame: pd.DataFrame,
    candidates: Sequence[Candidate],
    feature_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Audit dynamic-state identifiability against the joint memoryless-null span."""
    memoryless = [candidate for candidate in candidates if candidate.family in MEMORYLESS_FAMILIES]
    memoryless_features = np.column_stack([feature_cache[candidate.candidate_id] for candidate in memoryless])
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.family not in DYNAMIC_FAMILIES:
            continue
        dynamic_features = feature_cache[candidate.candidate_id]
        global_metrics = feature_subspace_separation(
            dynamic_features,
            memoryless_features,
            coordinate_weights(frame),
        )
        fold_metrics: list[dict[str, float | int | bool]] = []
        for switch_step in sorted(frame["switch_step"].astype(int).unique()):
            mask = frame["switch_step"].astype(int).to_numpy() == switch_step
            metrics = feature_subspace_separation(
                dynamic_features[mask],
                memoryless_features[mask],
                coordinate_weights(frame.loc[mask]),
            )
            post_switch_steps = int(
                frame.loc[
                    mask & frame["global_step"].astype(int).ge(frame["switch_step"].astype(int)), "global_step"
                ].nunique()
            )
            fold_metrics.append(
                {
                    "switch_step": int(switch_step),
                    "post_switch_transition_steps": post_switch_steps,
                    "eligible": post_switch_steps >= 2,
                    **metrics,
                }
            )
        fold_residuals = [float(item["projection_residual"]) for item in fold_metrics]
        eligible_fold_residuals = [float(item["projection_residual"]) for item in fold_metrics if bool(item["eligible"])]
        if not eligible_fold_residuals:
            raise ValueError("Dynamic-static separation has no switch fold with two post-switch observations")
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "clock": candidate.clock,
                "acquisition_rate": candidate.acquisition_rate,
                "forgetting_ratio": candidate.forgetting_ratio,
                "response_mode": candidate.response_mode,
                "global_projection_residual": global_metrics["projection_residual"],
                "global_minimum_principal_angle_degrees": global_metrics["minimum_principal_angle_degrees"],
                "global_maximum_principal_angle_degrees": global_metrics["maximum_principal_angle_degrees"],
                "minimum_switch_projection_residual": min(fold_residuals),
                "minimum_eligible_switch_projection_residual": min(eligible_fold_residuals),
                "median_switch_projection_residual": float(np.median(fold_residuals)),
                "maximum_switch_projection_residual": max(fold_residuals),
                "eligible_switch_folds": len(eligible_fold_residuals),
                "separated_eligible_switch_folds": sum(
                    value >= design.MIN_STATIC_SUBSPACE_RESIDUAL for value in eligible_fold_residuals
                ),
                "passes_static_separation_floor": min(eligible_fold_residuals) >= design.MIN_STATIC_SUBSPACE_RESIDUAL,
                "switch_fold_metrics_json": json.dumps(fold_metrics, sort_keys=True),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("candidate_id").reset_index(drop=True)


def sealed_feature_design_frame(manifest: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    """Build the outcome-free sealed design used to audit extrapolation identifiability."""
    asymmetric = manifest.loc[~manifest["role"].astype(str).eq("spine_tied_control")].copy()
    steps = pd.DataFrame({"global_step": sealed_decay_steps(protocol)})
    frame = asymmetric.merge(steps, how="cross")
    frame["target"] = str(protocol["targets"]["primary"])
    return frame.sort_values(["switch_step", "coordinate_id", "run_seed", "global_step"]).reset_index(drop=True)


def coordinate_weights(frame: pd.DataFrame) -> np.ndarray:
    """Balance pre/post limbs, then independently trained coordinates within each limb."""
    limbs = np.where(
        frame["global_step"].astype(int).to_numpy() < frame["switch_step"].astype(int).to_numpy(),
        "pre",
        "post",
    )
    keyed = frame.assign(_limb=limbs)
    row_counts = keyed.groupby(["_limb", "coordinate_id"])["coordinate_id"].transform("size").to_numpy(float)
    coordinates_per_limb = keyed.groupby("_limb")["coordinate_id"].transform("nunique").to_numpy(float)
    weights = 1.0 / (row_counts * coordinates_per_limb)
    return weights / weights.sum()


def fit_head(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    candidate: Candidate,
) -> HeadFit | None:
    weighted_features = np.sqrt(weights)[:, None] * features
    weighted_target = np.sqrt(weights) * target
    singular_values = np.linalg.svd(weighted_features, compute_uv=False)
    if len(singular_values) == 0 or singular_values[-1] <= 1e-12:
        return None
    condition = float(singular_values[0] / singular_values[-1])
    if not math.isfinite(condition) or condition > MAX_TWO_STATE_CONDITION:
        return None
    gram = weighted_features.T @ weighted_features
    rhs = weighted_features.T @ weighted_target
    lower_bounds = np.asarray(candidate.coefficient_lower_bounds, dtype=float)
    if lower_bounds.shape != (features.shape[1],):
        raise ValueError(f"Coefficient-bound shape mismatch for {candidate.candidate_id}")
    bounded = tuple(int(index) for index in np.flatnonzero(np.isfinite(lower_bounds)))
    if bounded:
        active_solutions: list[np.ndarray] = []
        for active_mask in range(1 << len(bounded)):
            active = {bounded[index] for index in range(len(bounded)) if active_mask & (1 << index)}
            free = [index for index in range(features.shape[1]) if index not in active]
            coefficients = np.zeros(features.shape[1], dtype=float)
            if free:
                try:
                    coefficients[free] = np.linalg.solve(gram[np.ix_(free, free)], rhs[free])
                except np.linalg.LinAlgError:
                    continue
            if np.any(coefficients < lower_bounds - 1e-12):
                continue
            active_solutions.append(coefficients)
        if not active_solutions:
            return None
        coefficients = min(
            active_solutions,
            key=lambda values: float(np.sum((weighted_features @ values - weighted_target) ** 2)),
        )
    else:
        try:
            coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return None
    if not np.isfinite(coefficients).all():
        return None
    return HeadFit(tuple(float(value) for value in coefficients), condition)


def predict(features: np.ndarray, fit: HeadFit) -> np.ndarray:
    return features @ np.asarray(fit.coefficients, dtype=float)


def limb_balanced_rmse(frame: pd.DataFrame, residual: np.ndarray) -> float:
    """Return RMSE under the same pre/post and coordinate balance used for fitting."""
    return float(np.sqrt(np.sum(coordinate_weights(frame) * residual**2)))


def coordinate_rmse(frame: pd.DataFrame, residual: np.ndarray) -> float:
    """Return an unweighted per-coordinate diagnostic RMSE."""
    squared = pd.DataFrame({"coordinate_id": frame["coordinate_id"].astype(str).to_numpy(), "squared": residual**2})
    return float(math.sqrt(squared.groupby("coordinate_id")["squared"].mean().mean()))


def oof_predictions(
    frame: pd.DataFrame,
    features: np.ndarray,
    candidate: Candidate,
    *,
    weights: np.ndarray | None = None,
    fold_column: str = "switch_step",
) -> np.ndarray | None:
    predictions = np.full(len(frame), np.nan, dtype=float)
    target = frame["observed_delta"].to_numpy(float)
    if weights is None:
        weights = coordinate_weights(frame)
    folds = sorted(frame[fold_column].unique())
    for fold in folds:
        heldout = frame[fold_column].to_numpy() == fold
        training = ~heldout
        fit = fit_head(features[training], target[training], weights[training], candidate)
        if fit is None:
            return None
        predictions[heldout] = predict(features[heldout], fit)
    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions are incomplete")
    return predictions


def select_candidate(
    frame: pd.DataFrame,
    candidates: Sequence[Candidate],
    features: dict[str, np.ndarray],
    *,
    fold_column: str = "switch_step",
) -> tuple[Candidate, pd.DataFrame]:
    """Select the minimum leave-switch-out RMSE candidate on the supplied rows."""
    rows: list[dict[str, Any]] = []
    valid_candidates: list[tuple[float, Candidate]] = []
    target = frame["observed_delta"].to_numpy(float)
    weights = coordinate_weights(frame)
    for candidate in candidates:
        predictions = oof_predictions(
            frame,
            features[candidate.candidate_id],
            candidate,
            weights=weights,
            fold_column=fold_column,
        )
        if predictions is None:
            continue
        rmse = limb_balanced_rmse(frame, predictions - target)
        rows.append({"candidate_id": candidate.candidate_id, "family": candidate.family, "rmse": rmse})
        valid_candidates.append((rmse, candidate))
    if not valid_candidates:
        raise RuntimeError("No candidate produced a valid leave-switch-out fit")
    minimum_rmse = min(item[0] for item in valid_candidates)
    numerical_ties = [candidate for rmse, candidate in valid_candidates if rmse <= minimum_rmse + 1e-12]
    selected = min(numerical_ties, key=lambda candidate: (candidate.feature_dimension, candidate.candidate_id))
    return selected, pd.DataFrame.from_records(rows).sort_values(["rmse", "candidate_id"])


def nested_family_predictions(
    frame: pd.DataFrame,
    candidates: Sequence[Candidate],
    features: dict[str, np.ndarray],
    *,
    outer_fold_column: str = "switch_step",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return fully nested outer-fold predictions and selected candidates."""
    predictions = np.full(len(frame), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    target = frame["observed_delta"].to_numpy(float)
    folds = sorted(frame[outer_fold_column].unique())
    for outer_fold in folds:
        outer = frame[outer_fold_column].to_numpy() == outer_fold
        training_indices = np.flatnonzero(~outer)
        training_frame = frame.iloc[training_indices].reset_index(drop=True)
        training_features = {key: value[training_indices] for key, value in features.items()}
        selected, _ = select_candidate(training_frame, candidates, training_features)
        fit = fit_head(
            features[selected.candidate_id][~outer],
            target[~outer],
            coordinate_weights(frame.loc[~outer]),
            selected,
        )
        if fit is None:
            raise RuntimeError(f"Selected outer-fold candidate became singular: {selected.candidate_id}")
        predictions[outer] = predict(features[selected.candidate_id][outer], fit)
        selections.append(
            {
                "outer_switch_step": int(outer_fold) if outer_fold_column == "switch_step" else None,
                "outer_fold_column": outer_fold_column,
                "outer_fold_value": str(outer_fold),
                **asdict(selected),
                "candidate_id": selected.candidate_id,
                "response_coefficients": list(fit.coefficients),
            }
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Nested predictions are incomplete")
    return predictions, selections


def response_form_comparison(
    frame: pd.DataFrame,
    selected: Candidate,
    candidates: Sequence[Candidate],
    features: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare response heads while holding the selected latent state fixed."""
    if selected.family not in CONTROL_POTENTIAL_FAMILIES:
        return {"applicable": False, "passed": False, "reason": "selected candidate has no response-mode family"}
    state_key = control_state_key(selected)
    alternatives = tuple(
        candidate
        for candidate in candidates
        if candidate.family in CONTROL_POTENTIAL_FAMILIES
        and control_state_key(candidate) == state_key
        and candidate.candidate_id != selected.candidate_id
    )
    if {candidate.response_mode for candidate in alternatives} != set(CONTROL_RESPONSE_MODES) - {selected.response_mode}:
        raise ValueError(f"Response-form alternatives are incomplete for {selected.candidate_id}")
    target = frame["observed_delta"].to_numpy(float)
    folds: dict[str, Any] = {}
    direct_advantage_pass = True
    for fold_column in ("switch_step", "anchor_id"):
        selected_prediction = oof_predictions(
            frame,
            features[selected.candidate_id],
            selected,
            fold_column=fold_column,
        )
        if selected_prediction is None:
            raise RuntimeError(f"Selected response form is singular for {fold_column}")
        selected_rmse = limb_balanced_rmse(frame, selected_prediction - target)
        alternative_predictions: dict[str, np.ndarray] = {}
        alternative_rmses: dict[str, float] = {}
        for alternative in alternatives:
            prediction = oof_predictions(
                frame,
                features[alternative.candidate_id],
                alternative,
                fold_column=fold_column,
            )
            if prediction is None:
                continue
            alternative_predictions[alternative.candidate_id] = prediction
            alternative_rmses[alternative.candidate_id] = limb_balanced_rmse(frame, prediction - target)
        if not alternative_rmses:
            raise RuntimeError(f"No state-matched response alternative is identifiable for {fold_column}")
        strongest_alternative = min(alternative_rmses, key=alternative_rmses.__getitem__)
        bootstrap = cluster_bootstrap_rmse_difference(
            frame,
            selected_prediction - target,
            alternative_predictions[strongest_alternative] - target,
        )
        fold_pass = selected_rmse <= 0.95 * alternative_rmses[strongest_alternative] and bootstrap["ci_high"] < 0.0
        folds[fold_column] = {
            "selected_rmse": selected_rmse,
            "alternative_rmses": alternative_rmses,
            "strongest_alternative": strongest_alternative,
            "candidate_minus_alternative_bootstrap": bootstrap,
            "passed": fold_pass,
        }
        direct_advantage_pass &= fold_pass
    return {
        "applicable": True,
        "selected_response_mode": selected.response_mode,
        "state_key": list(state_key),
        "folds": folds,
        "direct_advantage_pass": direct_advantage_pass,
        "diagnostic_only": True,
    }


def nested_selection_stability(
    selected: Candidate,
    switch_selections: Sequence[dict[str, Any]],
    anchor_selections: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Require every blocked endpoint model to retain the licensed mechanism."""
    rows = [
        {"fold_scheme": scheme, **selection}
        for scheme, selections in (("switch_step", switch_selections), ("anchor_id", anchor_selections))
        for selection in selections
    ]
    if not rows:
        raise ValueError("Nested selection stability received no outer folds")
    family_counts = Counter(str(row["family"]) for row in rows)
    clock_counts = Counter(str(row["clock"]) for row in rows)
    response_mode_counts = Counter(str(row["response_mode"]) for row in rows)
    dynamic_all = all(str(row["family"]) in DYNAMIC_FAMILIES for row in rows)
    family_match = all(str(row["family"]) == selected.family for row in rows)
    clock_match = all(str(row["clock"]) == selected.clock for row in rows)
    response_mode_match = all(str(row["response_mode"]) == selected.response_mode for row in rows)
    return {
        "full_selected_candidate": selected.candidate_id,
        "outer_folds": len(rows),
        "family_counts": dict(family_counts),
        "clock_counts": dict(clock_counts),
        "response_mode_counts": dict(response_mode_counts),
        "dynamic_in_every_fold": dynamic_all,
        "family_matches_full_selection": family_match,
        "clock_matches_full_selection": clock_match,
        "response_mode_matches_full_selection": response_mode_match,
        "rate_selections": [
            {
                "fold_scheme": str(row["fold_scheme"]),
                "fold_value": str(row["outer_fold_value"]),
                "acquisition_rate": row["acquisition_rate"],
                "forgetting_ratio": row["forgetting_ratio"],
            }
            for row in rows
        ],
        "passed": dynamic_all and response_mode_match,
        "licensing_rule": "all folds dynamic with the same response mode; family, clock, and rates are diagnostic",
    }


def freeze_blocked_fold_models(
    paired: pd.DataFrame,
    *,
    protocol: dict[str, Any],
    primary_selections: Sequence[dict[str, Any]],
    null_candidates: Sequence[Candidate],
    spines: dict[str, Any],
    fold_column: str,
) -> dict[str, dict[str, Any]]:
    """Fit target heads without the fold to which each model will be applied."""
    selection_by_fold = {
        str(selection["outer_fold_value"]): candidate_from_dict(selection) for selection in primary_selections
    }
    expected_folds = set(paired[fold_column].astype(str).unique())
    if set(selection_by_fold) != expected_folds:
        raise ValueError(f"Primary nested selections do not cover every asymmetric {fold_column} fold")

    frozen: dict[str, dict[str, Any]] = {}
    selected_candidates = tuple(selection_by_fold.values())
    for target_name in target_columns(protocol):
        target_frame = paired.loc[paired["target"].eq(target_name)].reset_index(drop=True)
        target = target_frame["observed_delta"].to_numpy(float)
        target_models: dict[str, Any] = {}
        feature_candidates = tuple(
            {candidate.candidate_id: candidate for candidate in (*selected_candidates, *null_candidates)}.values()
        )
        target_feature_cache = feature_cache_for_candidates(target_frame, feature_candidates, protocol, spines)
        fold_values = target_frame[fold_column].astype(str).to_numpy()
        for fold_value, selected in sorted(selection_by_fold.items()):
            heldout = fold_values == fold_value
            training = ~heldout
            selected_features = target_feature_cache[selected.candidate_id]
            selected_head = fit_head(
                selected_features[training],
                target[training],
                coordinate_weights(target_frame.loc[training]),
                selected,
            )
            if selected_head is None:
                raise RuntimeError(f"Blocked selected head is singular: {target_name} {fold_column}={fold_value}")
            blocked_null_heads: dict[str, Any] = {}
            for null_candidate in null_candidates:
                null_head = fit_head(
                    target_feature_cache[null_candidate.candidate_id][training],
                    target[training],
                    coordinate_weights(target_frame.loc[training]),
                    null_candidate,
                )
                if null_head is None:
                    raise RuntimeError(
                        f"Blocked null head is singular: {target_name} {fold_column}={fold_value} "
                        f"candidate={null_candidate.candidate_id}"
                    )
                blocked_null_heads[null_candidate.candidate_id] = asdict(null_head)
            target_models[fold_value] = {
                "selected_candidate": {**asdict(selected), "candidate_id": selected.candidate_id},
                "selected_head": asdict(selected_head),
                "null_heads": blocked_null_heads,
            }
        frozen[target_name] = target_models
    return frozen


def predict_blocked_folds(
    frame: pd.DataFrame,
    *,
    protocol: dict[str, Any],
    frozen_models: dict[str, Any],
    null_candidates: Sequence[Candidate],
    spines: dict[str, Any],
    fold_column: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply each frozen model only to its held-out fold."""
    selected_predictions = np.full(len(frame), np.nan, dtype=float)
    selected_candidate_ids = np.full(len(frame), "", dtype=object)
    null_predictions = {
        candidate.candidate_id: np.full(len(frame), np.nan, dtype=float) for candidate in null_candidates
    }
    selected_candidates = tuple(candidate_from_dict(payload["selected_candidate"]) for payload in frozen_models.values())
    feature_candidates = tuple(
        {candidate.candidate_id: candidate for candidate in (*selected_candidates, *null_candidates)}.values()
    )
    feature_cache = feature_cache_for_candidates(frame, feature_candidates, protocol, spines)
    fold_values = frame[fold_column].astype(str).to_numpy()
    for fold_value, payload in frozen_models.items():
        heldout = fold_values == str(fold_value)
        if not heldout.any():
            raise ValueError(f"Frozen {fold_column} fold {fold_value} has no rows")
        selected = candidate_from_dict(payload["selected_candidate"])
        selected_features = feature_cache[selected.candidate_id]
        selected_predictions[heldout] = predict(selected_features[heldout], head_from_dict(payload["selected_head"]))
        selected_candidate_ids[heldout] = selected.candidate_id
        for null_candidate in null_candidates:
            null_predictions[null_candidate.candidate_id][heldout] = predict(
                feature_cache[null_candidate.candidate_id][heldout],
                head_from_dict(payload["null_heads"][null_candidate.candidate_id]),
            )
    if not np.isfinite(selected_predictions).all() or np.any(selected_candidate_ids == ""):
        raise ValueError("Blocked selected predictions are incomplete")
    if any(not np.isfinite(values).all() for values in null_predictions.values()):
        raise ValueError("Blocked null predictions are incomplete")
    return selected_predictions, selected_candidate_ids, null_predictions


def cluster_bootstrap_rmse_difference(
    frame: pd.DataFrame,
    candidate_residual: np.ndarray,
    reference_residual: np.ndarray,
) -> dict[str, float | int]:
    """Bootstrap anchor-seed blocks sharing the same observed tied control."""
    required = ("anchor_id", "run_seed", "coordinate_id", "global_step", "switch_step")
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"Tied-control cluster bootstrap lacks columns: {sorted(missing)}")
    raw = frame[list(required)].reset_index(drop=True).copy()
    raw["metric_weight"] = coordinate_weights(frame)
    raw["candidate_squared"] = raw["metric_weight"] * candidate_residual**2
    raw["reference_squared"] = raw["metric_weight"] * reference_residual**2
    block_rows: list[dict[str, float | int | str]] = []
    for (anchor_id, run_seed), block in raw.groupby(["anchor_id", "run_seed"], sort=True):
        block_rows.append(
            {
                "anchor_id": str(anchor_id),
                "run_seed": int(run_seed),
                "metric_weight": float(block["metric_weight"].sum()),
                "candidate_squared": float(block["candidate_squared"].sum()),
                "reference_squared": float(block["reference_squared"].sum()),
            }
        )
    values = pd.DataFrame.from_records(block_rows)
    if len(values) < 8:
        raise ValueError(f"Tied-control cluster bootstrap has only {len(values)} anchor-seed blocks")
    block_weights = values["metric_weight"].to_numpy(float)
    candidate_sums = values["candidate_squared"].to_numpy(float)
    reference_sums = values["reference_squared"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    sampled_weights = block_weights[draws].sum(axis=1)
    differences = np.sqrt(candidate_sums[draws].sum(axis=1) / sampled_weights) - np.sqrt(
        reference_sums[draws].sum(axis=1) / sampled_weights
    )
    return {
        "estimate": float(math.sqrt(candidate_sums.sum()) - math.sqrt(reference_sums.sum())),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "clusters": len(values),
    }


def candidate_from_dict(value: dict[str, Any]) -> Candidate:
    fields = {
        key: value.get(key)
        for key in (
            "family",
            "clock",
            "acquisition_rate",
            "forgetting_ratio",
            "repetition_power",
            "response_mode",
        )
    }
    return Candidate(**fields)


def frozen_null_candidates(protocol: dict[str, Any]) -> tuple[Candidate, ...]:
    """Return the predeclared memoryless endpoint comparators."""
    all_candidates = candidate_grid(protocol)
    return tuple(candidate for candidate in all_candidates if candidate.family in MEMORYLESS_FAMILIES)


def transition_gate_summary(
    *,
    frame: pd.DataFrame,
    nested_predictions: np.ndarray,
    anchor_nested_predictions: np.ndarray,
    null_predictions: dict[str, np.ndarray],
    anchor_null_predictions: dict[str, np.ndarray],
    selections: list[dict[str, Any]],
    clock_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    target = frame["observed_delta"].to_numpy(float)
    zero = np.zeros_like(target)
    nested_rmse = limb_balanced_rmse(frame, nested_predictions - target)
    null_rmses = {
        candidate_id: limb_balanced_rmse(frame, values - target) for candidate_id, values in null_predictions.items()
    }
    strongest_null_id = min(null_rmses, key=null_rmses.__getitem__)
    strongest_null_rmse = null_rmses[strongest_null_id]
    zero_rmse = limb_balanced_rmse(frame, zero - target)
    bootstrap = cluster_bootstrap_rmse_difference(
        frame,
        nested_predictions - target,
        null_predictions[strongest_null_id] - target,
    )
    anchor_nested_rmse = limb_balanced_rmse(frame, anchor_nested_predictions - target)
    anchor_null_rmses = {
        candidate_id: limb_balanced_rmse(frame, values - target)
        for candidate_id, values in anchor_null_predictions.items()
    }
    strongest_anchor_null_id = min(anchor_null_rmses, key=anchor_null_rmses.__getitem__)
    anchor_bootstrap = cluster_bootstrap_rmse_difference(
        frame,
        anchor_nested_predictions - target,
        anchor_null_predictions[strongest_anchor_null_id] - target,
    )

    history_mask = frame["design_arm"].isin({"fixed_late_mixture", "fixed_early_mixture"}).to_numpy()
    terminal_id = Candidate("terminal_level_null").candidate_id
    history_candidate_rmse = limb_balanced_rmse(
        frame.loc[history_mask], nested_predictions[history_mask] - target[history_mask]
    )
    history_terminal_rmse = limb_balanced_rmse(
        frame.loc[history_mask], null_predictions[terminal_id][history_mask] - target[history_mask]
    )
    history_bootstrap = cluster_bootstrap_rmse_difference(
        frame.loc[history_mask],
        nested_predictions[history_mask] - target[history_mask],
        null_predictions[terminal_id][history_mask] - target[history_mask],
    )
    history_null_rmses = {
        candidate_id: limb_balanced_rmse(
            frame.loc[history_mask],
            values[history_mask] - target[history_mask],
        )
        for candidate_id, values in null_predictions.items()
    }
    strongest_history_null_id = min(history_null_rmses, key=history_null_rmses.__getitem__)
    strongest_history_bootstrap = cluster_bootstrap_rmse_difference(
        frame.loc[history_mask],
        nested_predictions[history_mask] - target[history_mask],
        null_predictions[strongest_history_null_id][history_mask] - target[history_mask],
    )

    static_ids = [candidate_id for candidate_id in null_rmses if candidate_id.startswith("static_switch_control_null")]
    static_id = min(static_ids, key=null_rmses.__getitem__)
    static_rmse = null_rmses[static_id]
    dynamic_minus_static = cluster_bootstrap_rmse_difference(
        frame,
        nested_predictions - target,
        null_predictions[static_id] - target,
    )
    history_static_rmse = limb_balanced_rmse(
        frame.loc[history_mask],
        null_predictions[static_id][history_mask] - target[history_mask],
    )
    history_static_bootstrap = cluster_bootstrap_rmse_difference(
        frame.loc[history_mask],
        nested_predictions[history_mask] - target[history_mask],
        null_predictions[static_id][history_mask] - target[history_mask],
    )
    history_fold_counts = frame.loc[history_mask].groupby("switch_step")["coordinate_id"].nunique().astype(int).to_dict()
    expected_history_folds = set(design.FIXED_LATE_SWITCH_STEPS) | set(design.FIXED_EARLY_SWITCH_STEPS)
    history_design_pass = set(history_fold_counts) == expected_history_folds and all(
        count >= 3 for count in history_fold_counts.values()
    )
    history_anchor_metrics: dict[str, Any] = {}
    for anchor_id in sorted(frame.loc[history_mask, "anchor_id"].astype(str).unique()):
        anchor_mask = history_mask & frame["anchor_id"].astype(str).eq(anchor_id).to_numpy()
        anchor_candidate_rmse = limb_balanced_rmse(
            frame.loc[anchor_mask],
            nested_predictions[anchor_mask] - target[anchor_mask],
        )
        anchor_null_rmses = {
            candidate_id: limb_balanced_rmse(
                frame.loc[anchor_mask],
                values[anchor_mask] - target[anchor_mask],
            )
            for candidate_id, values in null_predictions.items()
        }
        anchor_strongest_null = min(anchor_null_rmses, key=anchor_null_rmses.__getitem__)
        anchor_static_rmse = anchor_null_rmses[static_id]
        history_anchor_metrics[anchor_id] = {
            "coordinates": int(frame.loc[anchor_mask, "coordinate_id"].nunique()),
            "candidate_rmse": anchor_candidate_rmse,
            "strongest_memoryless_id": anchor_strongest_null,
            "strongest_memoryless_rmse": anchor_null_rmses[anchor_strongest_null],
            "static_rmse": anchor_static_rmse,
            "point_improvement_pass": (
                anchor_candidate_rmse <= 0.95 * anchor_null_rmses[anchor_strongest_null]
                and anchor_candidate_rmse <= 0.95 * anchor_static_rmse
            ),
        }
    lower_basin_history_point_pass = bool(history_anchor_metrics["tied_basin_lower_anchor"]["point_improvement_pass"])

    clock_rmses = {
        clock: limb_balanced_rmse(frame, prediction - target) for clock, prediction in clock_predictions.items()
    }
    clock_difference = cluster_bootstrap_rmse_difference(
        frame,
        clock_predictions["token"] - target,
        clock_predictions["lr_mass"] - target,
    )
    if clock_difference["ci_high"] < 0.0:
        identified_clock = "token"
    elif clock_difference["ci_low"] > 0.0:
        identified_clock = "lr_mass"
    else:
        identified_clock = "ambiguous"

    family_counts = Counter(str(row["family"]) for row in selections)
    clock_counts = Counter(str(row["clock"]) for row in selections if str(row["clock"]) != "none")
    response_mode_counts = Counter(str(row["response_mode"]) for row in selections)
    dynamic_selections = [row for row in selections if str(row["family"]) in DYNAMIC_FAMILIES]
    rate_interior = 0
    if dynamic_selections:
        for row in dynamic_selections:
            acquisition = float(row["acquisition_rate"])
            forgetting = float(row["forgetting_ratio"])
            acquisition_grid = design.ACQUISITION_RATE_GRID
            forgetting_grid = design.FORGETTING_RATIO_GRID
            acquisition_ok = min(acquisition_grid) < acquisition < max(acquisition_grid)
            forgetting_ok = min(forgetting_grid) < forgetting < max(forgetting_grid)
            rate_interior += int(acquisition_ok and forgetting_ok)

    structure_pass = (
        nested_rmse <= 0.95 * zero_rmse and nested_rmse <= 0.95 * strongest_null_rmse and bootstrap["ci_high"] < 0.0
    )
    anchor_structure_pass = (
        anchor_nested_rmse <= 0.95 * anchor_null_rmses[strongest_anchor_null_id] and anchor_bootstrap["ci_high"] < 0.0
    )
    history_pass = (
        history_design_pass
        and lower_basin_history_point_pass
        and history_candidate_rmse <= 0.95 * history_terminal_rmse
        and history_bootstrap["ci_high"] < 0.0
        and history_candidate_rmse <= 0.95 * history_static_rmse
        and history_static_bootstrap["ci_high"] < 0.0
        and history_candidate_rmse <= 0.95 * history_null_rmses[strongest_history_null_id]
        and strongest_history_bootstrap["ci_high"] < 0.0
    )
    dynamic_vs_static_pass = nested_rmse <= 0.95 * static_rmse and dynamic_minus_static["ci_high"] < 0.0
    return {
        "nested_rmse": nested_rmse,
        "zero_rmse": zero_rmse,
        "null_rmses": null_rmses,
        "strongest_null_id": strongest_null_id,
        "strongest_null_rmse": strongest_null_rmse,
        "candidate_minus_null_bootstrap": bootstrap,
        "leave_anchor_out_nested_rmse": anchor_nested_rmse,
        "leave_anchor_out_strongest_null_id": strongest_anchor_null_id,
        "leave_anchor_out_strongest_null_rmse": anchor_null_rmses[strongest_anchor_null_id],
        "leave_anchor_out_candidate_minus_null_bootstrap": anchor_bootstrap,
        "fixed_phase_history_candidate_rmse": history_candidate_rmse,
        "fixed_phase_history_terminal_rmse": history_terminal_rmse,
        "fixed_phase_history_candidate_minus_terminal_bootstrap": history_bootstrap,
        "fixed_phase_history_strongest_memoryless_id": strongest_history_null_id,
        "fixed_phase_history_strongest_memoryless_rmse": history_null_rmses[strongest_history_null_id],
        "fixed_phase_history_candidate_minus_strongest_memoryless_bootstrap": strongest_history_bootstrap,
        "fixed_phase_history_static_rmse": history_static_rmse,
        "fixed_phase_history_candidate_minus_static_bootstrap": history_static_bootstrap,
        "fixed_phase_history_switch_coordinate_counts": {str(key): value for key, value in history_fold_counts.items()},
        "fixed_phase_history_anchor_metrics": history_anchor_metrics,
        "tied_basin_lower_history_point_pass": lower_basin_history_point_pass,
        "fixed_phase_history_design_pass": history_design_pass,
        "static_switch_control_rmse": static_rmse,
        "candidate_minus_static_bootstrap": dynamic_minus_static,
        "dynamic_vs_static_pass": dynamic_vs_static_pass,
        "clock_rmses": clock_rmses,
        "token_minus_lr_clock_bootstrap": clock_difference,
        "identified_clock": identified_clock,
        "family_selection_counts": dict(family_counts),
        "clock_selection_counts": dict(clock_counts),
        "response_mode_selection_counts": dict(response_mode_counts),
        "interior_dynamic_rate_selections": rate_interior,
        "structure_pass": structure_pass,
        "anchor_structure_pass": anchor_structure_pass,
        "history_pass": history_pass,
        "exact_clock_identified": identified_clock != "ambiguous",
        "licensed_final_unseal": structure_pass and anchor_structure_pass and history_pass and dynamic_vs_static_pass,
    }


def freeze_transition(output_dir: Path, transition_path: Path) -> Path:
    """Select and freeze the temporal transition without reading the final step."""
    assert_endpoint_still_sealed(output_dir)
    protocol = load_protocol()
    manifest = load_manifest(protocol)
    raw = pd.read_csv(transition_path)
    verify_stage_steps(raw, transition_steps(protocol), stage="transition freeze")
    if int(raw["global_step"].max()) > TRANSITION_MAX_STEP:
        raise ValueError("Transition freeze received a sealed endpoint")
    spines = fit_tied_spine_bundle(raw, manifest, protocol)
    spine_diagnostics, spine_identification = aggregate_spine_diagnostics(raw, manifest, protocol)
    spine_diagnostics.to_csv(output_dir / "aggregate_spine_diagnostics.csv", index=False)
    write_json(output_dir / "aggregate_spine_identification.json", spine_identification)
    write_json(output_dir / "tied_spines.json", spines)
    paired = residualize_with_observed_tied_controls(raw, manifest)
    primary_target = str(protocol["targets"]["primary"])
    primary = paired.loc[paired["target"].eq(primary_target)].reset_index(drop=True)
    candidates = candidate_grid(protocol)
    feature_cache = feature_cache_for_candidates(primary, candidates, protocol, spines)
    separation = dynamic_static_separation_table(primary, candidates, feature_cache)
    separation.to_csv(output_dir / "dynamic_static_feature_separation.csv", index=False)

    selected, comparison = select_candidate(primary, candidates, feature_cache)
    null_candidates = frozen_null_candidates(protocol)
    sealed_design = sealed_feature_design_frame(manifest, protocol)
    sealed_feature_candidates = (selected, *null_candidates)
    sealed_feature_cache = feature_cache_for_candidates(
        sealed_design,
        sealed_feature_candidates,
        protocol,
        spines,
    )
    sealed_separation = dynamic_static_separation_table(
        sealed_design,
        sealed_feature_candidates,
        sealed_feature_cache,
    )
    sealed_separation.to_csv(output_dir / "sealed_dynamic_static_feature_separation.csv", index=False)
    nested_predictions, nested_selections = nested_family_predictions(primary, candidates, feature_cache)
    anchor_nested_predictions, anchor_nested_selections = nested_family_predictions(
        primary,
        candidates,
        feature_cache,
        outer_fold_column="anchor_id",
    )
    clock_predictions: dict[str, np.ndarray] = {}
    clock_selections: dict[str, list[dict[str, Any]]] = {}
    for clock in ("token", "lr_mass"):
        clock_candidates = [
            candidate for candidate in candidates if candidate.family in DYNAMIC_FAMILIES and candidate.clock == clock
        ]
        clock_predictions[clock], clock_selections[clock] = nested_family_predictions(
            primary,
            clock_candidates,
            feature_cache,
        )
    null_oof: dict[str, np.ndarray] = {}
    anchor_null_oof: dict[str, np.ndarray] = {}
    for candidate in null_candidates:
        values = oof_predictions(primary, feature_cache[candidate.candidate_id], candidate)
        if values is None:
            raise RuntimeError(f"Frozen null became singular: {candidate.candidate_id}")
        null_oof[candidate.candidate_id] = values
        anchor_values = oof_predictions(
            primary,
            feature_cache[candidate.candidate_id],
            candidate,
            fold_column="anchor_id",
        )
        if anchor_values is None:
            raise RuntimeError(f"Frozen leave-anchor-out null became singular: {candidate.candidate_id}")
        anchor_null_oof[candidate.candidate_id] = anchor_values
    gates = transition_gate_summary(
        frame=primary,
        nested_predictions=nested_predictions,
        anchor_nested_predictions=anchor_nested_predictions,
        null_predictions=null_oof,
        anchor_null_predictions=anchor_null_oof,
        selections=nested_selections,
        clock_predictions=clock_predictions,
    )
    selected_separation = separation.loc[separation["candidate_id"].eq(selected.candidate_id)]
    selected_separation_pass = False
    selected_separation_payload: dict[str, Any] | None = None
    if len(selected_separation) == 1:
        separation_row = selected_separation.iloc[0]
        selected_separation_payload = {
            "candidate_id": str(separation_row["candidate_id"]),
            "global_projection_residual": float(separation_row["global_projection_residual"]),
            "global_minimum_principal_angle_degrees": float(separation_row["global_minimum_principal_angle_degrees"]),
            "global_maximum_principal_angle_degrees": float(separation_row["global_maximum_principal_angle_degrees"]),
            "minimum_switch_projection_residual": float(separation_row["minimum_switch_projection_residual"]),
            "minimum_eligible_switch_projection_residual": float(
                separation_row["minimum_eligible_switch_projection_residual"]
            ),
            "median_switch_projection_residual": float(separation_row["median_switch_projection_residual"]),
            "maximum_switch_projection_residual": float(separation_row["maximum_switch_projection_residual"]),
            "eligible_switch_folds": int(separation_row["eligible_switch_folds"]),
            "required_minimum_switch_projection_residual": design.MIN_STATIC_SUBSPACE_RESIDUAL,
        }
        selected_separation_pass = bool(separation_row["passes_static_separation_floor"])
    gates["selected_dynamic_static_feature_separation"] = selected_separation_payload
    gates["selected_dynamic_static_feature_separation_pass"] = selected_separation_pass
    selected_sealed_separation = sealed_separation.loc[sealed_separation["candidate_id"].eq(selected.candidate_id)]
    selected_sealed_separation_pass = False
    selected_sealed_separation_payload: dict[str, Any] | None = None
    if len(selected_sealed_separation) == 1:
        sealed_row = selected_sealed_separation.iloc[0]
        selected_sealed_separation_payload = {
            "candidate_id": str(sealed_row["candidate_id"]),
            "global_projection_residual": float(sealed_row["global_projection_residual"]),
            "minimum_eligible_switch_projection_residual": float(
                sealed_row["minimum_eligible_switch_projection_residual"]
            ),
            "eligible_switch_folds": int(sealed_row["eligible_switch_folds"]),
            "separated_eligible_switch_folds": int(sealed_row["separated_eligible_switch_folds"]),
            "required_minimum_switch_projection_residual": design.MIN_STATIC_SUBSPACE_RESIDUAL,
            "required_separated_switch_folds": design.MIN_SEALED_SEPARATED_SWITCH_FOLDS,
            "outcomes_accessed": False,
        }
        selected_sealed_separation_pass = bool(
            float(sealed_row["global_projection_residual"]) >= design.MIN_STATIC_SUBSPACE_RESIDUAL
            and int(sealed_row["separated_eligible_switch_folds"]) >= design.MIN_SEALED_SEPARATED_SWITCH_FOLDS
        )
    gates["selected_sealed_dynamic_static_feature_separation"] = selected_sealed_separation_payload
    gates["selected_sealed_dynamic_static_feature_separation_pass"] = selected_sealed_separation_pass
    spine_required = selected.response_mode == "potential_constrained"
    gates["aggregate_spine_identification"] = spine_identification
    gates["aggregate_spine_required_for_selected_response"] = spine_required
    gates["aggregate_spine_identification_pass"] = bool(not spine_required or spine_identification["passed"])
    response_forms = response_form_comparison(primary, selected, candidates, feature_cache)
    gates["response_form_comparison"] = response_forms
    stability = nested_selection_stability(selected, nested_selections, anchor_nested_selections)
    gates["nested_selection_stability"] = stability
    gates["nested_selection_stability_pass"] = bool(stability["passed"])
    timescale = timescale_identification(primary, selected, protocol)
    gates["timescale_identification"] = timescale
    gates["timescale_identification_pass"] = bool(timescale["passed"])
    gates["licensed_final_unseal"] = bool(
        gates["licensed_final_unseal"]
        and selected_separation_pass
        and selected_sealed_separation_pass
        and gates["aggregate_spine_identification_pass"]
        and stability["passed"]
        and timescale["passed"]
    )
    if selected.family not in DYNAMIC_FAMILIES:
        gates["licensed_final_unseal"] = False
        gates["selected_dynamic_mechanism"] = False
    else:
        gates["selected_dynamic_mechanism"] = True

    targets = target_columns(protocol)
    blocked_fold_models = {
        "switch_step": freeze_blocked_fold_models(
            paired,
            protocol=protocol,
            primary_selections=nested_selections,
            null_candidates=null_candidates,
            spines=spines,
            fold_column="switch_step",
        ),
        "anchor_id": freeze_blocked_fold_models(
            paired,
            protocol=protocol,
            primary_selections=anchor_nested_selections,
            null_candidates=null_candidates,
            spines=spines,
            fold_column="anchor_id",
        ),
    }
    response_heads: dict[str, dict[str, Any]] = {}
    null_heads: dict[str, dict[str, dict[str, Any]]] = {}
    prediction_rows: list[pd.DataFrame] = []
    for target_name in targets:
        target_frame = paired.loc[paired["target"].eq(target_name)].reset_index(drop=True)
        target_values = target_frame["observed_delta"].to_numpy(float)
        target_feature_cache = feature_cache_for_candidates(
            target_frame,
            (selected, *null_candidates),
            protocol,
            spines,
        )
        selected_features = target_feature_cache[selected.candidate_id]
        selected_fit = fit_head(selected_features, target_values, coordinate_weights(target_frame), selected)
        if selected_fit is None:
            raise RuntimeError(f"Selected response head is singular on {target_name}")
        response_heads[target_name] = asdict(selected_fit)
        target_predictions = target_frame[
            [
                "observation_id",
                "coordinate_id",
                "anchor_id",
                "design_arm",
                "role",
                "pair_id",
                "switch_step",
                "aggregate_code_weight",
                "signed_contrast",
                "phase_0_code_weight",
                "phase_1_code_weight",
                "run_seed",
                "global_step",
                "target",
                "observed_delta",
            ]
        ].copy()
        target_predictions["full_fit_prediction"] = predict(selected_features, selected_fit)
        blocked_prediction, blocked_candidate_ids, blocked_null_predictions = predict_blocked_folds(
            target_frame,
            protocol=protocol,
            frozen_models=blocked_fold_models["switch_step"][target_name],
            null_candidates=null_candidates,
            spines=spines,
            fold_column="switch_step",
        )
        anchor_prediction, anchor_candidate_ids, anchor_null_predictions = predict_blocked_folds(
            target_frame,
            protocol=protocol,
            frozen_models=blocked_fold_models["anchor_id"][target_name],
            null_candidates=null_candidates,
            spines=spines,
            fold_column="anchor_id",
        )
        target_predictions["blocked_prediction"] = blocked_prediction
        target_predictions["blocked_candidate_id"] = blocked_candidate_ids
        target_predictions["anchor_blocked_prediction"] = anchor_prediction
        target_predictions["anchor_blocked_candidate_id"] = anchor_candidate_ids
        null_heads[target_name] = {}
        for null_candidate in null_candidates:
            null_features = target_feature_cache[null_candidate.candidate_id]
            null_fit = fit_head(
                null_features,
                target_values,
                coordinate_weights(target_frame),
                null_candidate,
            )
            if null_fit is None:
                raise RuntimeError(f"Null response head is singular: {target_name} {null_candidate.candidate_id}")
            null_heads[target_name][null_candidate.candidate_id] = asdict(null_fit)
            target_predictions[f"null_full__{null_candidate.candidate_id}"] = predict(null_features, null_fit)
            target_predictions[f"null_blocked__{null_candidate.candidate_id}"] = blocked_null_predictions[
                null_candidate.candidate_id
            ]
            target_predictions[f"null_anchor_blocked__{null_candidate.candidate_id}"] = anchor_null_predictions[
                null_candidate.candidate_id
            ]
        prediction_rows.append(target_predictions)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions_path = output_dir / "transition_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    comparison.to_csv(output_dir / "candidate_comparison.csv", index=False)
    pd.DataFrame.from_records(nested_selections).to_csv(output_dir / "nested_selections.csv", index=False)
    pd.DataFrame.from_records(anchor_nested_selections).to_csv(
        output_dir / "leave_anchor_out_nested_selections.csv",
        index=False,
    )

    frozen_payload = {
        "candidate_id": str(protocol["candidate_id"]),
        "protocol_sha256": str(protocol["protocol_sha256"]),
        "evaluator_sha256": sha256_path(Path(__file__)),
        "transition_observations_sha256": sha256_path(transition_path),
        "transition_predictions_sha256": sha256_path(predictions_path),
        "selected_candidate": {**asdict(selected), "candidate_id": selected.candidate_id},
        "selection_target": primary_target,
        "response_heads": response_heads,
        "null_heads": null_heads,
        "blocked_fold_models": blocked_fold_models,
        "tied_spines": spines,
        "aggregate_spine_identification": spine_identification,
        "dynamic_static_feature_separation": selected_separation_payload,
        "sealed_dynamic_static_feature_separation": selected_sealed_separation_payload,
        "clock_nested_selections": clock_selections,
        "transition_gates": gates,
        "final_step_accessed": False,
        "transfer_policy": str(protocol["targets"]["selection_policy"]),
    }
    frozen_path = output_dir / "frozen_transition.json"
    write_json(frozen_path, frozen_payload)
    write_transition_report(output_dir / "transition_report.md", frozen_payload, comparison)
    write_json(
        output_dir / "transition_freeze_record.json",
        {
            "frozen_transition_sha256": sha256_path(frozen_path),
            "protocol_sha256": protocol["protocol_sha256"],
            "analysis_source_sha256": sha256_path(Path(__file__)),
            "transition_data_sha256": sha256_path(transition_path),
            "transition_predictions_sha256": sha256_path(predictions_path),
            "licensed_final_unseal": bool(gates["licensed_final_unseal"]),
            "outcomes_accessed": f"post-switch transition evaluations through step {TRANSITION_MAX_STEP} only",
        },
    )
    return frozen_path


def write_transition_report(path: Path, frozen: dict[str, Any], comparison: pd.DataFrame) -> None:
    top = comparison.head(12).to_markdown(index=False)
    gates = frozen["transition_gates"]
    selected = frozen["selected_candidate"]
    report = f"""# Switch-Time Transition Selection

- Protocol: `{frozen['protocol_sha256']}`
- Evaluator: `{frozen['evaluator_sha256']}`
- Selected mechanism: `{selected['candidate_id']}`
- Primary nested RMSE: `{gates['nested_rmse']:.8f}`
- Strongest null RMSE: `{gates['strongest_null_rmse']:.8f}`
- Dynamic/static feature separation passes: `{gates['selected_dynamic_static_feature_separation_pass']}`
- Tied aggregate spine identification passes: `{gates['aggregate_spine_identification_pass']}`
- Final unseal licensed: `{gates['licensed_final_unseal']}`
- Final endpoint accessed: `False`

## Frozen Gates

```json
{json.dumps(gates, indent=2, sort_keys=True)}
```

## Leading Primary-Target Candidates

{top}

Mechanism, clock, and nonlinear rates were selected only on the primary target.
Transfer targets reused the selected state and fitted only target-specific
zero-intercept response amplitudes on transition evaluations through step
{TRANSITION_MAX_STEP}. Full-data heads are descriptive; endpoint gates use the
target-specific head and mechanism frozen while that endpoint's switch fold was
excluded.
"""
    path.write_text(report)


def verify_frozen_transition(output_dir: Path, *, require_license: bool) -> dict[str, Any]:
    """Verify every frozen input before permitting final endpoint access."""
    protocol = load_protocol()
    frozen_path = output_dir / "frozen_transition.json"
    record_path = output_dir / "transition_freeze_record.json"
    transition_path = output_dir / "transition_observations.csv"
    predictions_path = output_dir / "transition_predictions.csv"
    if not all(path.exists() for path in (frozen_path, record_path, transition_path, predictions_path)):
        raise FileNotFoundError("Frozen transition bundle is incomplete")
    frozen = json.loads(frozen_path.read_text())
    record = json.loads(record_path.read_text())
    checks = {
        "protocol_sha256": str(protocol["protocol_sha256"]),
        "evaluator_sha256": sha256_path(Path(__file__)),
        "transition_observations_sha256": sha256_path(transition_path),
        "transition_predictions_sha256": sha256_path(predictions_path),
    }
    for key, observed in checks.items():
        if str(frozen[key]) != observed:
            raise ValueError(f"Frozen transition drift at {key}: {observed} != {frozen[key]}")
    if str(record["frozen_transition_sha256"]) != sha256_path(frozen_path):
        raise ValueError("Frozen transition record hash drifted")
    if require_license and not bool(frozen["transition_gates"]["licensed_final_unseal"]):
        raise PermissionError("Transition gates did not license final endpoint access")
    if bool(frozen["final_step_accessed"]):
        raise ValueError("Frozen transition incorrectly claims prior endpoint access")
    return frozen


def head_from_dict(value: dict[str, Any]) -> HeadFit:
    return HeadFit(tuple(float(item) for item in value["coefficients"]), float(value["condition_number"]))


def final_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    observed = frame["observed_delta"].to_numpy(float)
    predicted = frame[prediction_column].to_numpy(float)
    residual = predicted - observed
    weights = coordinate_weights(frame)
    coordinate_means = (
        pd.DataFrame(
            {
                "coordinate_id": frame["coordinate_id"].astype(str),
                "observed": observed,
                "predicted": predicted,
                "residual": residual,
            }
        )
        .groupby("coordinate_id")
        .mean()
    )
    slope = float("nan")
    if float(np.var(coordinate_means["predicted"])) > 0.0:
        slope = float(
            np.cov(coordinate_means["predicted"], coordinate_means["observed"], ddof=0)[0, 1]
            / np.var(coordinate_means["predicted"])
        )
    return {
        "rmse": float(np.sqrt(np.sum(weights * residual**2))),
        "bias": float(np.sum(weights * residual)),
        "coordinate_rmse": coordinate_rmse(frame, residual),
        "coordinate_bias": float(coordinate_means["residual"].mean()),
        "observed_on_predicted_slope": slope,
        "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "worst_optimism": float(np.min(residual)),
        "optimism_over_0p005": int(np.sum(coordinate_means["residual"] < -0.005)),
    }


def evaluate_final(output_dir: Path, final_path: Path) -> Path:
    """Evaluate the sealed endpoint using only the already frozen transition."""
    frozen = verify_frozen_transition(output_dir, require_license=True)
    if not endpoint_unseal_path(output_dir).exists():
        raise PermissionError("Final evaluation requires the irreversible endpoint-unseal marker")
    protocol = load_protocol()
    manifest = load_manifest(protocol)
    raw = pd.read_csv(final_path)
    verify_stage_steps(raw, sealed_decay_steps(protocol), stage="final evaluation")
    selected = candidate_from_dict(frozen["selected_candidate"])
    spines = frozen["tied_spines"]
    paired = residualize_with_observed_tied_controls(raw, manifest)
    null_candidates = frozen_null_candidates(protocol)
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for target_name in target_columns(protocol):
        target_frame = paired.loc[paired["target"].eq(target_name)].reset_index(drop=True)
        target_feature_cache = feature_cache_for_candidates(
            target_frame,
            (selected, *null_candidates),
            protocol,
            spines,
        )
        selected_features = target_feature_cache[selected.candidate_id]
        selected_head = head_from_dict(frozen["response_heads"][target_name])
        blocked_prediction, blocked_candidate_ids, blocked_null_predictions = predict_blocked_folds(
            target_frame,
            protocol=protocol,
            frozen_models=frozen["blocked_fold_models"]["switch_step"][target_name],
            null_candidates=null_candidates,
            spines=spines,
            fold_column="switch_step",
        )
        anchor_prediction, anchor_candidate_ids, anchor_null_predictions = predict_blocked_folds(
            target_frame,
            protocol=protocol,
            frozen_models=frozen["blocked_fold_models"]["anchor_id"][target_name],
            null_candidates=null_candidates,
            spines=spines,
            fold_column="anchor_id",
        )
        predictions = target_frame[
            [
                "observation_id",
                "coordinate_id",
                "anchor_id",
                "design_arm",
                "role",
                "pair_id",
                "switch_step",
                "aggregate_code_weight",
                "signed_contrast",
                "phase_0_code_weight",
                "phase_1_code_weight",
                "run_seed",
                "global_step",
                "target",
                "observed_delta",
            ]
        ].copy()
        predictions["full_fit_prediction"] = predict(selected_features, selected_head)
        predictions["blocked_prediction"] = blocked_prediction
        predictions["blocked_candidate_id"] = blocked_candidate_ids
        predictions["anchor_blocked_prediction"] = anchor_prediction
        predictions["anchor_blocked_candidate_id"] = anchor_candidate_ids
        post_switch = predictions.loc[
            predictions["global_step"].astype(int).ge(predictions["switch_step"].astype(int))
        ].copy()
        metric_rows.append(
            {
                "target": target_name,
                "model": "selected_blocked",
                "scope": "sealed_post_switch",
                **final_metrics(post_switch, "blocked_prediction"),
            }
        )
        metric_rows.append(
            {
                "target": target_name,
                "model": "selected_full_fit",
                "scope": "sealed_post_switch",
                **final_metrics(post_switch, "full_fit_prediction"),
            }
        )
        metric_rows.append(
            {
                "target": target_name,
                "model": "selected_anchor_blocked",
                "scope": "sealed_post_switch",
                **final_metrics(post_switch, "anchor_blocked_prediction"),
            }
        )
        for null_candidate in null_candidates:
            null_features = target_feature_cache[null_candidate.candidate_id]
            null_head = head_from_dict(frozen["null_heads"][target_name][null_candidate.candidate_id])
            full_column = f"null_full__{null_candidate.candidate_id}"
            blocked_column = f"null_blocked__{null_candidate.candidate_id}"
            anchor_blocked_column = f"null_anchor_blocked__{null_candidate.candidate_id}"
            predictions[full_column] = predict(null_features, null_head)
            predictions[blocked_column] = blocked_null_predictions[null_candidate.candidate_id]
            predictions[anchor_blocked_column] = anchor_null_predictions[null_candidate.candidate_id]
            post_switch = predictions.loc[
                predictions["global_step"].astype(int).ge(predictions["switch_step"].astype(int))
            ].copy()
            metric_rows.append(
                {
                    "target": target_name,
                    "model": blocked_column,
                    "scope": "sealed_post_switch",
                    **final_metrics(post_switch, blocked_column),
                }
            )
            metric_rows.append(
                {
                    "target": target_name,
                    "model": full_column,
                    "scope": "sealed_post_switch",
                    **final_metrics(post_switch, full_column),
                }
            )
            metric_rows.append(
                {
                    "target": target_name,
                    "model": anchor_blocked_column,
                    "scope": "sealed_post_switch",
                    **final_metrics(post_switch, anchor_blocked_column),
                }
            )
        metric_rows.append(
            {
                "target": target_name,
                "model": "zero",
                "scope": "sealed_post_switch",
                **final_metrics(post_switch.assign(zero_prediction=0.0), "zero_prediction"),
            }
        )
        prediction_rows.append(predictions)

    final_predictions = pd.concat(prediction_rows, ignore_index=True)
    final_metrics_frame = pd.DataFrame.from_records(metric_rows)
    predictions_path = output_dir / "final_predictions.csv"
    metrics_path = output_dir / "final_metrics.csv"
    final_predictions.to_csv(predictions_path, index=False)
    final_metrics_frame.to_csv(metrics_path, index=False)
    gates = final_gate_summary(final_predictions, final_metrics_frame, protocol)
    write_final_report(output_dir / "final_report.md", frozen, gates, final_metrics_frame)
    write_json(
        output_dir / "final_evaluation.json",
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "frozen_transition_sha256": sha256_path(output_dir / "frozen_transition.json"),
            "final_observations_sha256": sha256_path(final_path),
            "final_predictions_sha256": sha256_path(predictions_path),
            "final_metrics_sha256": sha256_path(metrics_path),
            "gates": gates,
            "refitting_after_unseal": False,
        },
    )
    return output_dir / "final_evaluation.json"


def phase_gain_diagnostics(primary_predictions: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    """Summarize preregistered repeated-coordinate endpoint effects by anchor."""
    intervention_anchors = {
        str(anchor["anchor_id"]) for anchor in protocol["anchors"] if bool(anchor["intervention_anchor"])
    }
    endpoint = primary_predictions.loc[
        primary_predictions["global_step"].eq(FINAL_STEP)
        & primary_predictions["design_arm"].eq("fixed_aggregate_contrast")
        & primary_predictions["anchor_id"].isin(intervention_anchors)
    ].copy()
    coordinate_summary = endpoint.groupby(["anchor_id", "coordinate_id"], as_index=False).agg(
        switch_step=("switch_step", "first"),
        signed_contrast=("signed_contrast", "first"),
        observed_mean=("observed_delta", "mean"),
        observed_sd=("observed_delta", "std"),
        blocked_prediction_mean=("blocked_prediction", "mean"),
        anchor_blocked_prediction_mean=("anchor_blocked_prediction", "mean"),
        seeds=("run_seed", "nunique"),
    )
    repeated = coordinate_summary.loc[coordinate_summary["seeds"].eq(len(design.ASYMMETRIC_SEED_VALUES))].copy()
    expected_repeated = len(intervention_anchors) * 2 * len(design.REPLICATED_MAIN_SWITCH_STEPS)
    if len(repeated) != expected_repeated:
        raise ValueError(f"Expected {expected_repeated} repeated phase coordinates, found {len(repeated)}")
    repeated_ids = set(repeated["coordinate_id"].astype(str))
    repeated_rows = endpoint.loc[endpoint["coordinate_id"].astype(str).isin(repeated_ids)].copy()
    seed_summary = repeated_rows.groupby(["anchor_id", "run_seed"], as_index=False).agg(
        observed_mean=("observed_delta", "mean"),
        blocked_prediction_mean=("blocked_prediction", "mean"),
        anchor_blocked_prediction_mean=("anchor_blocked_prediction", "mean"),
    )
    simultaneous_t = float(
        student_t.ppf(1.0 - design.ALPHA / (2.0 * len(intervention_anchors)), df=len(design.ASYMMETRIC_SEED_VALUES) - 1)
    )

    anchors: dict[str, Any] = {}
    for anchor_id in sorted(intervention_anchors):
        all_anchor = coordinate_summary.loc[coordinate_summary["anchor_id"].eq(anchor_id)]
        repeated_anchor = repeated.loc[repeated["anchor_id"].eq(anchor_id)]
        seed_anchor = seed_summary.loc[seed_summary["anchor_id"].eq(anchor_id)]
        if len(seed_anchor) != len(design.ASYMMETRIC_SEED_VALUES):
            raise ValueError(f"Anchor {anchor_id} lacks the complete repeated-coordinate seed block")
        observed_mean = float(seed_anchor["observed_mean"].mean())
        observed_sd = float(seed_anchor["observed_mean"].std(ddof=1))
        half_width = simultaneous_t * observed_sd / math.sqrt(len(seed_anchor))
        anchors[anchor_id] = {
            "coordinates": len(all_anchor),
            "repeated_coordinates": len(repeated_anchor),
            "repeated_set_seed_averaged_observed_delta": observed_mean,
            "repeated_set_simultaneous_ci_low": observed_mean - half_width,
            "repeated_set_simultaneous_ci_high": observed_mean + half_width,
            "repeated_set_switch_blocked_predicted_delta": float(seed_anchor["blocked_prediction_mean"].mean()),
            "repeated_set_anchor_blocked_predicted_delta": float(seed_anchor["anchor_blocked_prediction_mean"].mean()),
            "selection_biased_best_observed_gain": max(0.0, -float(all_anchor["observed_mean"].min())),
            "selection_biased_best_switch_blocked_predicted_gain": max(
                0.0,
                -float(all_anchor["blocked_prediction_mean"].min()),
            ),
            "selection_biased_best_anchor_blocked_predicted_gain": max(
                0.0,
                -float(all_anchor["anchor_blocked_prediction_mean"].min()),
            ),
        }
    return {
        "endpoint_step": FINAL_STEP,
        "simultaneous_two_sided_t_for_anchor_means": simultaneous_t,
        "estimand": "mean endpoint delta over the frozen repeated-coordinate set, averaged within seed",
        "anchors": anchors,
        "coordinatewise_best_values_are_selection_biased": True,
        "diagnostic_only": True,
    }


def final_gate_summary(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    primary_target = str(protocol["targets"]["primary"])
    primary_all_predictions = predictions.loc[predictions["target"].eq(primary_target)].reset_index(drop=True)
    primary_predictions = primary_all_predictions.loc[
        primary_all_predictions["global_step"].astype(int).ge(primary_all_predictions["switch_step"].astype(int))
    ].reset_index(drop=True)
    primary_metrics = metrics.loc[metrics["target"].eq(primary_target)].set_index("model")
    selected_rmse = float(primary_metrics.loc["selected_blocked", "rmse"])
    anchor_selected_rmse = float(primary_metrics.loc["selected_anchor_blocked", "rmse"])
    reference_models = [
        model for model in primary_metrics.index if model == "zero" or model.startswith("null_blocked__")
    ]
    anchor_reference_models = [
        model for model in primary_metrics.index if model == "zero" or model.startswith("null_anchor_blocked__")
    ]
    reference_rmses = {model: float(primary_metrics.loc[model, "rmse"]) for model in reference_models}
    anchor_reference_rmses = {model: float(primary_metrics.loc[model, "rmse"]) for model in anchor_reference_models}
    strongest_reference = min(reference_rmses, key=reference_rmses.__getitem__)
    strongest_anchor_reference = min(anchor_reference_rmses, key=anchor_reference_rmses.__getitem__)
    bootstrap = cluster_bootstrap_rmse_difference(
        primary_predictions,
        primary_predictions["blocked_prediction"].to_numpy(float)
        - primary_predictions["observed_delta"].to_numpy(float),
        (
            np.zeros(len(primary_predictions), dtype=float)
            if strongest_reference == "zero"
            else primary_predictions[strongest_reference].to_numpy(float)
        )
        - primary_predictions["observed_delta"].to_numpy(float),
    )
    anchor_bootstrap = cluster_bootstrap_rmse_difference(
        primary_predictions,
        primary_predictions["anchor_blocked_prediction"].to_numpy(float)
        - primary_predictions["observed_delta"].to_numpy(float),
        (
            np.zeros(len(primary_predictions), dtype=float)
            if strongest_anchor_reference == "zero"
            else primary_predictions[strongest_anchor_reference].to_numpy(float)
        )
        - primary_predictions["observed_delta"].to_numpy(float),
    )
    sealed_transfer_pass = (
        selected_rmse <= 0.90 * min(reference_rmses.values())
        and bootstrap["ci_high"] < 0.0
        and anchor_selected_rmse <= 0.90 * min(anchor_reference_rmses.values())
        and anchor_bootstrap["ci_high"] < 0.0
    )

    positive = primary_all_predictions.loc[
        primary_all_predictions["anchor_id"].eq("tied_basin_lower_anchor")
        & primary_all_predictions["design_arm"].eq("fixed_aggregate_contrast")
        & primary_all_predictions["switch_step"].eq(int(protocol["schedule"]["boundary_step"]))
        & np.isclose(primary_all_predictions["signed_contrast"].astype(float), 0.20)
        & primary_all_predictions["global_step"].eq(FINAL_STEP)
    ]
    if len(positive) != len(design.ASYMMETRIC_SEED_VALUES):
        raise ValueError(f"Expected three fresh 2B positive-control seeds, found {len(positive)}")
    positive_mean = float(positive["observed_delta"].mean())
    positive_descriptive_pass = positive_mean <= -design.POSITIVE_CONTROL_MIN_GAIN_BPB
    phase_gains = phase_gain_diagnostics(primary_all_predictions, protocol)

    code_transfer: dict[str, Any] = {}
    code_transfer_pass = True
    for target_name in protocol["targets"]["code_transfer"]:
        target_metrics = metrics.loc[metrics["target"].eq(target_name)].set_index("model")
        target_selected = float(target_metrics.loc["selected_blocked", "rmse"])
        target_anchor_selected = float(target_metrics.loc["selected_anchor_blocked", "rmse"])
        target_references = target_metrics.loc[
            [model for model in target_metrics.index if model == "zero" or model.startswith("null_blocked__")]
        ]
        target_anchor_references = target_metrics.loc[
            [model for model in target_metrics.index if model == "zero" or model.startswith("null_anchor_blocked__")]
        ]
        strongest_target_reference = str(target_references["rmse"].idxmin())
        strongest_target_anchor_reference = str(target_anchor_references["rmse"].idxmin())
        target_reference = float(target_references.loc[strongest_target_reference, "rmse"])
        target_anchor_reference = float(target_anchor_references.loc[strongest_target_anchor_reference, "rmse"])
        target_all_predictions = predictions.loc[predictions["target"].eq(target_name)].reset_index(drop=True)
        target_predictions = target_all_predictions.loc[
            target_all_predictions["global_step"].astype(int).ge(target_all_predictions["switch_step"].astype(int))
        ].reset_index(drop=True)
        target_bootstrap = cluster_bootstrap_rmse_difference(
            target_predictions,
            target_predictions["blocked_prediction"].to_numpy(float)
            - target_predictions["observed_delta"].to_numpy(float),
            (
                np.zeros(len(target_predictions), dtype=float)
                if strongest_target_reference == "zero"
                else target_predictions[strongest_target_reference].to_numpy(float)
            )
            - target_predictions["observed_delta"].to_numpy(float),
        )
        target_anchor_bootstrap = cluster_bootstrap_rmse_difference(
            target_predictions,
            target_predictions["anchor_blocked_prediction"].to_numpy(float)
            - target_predictions["observed_delta"].to_numpy(float),
            (
                np.zeros(len(target_predictions), dtype=float)
                if strongest_target_anchor_reference == "zero"
                else target_predictions[strongest_target_anchor_reference].to_numpy(float)
            )
            - target_predictions["observed_delta"].to_numpy(float),
        )
        passed = (
            target_selected <= 1.05 * target_reference
            and target_bootstrap["ci_high"] <= 0.05 * target_reference
            and target_anchor_selected <= 1.05 * target_anchor_reference
            and target_anchor_bootstrap["ci_high"] <= 0.05 * target_anchor_reference
        )
        code_transfer[str(target_name)] = {
            "selected_rmse": target_selected,
            "anchor_selected_rmse": target_anchor_selected,
            "strongest_reference": strongest_target_reference,
            "strongest_reference_rmse": target_reference,
            "candidate_minus_reference_bootstrap": target_bootstrap,
            "strongest_anchor_reference": strongest_target_anchor_reference,
            "strongest_anchor_reference_rmse": target_anchor_reference,
            "anchor_candidate_minus_reference_bootstrap": target_anchor_bootstrap,
            "passed": passed,
        }
        code_transfer_pass &= passed

    broad_controls: dict[str, Any] = {}
    broad_pass = True
    for target_name in protocol["targets"]["broad_text_negative_controls"]:
        target_all_predictions = predictions.loc[predictions["target"].eq(target_name)]
        target_predictions = target_all_predictions.loc[
            target_all_predictions["global_step"].astype(int).ge(target_all_predictions["switch_step"].astype(int))
        ]
        maximum_observed_apparent_gain = max(
            0.0,
            -float(target_predictions.groupby("coordinate_id")["observed_delta"].mean().min()),
        )
        maximum_false_gain = max(
            0.0,
            -float(target_predictions.groupby("coordinate_id")["blocked_prediction"].mean().min()),
        )
        maximum_anchor_false_gain = max(
            0.0,
            -float(target_predictions.groupby("coordinate_id")["anchor_blocked_prediction"].mean().min()),
        )
        passed = max(maximum_false_gain, maximum_anchor_false_gain) <= design.EQUIVALENCE_BPB
        broad_controls[str(target_name)] = {
            "maximum_observed_apparent_gain": maximum_observed_apparent_gain,
            "maximum_switch_blocked_false_gain": maximum_false_gain,
            "maximum_anchor_blocked_false_gain": maximum_anchor_false_gain,
            "observed_gain_is_diagnostic_only": True,
            "passed": passed,
        }
        broad_pass &= passed

    return {
        "sealed_primary_transfer_pass": sealed_transfer_pass,
        "selected_primary_rmse": selected_rmse,
        "selected_primary_anchor_blocked_rmse": anchor_selected_rmse,
        "strongest_primary_reference": strongest_reference,
        "strongest_primary_reference_rmse": reference_rmses[strongest_reference],
        "candidate_minus_reference_bootstrap": bootstrap,
        "strongest_primary_anchor_reference": strongest_anchor_reference,
        "strongest_primary_anchor_reference_rmse": anchor_reference_rmses[strongest_anchor_reference],
        "anchor_candidate_minus_reference_bootstrap": anchor_bootstrap,
        "positive_control_mean_delta": positive_mean,
        "positive_control_descriptive_pass": positive_descriptive_pass,
        "positive_control_is_license_gate": False,
        "phase_gain_diagnostics": phase_gains,
        "code_transfer": code_transfer,
        "code_transfer_pass": code_transfer_pass,
        "broad_text_response_bounds": broad_controls,
        "broad_text_response_bounds_pass": broad_pass,
        "broad_text_response_bounds_are_diagnostic_only": True,
        "all_final_gates_pass": sealed_transfer_pass and code_transfer_pass,
    }


def write_final_report(
    path: Path,
    frozen: dict[str, Any],
    gates: dict[str, Any],
    metrics: pd.DataFrame,
) -> None:
    report = f"""# Switch-Time Sealed Endpoint Evaluation

- Frozen mechanism: `{frozen['selected_candidate']['candidate_id']}`
- Refitting after endpoint unseal: `False`
- Gate predictions: `leave-switch-out and leave-anchor-out models and target heads frozen before endpoint access`
- Full-data predictions: `descriptive only`
- Endpoint scope: `full sealed post-step-{TRANSITION_MAX_STEP} cosine-decay trajectory through step {FINAL_STEP}`
- Response conditioning at endpoint: `frozen response mode; only the explicit potential-null comparator reuses the
  step-{TRANSITION_MAX_STEP} tied-spine derivative`
- Observed sealed contrast: `asymmetric minus same-anchor, same-seed tied BPB at each step;
  tied outcomes cannot refit any model component`
- All final gates pass: `{gates['all_final_gates_pass']}`

## Gates

```json
{json.dumps(gates, indent=2, sort_keys=True)}
```

## Endpoint Metrics

{metrics.to_markdown(index=False)}
"""
    path.write_text(report)


def synthetic_raw(
    *,
    protocol: dict[str, Any],
    manifest: pd.DataFrame,
    steps: Sequence[int],
    true_candidate: Candidate,
    targets_override: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate trajectories from a known aggregate-conditioned bounded transition."""
    all_targets = target_columns(protocol)
    targets = tuple(targets_override) if targets_override is not None else all_targets
    if true_candidate.response_mode == "potential_constrained":
        primary_coefficients = (1.0, 0.20)
    elif true_candidate.response_mode == "aggregate_linear_signed":
        primary_coefficients = (-1.0, 1.50, 0.20)
    elif true_candidate.response_mode == "unconditioned_signed":
        primary_coefficients = (-1.0, 0.20)
    else:
        raise ValueError(f"Unknown synthetic response mode: {true_candidate.response_mode}")

    def scaled_coefficients(scale: float) -> tuple[float, ...]:
        return tuple(scale * value for value in primary_coefficients)

    all_coefficients = {
        all_targets[0]: primary_coefficients,
        all_targets[1]: scaled_coefficients(0.7),
        all_targets[2]: scaled_coefficients(0.6),
        all_targets[3]: scaled_coefficients(0.0),
        all_targets[4]: scaled_coefficients(0.0),
    }
    coefficients = {target: all_coefficients[target] for target in targets}
    curvature = 0.40
    total_steps = int(protocol["schedule"]["total_steps"])
    spines: dict[str, dict[str, dict[str, float]]] = {}
    for target_index, target_name in enumerate(targets):
        for step in steps:
            progress = (step + 1) / total_steps
            fit = SpineFit(
                intercept=1.15 - 0.12 * progress + 0.01 * target_index,
                slope=0.0,
                curvature=curvature,
                center=SPINE_CENTER,
                condition_number=1.0,
            )
            spines.setdefault(target_name, {})[str(step)] = asdict(fit)
    frame = manifest.merge(pd.DataFrame({"global_step": tuple(steps)}), how="cross")
    frame = frame.merge(pd.DataFrame({"target": targets}), how="cross")
    feature = feature_cache_for_candidates(frame, (true_candidate,), protocol, spines)[true_candidate.candidate_id]
    if feature.shape[1] != len(primary_coefficients):
        raise ValueError("Synthetic transition truth feature dimension does not match its frozen response coefficients")
    target_indices = frame["target"].map({target: index for index, target in enumerate(targets)}).to_numpy(int)
    progress = (frame["global_step"].to_numpy(float) + 1.0) / total_steps
    aggregate = frame["aggregate_code_weight"].to_numpy(float)
    baseline = 1.15 - 0.12 * progress + 0.01 * target_indices + curvature * (aggregate - SPINE_CENTER) ** 2
    coefficient_matrix = np.asarray([coefficients[str(target)] for target in frame["target"]], dtype=float)
    value = baseline + np.sum(feature * coefficient_matrix, axis=1)
    return frame.assign(value=value)[["observation_id", "global_step", "target", "value"]]


def add_synthetic_trajectory_noise(
    base: pd.DataFrame,
    *,
    protocol: dict[str, Any],
    seed: int,
    run_intercept_variance_fraction: float,
    step_ar1_correlation: float,
) -> pd.DataFrame:
    """Add marginally calibrated run-level and step-varying trajectory noise."""
    if not 0.0 <= run_intercept_variance_fraction <= 1.0:
        raise ValueError("Run-intercept variance fraction must lie in [0,1]")
    if not 0.0 <= step_ar1_correlation < 1.0:
        raise ValueError("Synthetic AR(1) correlation must lie in [0,1)")
    noisy = base.copy()
    rng = np.random.default_rng(seed)
    paired_sd = float(protocol["power"]["historical_same_configuration_paired_sd_bpb"])
    run_sd = paired_sd / math.sqrt(2.0)
    intercept_scale = math.sqrt(run_intercept_variance_fraction)
    trajectory_scale = math.sqrt(1.0 - run_intercept_variance_fraction)
    innovations_scale = math.sqrt(1.0 - step_ar1_correlation**2)
    noise = np.zeros(len(noisy), dtype=float)
    for _, group in noisy.groupby(["observation_id", "target"], sort=True):
        ordered = group.sort_values("global_step")
        run_intercept = float(rng.normal())
        state = float(rng.normal())
        values = np.empty(len(ordered), dtype=float)
        for index in range(len(ordered)):
            if index > 0:
                state = step_ar1_correlation * state + innovations_scale * float(rng.normal())
            values[index] = run_sd * (intercept_scale * run_intercept + trajectory_scale * state)
        noise[ordered.index.to_numpy(int)] = values
    noisy["value"] = noisy["value"].to_numpy(float) + noise
    return noisy


def synthetic_selection_summary(
    raw: pd.DataFrame,
    *,
    protocol: dict[str, Any],
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Run the primary-target leave-switch-out selector for one noisy synthetic panel."""
    spines = fit_tied_spine_bundle(raw, manifest, protocol)
    _, spine_identification = aggregate_spine_diagnostics(raw, manifest, protocol)
    paired = residualize_with_observed_tied_controls(raw, manifest)
    primary = paired.loc[paired["target"].eq(str(protocol["targets"]["primary"]))].reset_index(drop=True)
    candidates = candidate_grid(protocol)
    features = feature_cache_for_candidates(primary, candidates, protocol, spines)
    selected, comparison = select_candidate(primary, candidates, features)
    separation = dynamic_static_separation_table(primary, candidates, features)
    selected_separation = separation.loc[separation["candidate_id"].eq(selected.candidate_id)]
    selected_identifiable_dynamic = bool(
        selected.family in DYNAMIC_FAMILIES
        and len(selected_separation) == 1
        and selected_separation.iloc[0]["passes_static_separation_floor"]
        and (selected.response_mode != "potential_constrained" or spine_identification["passed"])
    )
    best_dynamic = comparison.loc[comparison["family"].isin(DYNAMIC_FAMILIES)].iloc[0]
    static = comparison.loc[comparison["family"].eq("static_switch_control_null")].iloc[0]
    return {
        "selected_candidate": selected.candidate_id,
        "selected_family": selected.family,
        "selected_clock": selected.clock,
        "selected_acquisition_rate": selected.acquisition_rate,
        "selected_forgetting_ratio": selected.forgetting_ratio,
        "selected_identifiable_dynamic": selected_identifiable_dynamic,
        "aggregate_spine_identification_pass": bool(spine_identification["passed"]),
        "best_dynamic_rmse": float(best_dynamic["rmse"]),
        "static_rmse": float(static["rmse"]),
        "dynamic_minus_static_rmse": float(best_dynamic["rmse"] - static["rmse"]),
    }


def synthetic_power_audit(
    *,
    protocol: dict[str, Any],
    manifest: pd.DataFrame,
    replicates: int = 20,
) -> dict[str, Any]:
    """Estimate mechanism-selection power and false promotion at measured repeat noise."""
    truths = {
        "on_grid_dynamic": Candidate(
            "token_clock_acquisition_forgetting",
            clock="token",
            acquisition_rate=0.08,
            forgetting_ratio=1.0,
            response_mode="unconditioned_signed",
        ),
        "off_grid_dynamic": Candidate(
            "token_clock_acquisition_forgetting",
            clock="token",
            acquisition_rate=0.11,
            forgetting_ratio=0.75,
            response_mode="unconditioned_signed",
        ),
        "static_switch_null": Candidate("static_switch_control_null", response_mode="aggregate_linear_signed"),
    }
    rows: list[dict[str, Any]] = []
    steps = transition_steps(protocol)
    paired_sd = float(protocol["power"]["historical_same_configuration_paired_sd_bpb"])
    for truth_index, (truth_name, truth_candidate) in enumerate(truths.items()):
        base = synthetic_raw(
            protocol=protocol,
            manifest=manifest,
            steps=steps,
            true_candidate=truth_candidate,
            targets_override=(str(protocol["targets"]["primary"]),),
        )
        for regime_index, (regime_name, regime) in enumerate(design.SYNTHETIC_NOISE_REGIMES.items()):
            intercept_fraction, ar1_correlation = regime
            for replicate in range(replicates):
                raw = add_synthetic_trajectory_noise(
                    base,
                    protocol=protocol,
                    seed=20260801 + 100_000 * truth_index + 10_000 * regime_index + replicate,
                    run_intercept_variance_fraction=intercept_fraction,
                    step_ar1_correlation=ar1_correlation,
                )
                rows.append(
                    {
                        "truth": truth_name,
                        "noise_regime": regime_name,
                        "run_intercept_variance_fraction": intercept_fraction,
                        "step_ar1_correlation": ar1_correlation,
                        "replicate": replicate,
                        **synthetic_selection_summary(raw, protocol=protocol, manifest=manifest),
                    }
                )
    frame = pd.DataFrame.from_records(rows)
    summaries: dict[str, Any] = {}
    for (truth_name, regime_name), group in frame.groupby(["truth", "noise_regime"], sort=True):
        selected_dynamic = group["selected_identifiable_dynamic"].astype(bool)
        summaries[f"{truth_name}::{regime_name}"] = {
            "truth": str(truth_name),
            "noise_regime": str(regime_name),
            "replicates": len(group),
            "dynamic_selection_rate": float(selected_dynamic.mean()),
            "token_clock_selection_rate": float(group["selected_clock"].eq("token").mean()),
            "dynamic_beats_static_rate": float(group["dynamic_minus_static_rmse"].lt(0.0).mean()),
            "median_dynamic_minus_static_rmse": float(group["dynamic_minus_static_rmse"].median()),
            "selected_candidate_counts": group["selected_candidate"].value_counts().to_dict(),
        }
    dynamic_power = min(
        item["dynamic_selection_rate"]
        for item in summaries.values()
        if item["truth"] in {"on_grid_dynamic", "off_grid_dynamic"}
    )
    static_false_promotion = max(
        item["dynamic_selection_rate"] for item in summaries.values() if item["truth"] == "static_switch_null"
    )
    return {
        "paired_noise_sd_bpb": paired_sd,
        "noise_regimes": {
            name: {
                "run_intercept_variance_fraction": values[0],
                "step_ar1_correlation": values[1],
            }
            for name, values in design.SYNTHETIC_NOISE_REGIMES.items()
        },
        "replicates_per_truth": replicates,
        "summaries": summaries,
        "minimum_dynamic_selection_power": dynamic_power,
        "maximum_static_null_dynamic_false_promotion": static_false_promotion,
        "power_gate": dynamic_power >= 0.80,
        "false_promotion_gate": static_false_promotion <= 0.10,
    }


def synthetic_preflight(output_dir: Path) -> Path:
    """Exercise structural recovery and the endpoint hash seal without W&B."""
    protocol = load_protocol()
    manifest = load_manifest(protocol)
    response_truths = tuple(
        Candidate(
            "token_clock_acquisition_forgetting",
            clock="token",
            acquisition_rate=0.08,
            forgetting_ratio=1.0,
            response_mode=response_mode,
        )
        for response_mode in CONTROL_RESPONSE_MODES
    )
    response_mode_recovery: dict[str, str] = {}
    for response_truth in response_truths:
        response_raw = synthetic_raw(
            protocol=protocol,
            manifest=manifest,
            steps=transition_steps(protocol),
            true_candidate=response_truth,
            targets_override=(str(protocol["targets"]["primary"]),),
        )
        response_summary = synthetic_selection_summary(
            response_raw,
            protocol=protocol,
            manifest=manifest,
        )
        selected_response = str(response_summary["selected_candidate"])
        if selected_response != response_truth.candidate_id:
            raise AssertionError(
                f"Synthetic response-mode recovery failed: {selected_response} != {response_truth.candidate_id}"
            )
        response_mode_recovery[response_truth.response_mode] = selected_response

    true_candidate = Candidate(
        "token_clock_acquisition_forgetting",
        clock="token",
        acquisition_rate=0.08,
        forgetting_ratio=1.0,
        response_mode="unconditioned_signed",
    )
    synthetic_dir = output_dir / "synthetic_preflight"
    if synthetic_dir.exists():
        shutil.rmtree(synthetic_dir)
    synthetic_dir.mkdir(parents=True)
    transition = synthetic_raw(
        protocol=protocol,
        manifest=manifest,
        steps=transition_steps(protocol),
        true_candidate=true_candidate,
    )
    transition_path = synthetic_dir / "transition_observations.csv"
    transition.to_csv(transition_path, index=False)
    frozen_path = freeze_transition(synthetic_dir, transition_path)
    frozen = json.loads(frozen_path.read_text())
    selected = candidate_from_dict(frozen["selected_candidate"])
    if selected != true_candidate:
        raise AssertionError(f"Synthetic mechanism recovery failed: {selected} != {true_candidate}")
    if not bool(frozen["transition_gates"]["licensed_final_unseal"]):
        raise AssertionError("Synthetic transition did not license final unseal")
    if int(transition["global_step"].max()) != TRANSITION_MAX_STEP:
        raise AssertionError("Synthetic transition stage leaked the final endpoint")

    final = synthetic_raw(
        protocol=protocol,
        manifest=manifest,
        steps=sealed_decay_steps(protocol),
        true_candidate=true_candidate,
    )
    final_path = synthetic_dir / "final_observations.csv"
    final.to_csv(final_path, index=False)
    mark_endpoint_unsealed(synthetic_dir, frozen)
    evaluation_path = evaluate_final(synthetic_dir, final_path)
    evaluation = json.loads(evaluation_path.read_text())
    final_gates = evaluation["gates"]
    expected_passes = (
        "sealed_primary_transfer_pass",
        "code_transfer_pass",
    )
    failed_expected = [gate for gate in expected_passes if not bool(final_gates[gate])]
    if failed_expected:
        raise AssertionError(f"Synthetic implementation gates unexpectedly failed: {failed_expected}")
    try:
        freeze_transition(synthetic_dir, transition_path)
    except PermissionError:
        pass
    else:
        raise AssertionError("Transition refreeze was not blocked after endpoint unseal")
    result = {
        "protocol_sha256": protocol["protocol_sha256"],
        "true_candidate": true_candidate.candidate_id,
        "selected_candidate": selected.candidate_id,
        "response_mode_recovery": response_mode_recovery,
        "transition_rows": len(transition),
        "final_rows": len(final),
        "transition_max_step": int(transition["global_step"].max()),
        "sealed_final_step": int(final["global_step"].max()),
        "frozen_transition_sha256": sha256_path(frozen_path),
        "final_evaluation_sha256": sha256_path(evaluation_path),
        "scientific_all_final_gates_pass": bool(final_gates["all_final_gates_pass"]),
        "endpoint_refreeze_blocked": True,
        "scope": (
            "implementation preflight: tied-only spine recovery, exact mechanism recovery, blocked endpoint "
            "transfer, signed temporal response, and irreversible seal integrity"
        ),
        "passed": True,
    }
    result_path = synthetic_dir / "synthetic_preflight.json"
    write_json(result_path, result)
    return result_path


def write_synthetic_power_audit(output_dir: Path, *, replicates: int) -> Path:
    """Run the explicit Monte Carlo power audit separately from structural iteration."""
    if replicates < 100:
        raise ValueError("The frozen synthetic power audit requires at least 100 replicates per truth and regime")
    protocol = load_protocol()
    manifest = load_manifest(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = synthetic_power_audit(protocol=protocol, manifest=manifest, replicates=replicates)
    path = output_dir / "synthetic_power_audit.json"
    write_json(path, audit)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("synthetic-preflight")
    power = subparsers.add_parser("synthetic-power-audit")
    power.add_argument("--replicates", type=int, default=100)

    transition = subparsers.add_parser("materialize-transition")
    transition.add_argument("--wandb-path", default=WANDB_PATH)
    transition.add_argument("--refresh", action="store_true")
    transition.add_argument("--max-workers", type=int, default=16)

    freeze = subparsers.add_parser("freeze-transition")
    freeze.add_argument("--transition-data", type=Path)

    final = subparsers.add_parser("materialize-final")
    final.add_argument("--wandb-path", default=WANDB_PATH)
    final.add_argument("--refresh", action="store_true")
    final.add_argument("--max-workers", type=int, default=16)

    evaluate = subparsers.add_parser("evaluate-final")
    evaluate.add_argument("--final-data", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "synthetic-preflight":
        print(synthetic_preflight(output_dir))
        return
    if args.command == "synthetic-power-audit":
        print(write_synthetic_power_audit(output_dir, replicates=args.replicates))
        return
    if args.command == "materialize-transition":
        print(
            materialize_stage(
                output_dir=output_dir,
                stage="transition",
                wandb_path=args.wandb_path,
                refresh=args.refresh,
                max_workers=args.max_workers,
            )
        )
        return
    if args.command == "freeze-transition":
        transition_path = args.transition_data or output_dir / "transition_observations.csv"
        print(freeze_transition(output_dir, transition_path))
        return
    if args.command == "materialize-final":
        print(
            materialize_stage(
                output_dir=output_dir,
                stage="final",
                wandb_path=args.wandb_path,
                refresh=args.refresh,
                max_workers=args.max_workers,
            )
        )
        return
    if args.command == "evaluate-final":
        final_path = args.final_data or output_dir / "final_observations.csv"
        print(evaluate_final(output_dir, final_path))
        return
    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
