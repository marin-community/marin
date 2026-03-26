# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate action-diversity diagnostics for an offline-RL policy."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.policy_artifact import PolicyArtifactV1, clip_action, load_policy_artifact

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except ImportError:
    ks_2samp = None
    wasserstein_distance = None

DEFAULT_TRANSITIONS_PATH = "experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder/transitions.parquet"
DEFAULT_POLICY_ARTIFACT = (
    "experiments/domain_phase_mix/offline_rl/policy_payloads/three_phase_starcoder_cql_v1/policy_artifact_cluster.json"
)
DEFAULT_OUTPUT_DIR = "experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder/action_diversity_report"
DEFAULT_DPI = 320
DEFAULT_SWEEP_MIN = -2.5
DEFAULT_SWEEP_MAX = 2.5
DEFAULT_SWEEP_POINTS = 81
DEFAULT_RANDOM_PROBE_SAMPLES = 50_000
DEFAULT_RANDOM_SEED = 0
PHASE_ORDER = ("all", "phase_0", "phase_1", "phase_2")


@dataclass(frozen=True)
class ActionDiversityReportConfig:
    """Configuration for action-diversity reporting."""

    transitions_path: str
    policy_artifact_path: str
    output_dir: str
    device: str = "cpu"
    sweep_min: float = DEFAULT_SWEEP_MIN
    sweep_max: float = DEFAULT_SWEEP_MAX
    sweep_points: int = DEFAULT_SWEEP_POINTS
    random_probe_samples: int = DEFAULT_RANDOM_PROBE_SAMPLES
    random_seed: int = DEFAULT_RANDOM_SEED
    dpi: int = DEFAULT_DPI


def _load_policy(artifact_path: str | Path, device: str):
    artifact = load_policy_artifact(artifact_path)
    artifact_file = Path(artifact_path).resolve()
    artifact_dir = artifact_file.parent
    model_path = Path(artifact.model_path)

    if model_path.is_absolute():
        if not model_path.exists():
            candidates = [
                artifact_dir / model_path.name,
                artifact_dir / "terminal_reward" / model_path.name,
                artifact_dir / "delta_reward" / model_path.name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Policy model path {model_path} does not exist and no fallback was found near {artifact_dir}."
                )
    else:
        model_path = (artifact_dir / model_path).resolve()

    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError("d3rlpy is required for action-diversity reporting.") from exc

    if hasattr(d3rlpy, "load_learnable"):
        policy = d3rlpy.load_learnable(str(model_path), device=device)
    else:
        from d3rlpy.base import load_learnable

        policy = load_learnable(str(model_path), device=device)

    return artifact, policy


def _predict_actions(policy, normalized_states: np.ndarray, artifact: PolicyArtifactV1) -> tuple[np.ndarray, np.ndarray]:
    action = policy.predict(normalized_states)
    action_arr = np.asarray(action, dtype=np.float32)
    if action_arr.ndim == 1:
        raw = action_arr
    else:
        raw = action_arr.reshape(action_arr.shape[0], -1)[:, 0]

    clipped = np.asarray([clip_action(float(value), artifact) for value in raw], dtype=np.float32)
    return raw, clipped


def _phase_label(phase_idx: int | float) -> str:
    return f"phase_{int(phase_idx)}"


def _series_stats(values: np.ndarray) -> dict[str, float]:
    quantiles = (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)
    out = {
        "count": float(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }
    for q in quantiles:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(values, q))
    return out


def _support_row(
    *,
    phase: str,
    raw: np.ndarray,
    clipped: np.ndarray,
    action_low: float,
    action_high: float,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"phase": phase}
    row.update({f"raw_{k}": v for k, v in _series_stats(raw).items()})
    row.update({f"clipped_{k}": v for k, v in _series_stats(clipped).items()})
    row["frac_raw_below_low"] = float((raw < action_low).mean())
    row["frac_raw_above_high"] = float((raw > action_high).mean())
    row["frac_clipped_at_low"] = float(np.isclose(clipped, action_low).mean())
    row["frac_clipped_at_high"] = float(np.isclose(clipped, action_high).mean())
    return row


def _distribution_row(phase: str, behavior: np.ndarray, policy_clipped: np.ndarray) -> dict[str, float | str]:
    row: dict[str, float | str] = {"phase": phase, "count": float(behavior.size)}
    if wasserstein_distance is None:
        row["wasserstein_1"] = math.nan
    else:
        row["wasserstein_1"] = float(wasserstein_distance(behavior, policy_clipped))

    if ks_2samp is None:
        row["ks_statistic"] = math.nan
        row["ks_pvalue"] = math.nan
    else:
        ks = ks_2samp(behavior, policy_clipped)
        row["ks_statistic"] = float(ks.statistic)
        row["ks_pvalue"] = float(ks.pvalue)

    return row


def _build_predictions(transitions: pd.DataFrame, artifact: PolicyArtifactV1, policy) -> pd.DataFrame:
    required_cols = ["wandb_run_id", "t", "action_starcoder", *artifact.state_keys]
    missing = [column for column in required_cols if column not in transitions.columns]
    if missing:
        raise ValueError(f"Transitions are missing required columns: {missing}")

    states = transitions.loc[:, list(artifact.state_keys)].to_numpy(dtype=np.float32)
    mean = np.asarray(artifact.state_mean, dtype=np.float32)
    std = np.asarray(artifact.state_std, dtype=np.float32)
    safe_std = np.where(std <= 0.0, 1.0, std)
    normalized = (states - mean) / safe_std

    raw, clipped = _predict_actions(policy, normalized, artifact)

    prediction_df = transitions.loc[:, ["wandb_run_id", "t", "action_starcoder"]].copy()
    prediction_df = prediction_df.rename(columns={"action_starcoder": "behavior_action"})
    prediction_df["phase"] = prediction_df["t"].map(_phase_label)
    prediction_df["policy_action_raw"] = raw
    prediction_df["policy_action_clipped"] = clipped
    return prediction_df


def _compute_support_and_coverage(
    predictions: pd.DataFrame,
    *,
    action_low: float,
    action_high: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_rows = []
    distribution_rows = []

    group_frames: list[tuple[str, pd.DataFrame]] = [("all", predictions), *list(predictions.groupby("phase"))]
    for phase, frame in group_frames:
        raw = frame["policy_action_raw"].to_numpy(dtype=np.float32)
        clipped = frame["policy_action_clipped"].to_numpy(dtype=np.float32)
        behavior = frame["behavior_action"].to_numpy(dtype=np.float32)
        support_rows.append(
            _support_row(
                phase=phase,
                raw=raw,
                clipped=clipped,
                action_low=action_low,
                action_high=action_high,
            )
        )
        distribution_rows.append(_distribution_row(phase=phase, behavior=behavior, policy_clipped=clipped))

    support_df = pd.DataFrame(support_rows)
    support_df["phase"] = pd.Categorical(support_df["phase"], categories=PHASE_ORDER, ordered=True)
    support_df = support_df.sort_values("phase").reset_index(drop=True)

    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df["phase"] = pd.Categorical(distribution_df["phase"], categories=PHASE_ORDER, ordered=True)
    distribution_df = distribution_df.sort_values("phase").reset_index(drop=True)
    return support_df, distribution_df


def _compute_sensitivity_curves(
    policy,
    artifact: PolicyArtifactV1,
    *,
    sweep_min: float,
    sweep_max: float,
    sweep_points: int,
) -> pd.DataFrame:
    grid = np.linspace(sweep_min, sweep_max, sweep_points, dtype=np.float32)
    base = np.zeros((sweep_points, len(artifact.state_keys)), dtype=np.float32)

    rows = []
    for feature_idx, feature_key in enumerate(artifact.state_keys):
        states = base.copy()
        states[:, feature_idx] = grid
        raw, clipped = _predict_actions(policy, states, artifact)
        for value, raw_action, clipped_action in zip(grid, raw, clipped, strict=True):
            rows.append(
                {
                    "feature_key": feature_key,
                    "state_z": float(value),
                    "policy_action_raw": float(raw_action),
                    "policy_action_clipped": float(clipped_action),
                }
            )

    return pd.DataFrame(rows)


def _compute_random_probe(
    policy,
    artifact: PolicyArtifactV1,
    *,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n_samples, len(artifact.state_keys))).astype(np.float32)
    raw, clipped = _predict_actions(policy, samples, artifact)

    out = {
        "n_samples": float(n_samples),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_std": float(raw.std(ddof=0)),
        "clipped_min": float(clipped.min()),
        "clipped_max": float(clipped.max()),
        "clipped_mean": float(clipped.mean()),
        "clipped_std": float(clipped.std(ddof=0)),
        "frac_raw_below_low": float((raw < artifact.action_low).mean()),
        "frac_raw_above_high": float((raw > artifact.action_high).mean()),
        "frac_clipped_at_low": float(np.isclose(clipped, artifact.action_low).mean()),
        "frac_clipped_at_high": float(np.isclose(clipped, artifact.action_high).mean()),
    }
    for q in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999):
        out[f"raw_q{q:.3f}"] = float(np.quantile(raw, q))
        out[f"clipped_q{q:.3f}"] = float(np.quantile(clipped, q))
    return out


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = (np.arange(x.size, dtype=np.float64) + 1.0) / float(x.size)
    return x, y


def _plot_histograms(predictions: pd.DataFrame, artifact: PolicyArtifactV1, output_file: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [("all", predictions), *list(predictions.groupby("phase"))]
    bins = 40
    x_min = min(
        float(predictions["behavior_action"].min()),
        float(predictions["policy_action_raw"].min()),
        artifact.action_low,
    ) - 0.05
    x_max = max(
        float(predictions["behavior_action"].max()),
        float(predictions["policy_action_raw"].max()),
        artifact.action_high,
    ) + 0.05

    for axis, (label, frame) in zip(axes.ravel(), panels, strict=True):
        axis.hist(frame["behavior_action"], bins=bins, range=(x_min, x_max), alpha=0.35, label="behavior")
        axis.hist(frame["policy_action_raw"], bins=bins, range=(x_min, x_max), alpha=0.35, label="policy raw")
        axis.hist(frame["policy_action_clipped"], bins=bins, range=(x_min, x_max), alpha=0.35, label="policy clipped")
        axis.axvline(artifact.action_low, color="black", linestyle="--", linewidth=1)
        axis.axvline(artifact.action_high, color="black", linestyle="--", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Action (StarCoder weight)")
        axis.set_ylabel("Count")
    axes[0, 0].legend()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def _plot_ecdfs(predictions: pd.DataFrame, output_file: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [("all", predictions), *list(predictions.groupby("phase"))]

    for axis, (label, frame) in zip(axes.ravel(), panels, strict=True):
        behavior_x, behavior_y = _ecdf(frame["behavior_action"].to_numpy(dtype=np.float32))
        policy_x, policy_y = _ecdf(frame["policy_action_clipped"].to_numpy(dtype=np.float32))
        axis.plot(behavior_x, behavior_y, label="behavior")
        axis.plot(policy_x, policy_y, label="policy clipped")
        axis.set_title(label)
        axis.set_xlabel("Action (StarCoder weight)")
        axis.set_ylabel("ECDF")
        axis.set_ylim(0.0, 1.0)
    axes[0, 0].legend()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def _plot_clip_rates(support_df: pd.DataFrame, output_file: Path, dpi: int) -> None:
    rows = support_df.copy()
    x = np.arange(len(rows), dtype=np.float32)
    width = 0.35

    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.bar(x - width / 2, rows["frac_clipped_at_low"], width, label="clipped @ low")
    axis.bar(x + width / 2, rows["frac_clipped_at_high"], width, label="clipped @ high")
    axis.set_xticks(x, rows["phase"].astype(str))
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction")
    axis.set_title("Clipping Rates by Phase")
    axis.legend()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def _plot_sensitivity_curves(
    sensitivity_df: pd.DataFrame,
    artifact: PolicyArtifactV1,
    output_file: Path,
    dpi: int,
) -> None:
    feature_keys = list(artifact.state_keys)
    n_features = len(feature_keys)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.3 * n_rows), constrained_layout=True)
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, feature_key in zip(flat_axes[:n_features], feature_keys, strict=True):
        frame = sensitivity_df[sensitivity_df["feature_key"] == feature_key]
        axis.plot(frame["state_z"], frame["policy_action_raw"], label="raw")
        axis.plot(frame["state_z"], frame["policy_action_clipped"], label="clipped")
        axis.axhline(artifact.action_low, color="black", linestyle="--", linewidth=1)
        axis.axhline(artifact.action_high, color="black", linestyle="--", linewidth=1)
        axis.set_title(feature_key)
        axis.set_xlabel("Feature value (z-score)")
        axis.set_ylabel("Action")
    for axis in flat_axes[n_features:]:
        axis.axis("off")
    flat_axes[0].legend()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def run_report(config: ActionDiversityReportConfig) -> dict[str, Any]:
    """Run action-diversity diagnostics and write CSV/JSON/plot outputs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact, policy = _load_policy(config.policy_artifact_path, config.device)
    transitions = pd.read_parquet(config.transitions_path)
    predictions = _build_predictions(transitions, artifact, policy)
    support_df, distribution_df = _compute_support_and_coverage(
        predictions,
        action_low=artifact.action_low,
        action_high=artifact.action_high,
    )
    sensitivity_df = _compute_sensitivity_curves(
        policy,
        artifact,
        sweep_min=config.sweep_min,
        sweep_max=config.sweep_max,
        sweep_points=config.sweep_points,
    )
    random_probe = _compute_random_probe(
        policy,
        artifact,
        n_samples=config.random_probe_samples,
        seed=config.random_seed,
    )

    predictions.to_csv(output_dir / "action_predictions.csv", index=False)
    support_df.to_csv(output_dir / "action_support_metrics.csv", index=False)
    distribution_df.to_csv(output_dir / "distribution_coverage_metrics.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_curves.csv", index=False)

    dpi = max(DEFAULT_DPI, int(config.dpi))
    _plot_histograms(predictions, artifact, output_dir / "action_histograms.png", dpi=dpi)
    _plot_ecdfs(predictions, output_dir / "action_ecdfs.png", dpi=dpi)
    _plot_clip_rates(support_df, output_dir / "clip_rates.png", dpi=dpi)
    _plot_sensitivity_curves(sensitivity_df, artifact, output_dir / "feature_sensitivity.png", dpi=dpi)

    summary = {
        "transitions_path": str(Path(config.transitions_path).resolve()),
        "policy_artifact_path": str(Path(config.policy_artifact_path).resolve()),
        "model_kind": artifact.kind,
        "action_low": artifact.action_low,
        "action_high": artifact.action_high,
        "n_predictions": len(predictions),
        "support_metrics_file": "action_support_metrics.csv",
        "coverage_metrics_file": "distribution_coverage_metrics.csv",
        "predictions_file": "action_predictions.csv",
        "sensitivity_file": "sensitivity_curves.csv",
        "plots": [
            "action_histograms.png",
            "action_ecdfs.png",
            "clip_rates.png",
            "feature_sensitivity.png",
        ],
        "random_probe": random_probe,
    }
    with (output_dir / "random_probe_metrics.json").open("w") as f:
        json.dump(random_probe, f, indent=2, sort_keys=True)
    with (output_dir / "action_diversity_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate action-diversity diagnostics for a trained offline policy.")
    parser.add_argument(
        "--transitions-path",
        type=str,
        default=DEFAULT_TRANSITIONS_PATH,
        help="Path to transitions.parquet produced by build_transitions.",
    )
    parser.add_argument(
        "--policy-artifact-path",
        type=str,
        default=DEFAULT_POLICY_ARTIFACT,
        help="Path to policy_artifact.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where report files and plots are written.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for policy inference (cpu, mps, cuda:0).")
    parser.add_argument("--sweep-min", type=float, default=DEFAULT_SWEEP_MIN, help="Min z-score for feature sweeps.")
    parser.add_argument("--sweep-max", type=float, default=DEFAULT_SWEEP_MAX, help="Max z-score for feature sweeps.")
    parser.add_argument(
        "--sweep-points",
        type=int,
        default=DEFAULT_SWEEP_POINTS,
        help="Number of points in each one-feature sweep.",
    )
    parser.add_argument(
        "--random-probe-samples",
        type=int,
        default=DEFAULT_RANDOM_PROBE_SAMPLES,
        help="Number of random normalized states for global action-range probing.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for random probe.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Plot DPI (minimum enforced: {DEFAULT_DPI}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_report(
        ActionDiversityReportConfig(
            transitions_path=args.transitions_path,
            policy_artifact_path=args.policy_artifact_path,
            output_dir=args.output_dir,
            device=args.device,
            sweep_min=args.sweep_min,
            sweep_max=args.sweep_max,
            sweep_points=args.sweep_points,
            random_probe_samples=args.random_probe_samples,
            random_seed=args.random_seed,
            dpi=args.dpi,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
