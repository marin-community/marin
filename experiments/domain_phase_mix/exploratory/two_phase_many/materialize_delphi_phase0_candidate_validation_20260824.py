# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///
"""Materialize exact-boundary prefix validation and freeze branch prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_WEIGHTS = (
    SCRIPT_DIR / "reference_outputs" / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase0_candidate_validation_20260824"
DEFAULT_EXPERIMENT_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase0_prefix_candidates_20260824"
)
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_CHECKPOINT_STEP = 2_399
PROTECTED_CANDIDATE = "observed_cap10_best"
DEPLOYMENT_CANDIDATES = (
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
)
EXPECTED_CANDIDATES = (*DEPLOYMENT_CANDIDATES, PROTECTED_CANDIDATE, "proportional_control")
SELECTED_PREFIX_COUNT = 4
PRIMARY_BRANCH_SEED = 0
STABILITY_BRANCH_SEED = 1
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
DIAGNOSTIC_METRIC = "eval/uncheatable_eval/github_cpp/bpb"
MAX_MEAN_PLUS_SEM_BOUNDARY_REGRESSION = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--prefix-replay-code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selected-manifest-uri",
        default=f"{DEFAULT_EXPERIMENT_ROOT}/selected-prefixes/selected_prefixes.json",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_from_leaf(leaf: str, candidate_ids: tuple[str, ...]) -> tuple[str, int]:
    for candidate_id in sorted(candidate_ids, key=len, reverse=True):
        match = re.fullmatch(rf"prefix_{re.escape(candidate_id)}_seed([0-9]+)-[0-9a-f]+", leaf)
        if match:
            return candidate_id, int(match.group(1))
    raise ValueError(f"Unexpected candidate output identity: {leaf}")


def read_json_lines(fs: fsspec.AbstractFileSystem, path: str) -> list[dict[str, object]]:
    with fs.open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def materialize_rows(
    experiment_root: str,
    candidate_ids: tuple[str, ...],
    candidate_weights_sha256: str,
    prefix_replay_code_commit: str,
) -> pd.DataFrame:
    fs, root = fsspec.core.url_to_fs(experiment_root)
    metric_paths = sorted(fs.glob(f"{root}/*/checkpoints/eval_metrics.jsonl"))
    expected = {(candidate_id, seed) for candidate_id in candidate_ids for seed in EXPECTED_SEEDS}
    rows = []
    observed: set[tuple[str, int]] = set()
    for metric_path in metric_paths:
        leaf = PurePosixPath(metric_path).parents[1].name
        try:
            candidate_id, seed = identity_from_leaf(leaf, candidate_ids)
        except ValueError:
            continue
        identity = (candidate_id, seed)
        if identity in observed:
            raise ValueError(f"Duplicate candidate output: {identity}")
        records = [
            row for row in read_json_lines(fs, metric_path) if int(row.get("step", -1)) == EXPECTED_CHECKPOINT_STEP
        ]
        if len(records) != 1:
            raise ValueError(
                f"Expected one step-{EXPECTED_CHECKPOINT_STEP} metric row for {identity}; found {len(records)}"
            )
        checkpoint_path = f"{PurePosixPath(metric_path).parents[1]}/checkpoints/step-{EXPECTED_CHECKPOINT_STEP}"
        hf_path = f"{PurePosixPath(metric_path).parents[1]}/hf/step-{EXPECTED_CHECKPOINT_STEP}"
        if not fs.exists(f"{checkpoint_path}/metadata.json"):
            raise FileNotFoundError(f"Full trainer checkpoint is incomplete: {checkpoint_path}")
        if not fs.exists(f"{hf_path}/config.json"):
            raise FileNotFoundError(f"HF boundary checkpoint is incomplete: {hf_path}")
        record = records[0]
        if PRIMARY_METRIC not in record or DIAGNOSTIC_METRIC not in record:
            raise ValueError(f"Required boundary metrics are missing for {identity}")
        output_root = experiment_root.rstrip("/") + "/" + leaf
        provenance_path = f"{PurePosixPath(metric_path).parents[1]}/prefix_provenance.json"
        if not fs.exists(provenance_path):
            raise FileNotFoundError(f"Candidate provenance is missing: {provenance_path}")
        with fs.open(provenance_path, "rb") as handle:
            provenance_bytes = handle.read()
        provenance = json.loads(provenance_bytes)
        expected_provenance = {
            "experiment_name": "pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824",
            "candidate_id": candidate_id,
            "candidate_weights_sha256": candidate_weights_sha256,
            "replay_code_commit": prefix_replay_code_commit,
            "run_name": f"prefix_{candidate_id}_seed{seed}",
            "run_order": candidate_ids.index(candidate_id) * len(EXPECTED_SEEDS) + EXPECTED_SEEDS.index(seed),
            "run_id": 930_000 + candidate_ids.index(candidate_id) * len(EXPECTED_SEEDS) + EXPECTED_SEEDS.index(seed),
            "data_seed": 930_000 + seed,
            "trainer_seed": seed,
            "checkpoint_uri": output_root + f"/checkpoints/step-{EXPECTED_CHECKPOINT_STEP}",
            "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            "trainer_state_step": EXPECTED_CHECKPOINT_STEP + 1,
        }
        for key, expected_value in expected_provenance.items():
            if provenance.get(key) != expected_value:
                raise ValueError(f"Candidate provenance mismatch for {identity}: {key}={provenance.get(key)!r}")
        if not isinstance(provenance.get("phase_weights_sha256"), str):
            raise ValueError(f"Candidate phase-weight hash is missing for {identity}")
        row = {
            "candidate_id": candidate_id,
            "repeat_seed": seed,
            "output_root": output_root,
            "checkpoint_uri": output_root + f"/checkpoints/step-{EXPECTED_CHECKPOINT_STEP}",
            "hf_checkpoint_uri": output_root + f"/hf/step-{EXPECTED_CHECKPOINT_STEP}",
            "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        }
        row.update(
            {
                key.removeprefix("eval/uncheatable_eval/").replace("/", "::"): float(value)
                for key, value in record.items()
                if key.startswith("eval/uncheatable_eval/") and isinstance(value, (float, int))
            }
        )
        rows.append(row)
        observed.add(identity)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(f"Candidate validation is incomplete: missing={missing}, unexpected={unexpected}")
    return pd.DataFrame(rows).sort_values(["candidate_id", "repeat_seed"]).reset_index(drop=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("candidate_id", as_index=False)
        .agg(
            uncheatable_mean=("bpb", "mean"),
            uncheatable_sd=("bpb", "std"),
            github_cpp_mean=("github_cpp::bpb", "mean"),
            github_cpp_sd=("github_cpp::bpb", "std"),
        )
        .sort_values("candidate_id")
    )
    summary["uncheatable_sem"] = summary.uncheatable_sd / math.sqrt(len(EXPECTED_SEEDS))
    summary["selection_score_mean_plus_sem"] = summary.uncheatable_mean + summary.uncheatable_sem
    return summary


def boundary_safety(results: pd.DataFrame) -> pd.DataFrame:
    incumbent = results[results.candidate_id.eq(PROTECTED_CANDIDATE)].set_index("repeat_seed")
    rows = []
    for candidate_id in DEPLOYMENT_CANDIDATES:
        candidate = results[results.candidate_id.eq(candidate_id)].set_index("repeat_seed")
        delta = candidate.loc[list(EXPECTED_SEEDS), "bpb"] - incumbent.loc[list(EXPECTED_SEEDS), "bpb"]
        mean = float(delta.mean())
        sem = float(delta.std() / math.sqrt(len(delta)))
        rows.append(
            {
                "candidate_id": candidate_id,
                "paired_uncheatable_delta_mean": mean,
                "paired_uncheatable_delta_sem": sem,
                "paired_uncheatable_delta_mean_plus_sem": mean + sem,
                "passes_boundary_safety": mean + sem <= MAX_MEAN_PLUS_SEM_BOUNDARY_REGRESSION,
            }
        )
    safety = pd.DataFrame(rows)
    failed = safety[~safety.passes_boundary_safety]
    if not failed.empty:
        raise ValueError(
            "Deployment prefix failed the preregistered boundary safety gate: "
            + failed.to_dict(orient="records").__repr__()
        )
    return safety


def selected_manifest(
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    safety: pd.DataFrame,
    candidate_weights_sha256: str,
    prefix_replay_code_commit: str,
) -> dict[str, object]:
    selected = (PROTECTED_CANDIDATE, *DEPLOYMENT_CANDIDATES)
    rows = results[
        results.candidate_id.isin(selected) & results.repeat_seed.isin((PRIMARY_BRANCH_SEED, STABILITY_BRANCH_SEED))
    ]
    if len(rows) != SELECTED_PREFIX_COUNT * 2:
        raise ValueError("Selected prefix checkpoint rows are incomplete")
    return {
        "candidate_weights_sha256": candidate_weights_sha256,
        "prefix_replay_code_commit": prefix_replay_code_commit,
        "selection_target": "exact-boundary Uncheatable BPB",
        "selection_rule": (
            "Candidate identities were frozen before boundary validation. Protect observed_cap10_best and "
            "admit all three ensemble-KL challengers only if each paired Uncheatable mean-plus-SEM regression "
            f"is at most {MAX_MEAN_PLUS_SEM_BOUNDARY_REGRESSION:g} BPB. Proportional and GitHub C++ are diagnostic only."
        ),
        "selected_candidate_ids": list(selected),
        "prefixes": (
            rows[["candidate_id", "repeat_seed", "checkpoint_uri", "provenance_sha256"]].to_dict(orient="records")
        ),
        "candidate_summary": summary.to_dict(orient="records"),
        "boundary_safety": safety.to_dict(orient="records"),
    }


def write_uri_exact(uri: str, payload: bytes) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        with fs.open(path, "rb") as handle:
            existing = handle.read()
        if existing != payload:
            raise ValueError(f"Refusing to replace a different frozen selection manifest: {uri}")
        return
    parent = str(PurePosixPath(path).parent)
    fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as handle:
        handle.write(payload)


def main() -> None:
    args = parse_args()
    candidate_frame = pd.read_csv(args.candidate_weights)
    candidate_ids = tuple(candidate_frame.candidate_id.drop_duplicates())
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError(f"Candidate identities changed: {candidate_ids} != {EXPECTED_CANDIDATES}")
    candidate_weights_sha256 = file_sha256(args.candidate_weights)
    results = materialize_rows(
        args.experiment_root,
        candidate_ids,
        candidate_weights_sha256,
        args.prefix_replay_code_commit,
    )
    summary = summarize(results)
    safety = boundary_safety(results)
    payload = selected_manifest(
        results=results,
        summary=summary,
        safety=safety,
        candidate_weights_sha256=candidate_weights_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
    )
    payload_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_uri_exact(args.selected_manifest_uri, payload_bytes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "candidate_boundary_results.csv", index=False)
    summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    safety.to_csv(args.output_dir / "boundary_safety.csv", index=False)
    (args.output_dir / "selected_prefixes.json").write_bytes(payload_bytes)
    coverage = {
        "candidate_count": len(candidate_ids),
        "seed_count": len(EXPECTED_SEEDS),
        "result_rows": len(results),
        "selected_manifest_uri": args.selected_manifest_uri,
        "selected_manifest_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "selected_candidate_ids": payload["selected_candidate_ids"],
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    print("\n", json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
