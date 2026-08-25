# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///
"""Materialize cap-4/cap-6 boundary results and select one prefix per cap."""

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
DEFAULT_CANDIDATE_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase0_harsh_cap_candidates_20260825"
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "training_candidate_weights.csv"
DEFAULT_ALIASES = DEFAULT_CANDIDATE_DIR / "candidate_aliases.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase0_harsh_cap_validation_20260825"
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_harsh_cap_candidates_v6e_r2_20260825"
DEFAULT_EXPERIMENT_ROOT = f"gs://marin-us-east5/{EXPERIMENT_NAME}"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_CHECKPOINT_STEP = 2_399
RUN_ID_BASE = 965_000
DATA_SEED_BASE = 965_000
PRIMARY_BRANCH_SEED = 0
STABILITY_BRANCH_SEED = 1
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
DIAGNOSTIC_METRIC = "eval/uncheatable_eval/github_cpp/bpb"
EXPECTED_TPU_TYPE = "v6e-8"
EXPECTED_TPU_REGION = "us-east5"
EXPECTED_TPU_ZONE = "us-east5-b"
EXPECTED_DEVICE_COUNT = 8
PANEL_HARDWARE_STATUS = "v6e_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--candidate-aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--prefix-replay-code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        default=(4, 6),
        choices=(4, 6),
        help="Materialize and select only completed candidate caps.",
    )
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


def candidate_run_order(candidate_id: str, seed: int, all_candidate_ids: tuple[str, ...]) -> int:
    if candidate_id not in all_candidate_ids or seed not in EXPECTED_SEEDS:
        raise ValueError(f"Unknown candidate identity: {candidate_id}/seed{seed}")
    return all_candidate_ids.index(candidate_id) * len(EXPECTED_SEEDS) + EXPECTED_SEEDS.index(seed)


def unique_terminal_metric(records: list[dict[str, object]], identity: tuple[str, int]) -> dict[str, object]:
    if not records:
        raise ValueError(f"Expected a step-{EXPECTED_CHECKPOINT_STEP} metric row for {identity}")
    if any(record != records[0] for record in records[1:]):
        raise ValueError(f"Conflicting step-{EXPECTED_CHECKPOINT_STEP} metric rows for {identity}")
    return records[0]


def materialize_rows(
    *,
    experiment_root: str,
    candidate_ids: tuple[str, ...],
    all_candidate_ids: tuple[str, ...],
    candidate_weights_sha256: str,
    candidate_aliases_sha256: str,
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
        record = unique_terminal_metric(records, identity)
        output_root = experiment_root.rstrip("/") + "/" + leaf
        checkpoint_uri = output_root + f"/checkpoints/step-{EXPECTED_CHECKPOINT_STEP}"
        hf_checkpoint_uri = output_root + f"/hf/step-{EXPECTED_CHECKPOINT_STEP}"
        checkpoint_path = f"{PurePosixPath(metric_path).parents[1]}/checkpoints/step-{EXPECTED_CHECKPOINT_STEP}"
        if not fs.exists(f"{checkpoint_path}/metadata.json"):
            raise FileNotFoundError(f"Full trainer checkpoint is incomplete: {checkpoint_uri}")
        hf_path = f"{PurePosixPath(metric_path).parents[1]}/hf/step-{EXPECTED_CHECKPOINT_STEP}"
        if not fs.exists(f"{hf_path}/config.json"):
            raise FileNotFoundError(f"HF boundary checkpoint is incomplete: {hf_checkpoint_uri}")
        if PRIMARY_METRIC not in record or DIAGNOSTIC_METRIC not in record:
            raise ValueError(f"Required boundary metrics are missing for {identity}")
        provenance_path = f"{PurePosixPath(metric_path).parents[1]}/prefix_provenance.json"
        if not fs.exists(provenance_path):
            raise FileNotFoundError(f"Candidate provenance is missing: {provenance_path}")
        with fs.open(provenance_path, "rb") as handle:
            provenance_bytes = handle.read()
        provenance = json.loads(provenance_bytes)
        run_order = candidate_run_order(candidate_id, seed, all_candidate_ids)
        expected_provenance = {
            "experiment_name": EXPERIMENT_NAME,
            "candidate_id": candidate_id,
            "candidate_weights_sha256": candidate_weights_sha256,
            "candidate_aliases_sha256": candidate_aliases_sha256,
            "replay_code_commit": prefix_replay_code_commit,
            "run_name": f"prefix_{candidate_id}_seed{seed}",
            "run_order": run_order,
            "run_id": RUN_ID_BASE + run_order,
            "data_seed": DATA_SEED_BASE + seed,
            "trainer_seed": seed,
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            "trainer_state_step": EXPECTED_CHECKPOINT_STEP + 1,
            "tpu_type": EXPECTED_TPU_TYPE,
            "tpu_region": EXPECTED_TPU_REGION,
            "tpu_zone": EXPECTED_TPU_ZONE,
            "observed_global_device_count": EXPECTED_DEVICE_COUNT,
            "observed_local_device_count": EXPECTED_DEVICE_COUNT,
            "panel_hardware_status": PANEL_HARDWARE_STATUS,
        }
        for key, expected_value in expected_provenance.items():
            if provenance.get(key) != expected_value:
                raise ValueError(f"Candidate provenance mismatch for {identity}: {key}={provenance.get(key)!r}")
        if not isinstance(provenance.get("phase_weights_sha256"), str):
            raise ValueError(f"Candidate phase-weight hash is missing for {identity}")
        device_kinds = provenance.get("observed_device_kinds")
        if (
            not isinstance(device_kinds, list)
            or not device_kinds
            or any("v6" not in str(kind).lower() for kind in device_kinds)
        ):
            raise ValueError(f"Candidate did not run on v6 hardware for {identity}: {device_kinds}")
        row = {
            "canonical_candidate_id": candidate_id,
            "repeat_seed": seed,
            "output_root": output_root,
            "checkpoint_uri": checkpoint_uri,
            "hf_checkpoint_uri": hf_checkpoint_uri,
            "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            "tpu_type": EXPECTED_TPU_TYPE,
            "tpu_region": EXPECTED_TPU_REGION,
            "tpu_zone": EXPECTED_TPU_ZONE,
            "panel_hardware_status": PANEL_HARDWARE_STATUS,
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
    return pd.DataFrame(rows).sort_values(["canonical_candidate_id", "repeat_seed"]).reset_index(drop=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("canonical_candidate_id", as_index=False)
        .agg(
            uncheatable_mean=("bpb", "mean"),
            uncheatable_sd=("bpb", "std"),
            github_cpp_mean=("github_cpp::bpb", "mean"),
            github_cpp_sd=("github_cpp::bpb", "std"),
        )
        .sort_values("canonical_candidate_id")
    )
    summary["uncheatable_sem"] = summary.uncheatable_sd / math.sqrt(len(EXPECTED_SEEDS))
    return summary


def kl_penalty(candidate_id: str) -> float:
    label = candidate_id.rsplit("_kl", maxsplit=1)[1]
    return float(label.replace("p", "."))


def select_aliases(
    aliases: pd.DataFrame,
    summary: pd.DataFrame,
    expected_caps: tuple[int, ...] = (4, 6),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alias_results = aliases.merge(summary, on="canonical_candidate_id", validate="many_to_one")
    alias_results["kl_penalty"] = [
        kl_penalty(candidate_id) if eligible else math.nan
        for candidate_id, eligible in zip(
            alias_results.candidate_id, alias_results.selection_eligible.astype(bool), strict=True
        )
    ]
    eligible = alias_results[alias_results.selection_eligible.astype(bool)].copy()
    selected = (
        eligible.sort_values(["cap_epochs", "uncheatable_mean", "kl_penalty", "alias_id"])
        .groupby("cap_epochs", as_index=False)
        .first()
    )
    if tuple(selected.cap_epochs.astype(int)) != expected_caps:
        raise ValueError(f"Expected one selected prefix for caps {expected_caps}; got {tuple(selected.cap_epochs)}")
    return alias_results, selected


def selected_manifest(
    *,
    results: pd.DataFrame,
    alias_results: pd.DataFrame,
    selected: pd.DataFrame,
    candidate_weights_sha256: str,
    candidate_aliases_sha256: str,
    prefix_replay_code_commit: str,
) -> dict[str, object]:
    selected_ids = tuple(selected.canonical_candidate_id)
    rows = results[
        results.canonical_candidate_id.isin(selected_ids)
        & results.repeat_seed.isin((PRIMARY_BRANCH_SEED, STABILITY_BRANCH_SEED))
    ]
    if len(rows) != 2 * len(selected_ids):
        raise ValueError("Selected prefix checkpoint rows are incomplete")
    return {
        "candidate_weights_sha256": candidate_weights_sha256,
        "candidate_aliases_sha256": candidate_aliases_sha256,
        "prefix_replay_code_commit": prefix_replay_code_commit,
        "selection_target": "mean exact-boundary Uncheatable BPB over paired seeds 0, 1, and 2",
        "selection_rule": (
            "Within each cap, select the eligible KL candidate with the lowest three-seed mean boundary "
            "Uncheatable BPB; break an exact tie by the lower KL penalty. Controls are diagnostic only."
        ),
        "prefix_hardware": {
            "tpu_type": EXPECTED_TPU_TYPE,
            "region": EXPECTED_TPU_REGION,
            "zone": EXPECTED_TPU_ZONE,
        },
        "panel_hardware_status": PANEL_HARDWARE_STATUS,
        "frontier_claim_requirement": "confirm the selected final policy with fresh v6e-8 repeats",
        "selected_aliases": selected.to_dict(orient="records"),
        "prefixes": (
            rows[["canonical_candidate_id", "repeat_seed", "checkpoint_uri", "provenance_sha256"]].to_dict(
                orient="records"
            )
        ),
        "all_alias_results": alias_results.to_dict(orient="records"),
    }


def write_uri_exact(uri: str, payload: bytes) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        with fs.open(path, "rb") as handle:
            existing = handle.read()
        if existing != payload:
            raise ValueError(f"Refusing to replace a different frozen selection manifest: {uri}")
        return
    fs.makedirs(str(PurePosixPath(path).parent), exist_ok=True)
    with fs.open(path, "wb") as handle:
        handle.write(payload)


def main() -> None:
    args = parse_args()
    weights = pd.read_csv(args.candidate_weights)
    aliases = pd.read_csv(args.candidate_aliases)
    caps = tuple(sorted(set(args.caps)))
    selected_aliases = aliases[
        aliases.cap_epochs.astype(int).isin(caps) & aliases.selection_eligible.astype(bool)
    ].copy()
    candidate_ids = tuple(selected_aliases.canonical_candidate_id.drop_duplicates())
    all_candidate_ids = tuple(weights.candidate_id.drop_duplicates())
    if not candidate_ids:
        raise ValueError(f"No selection-eligible candidates for caps {caps}")
    if not set(candidate_ids).issubset(set(weights.candidate_id)):
        raise ValueError("Candidate alias map references an unknown canonical training candidate")
    candidate_weights_sha256 = file_sha256(args.candidate_weights)
    candidate_aliases_sha256 = file_sha256(args.candidate_aliases)
    results = materialize_rows(
        experiment_root=args.experiment_root,
        candidate_ids=candidate_ids,
        all_candidate_ids=all_candidate_ids,
        candidate_weights_sha256=candidate_weights_sha256,
        candidate_aliases_sha256=candidate_aliases_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
    )
    summary = summarize(results)
    alias_results, selected = select_aliases(selected_aliases, summary, caps)
    payload = selected_manifest(
        results=results,
        alias_results=alias_results,
        selected=selected,
        candidate_weights_sha256=candidate_weights_sha256,
        candidate_aliases_sha256=candidate_aliases_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
    )
    payload_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_uri_exact(args.selected_manifest_uri, payload_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "candidate_boundary_results.csv", index=False)
    summary.to_csv(args.output_dir / "canonical_candidate_summary.csv", index=False)
    alias_results.to_csv(args.output_dir / "alias_results.csv", index=False)
    selected.to_csv(args.output_dir / "selected_candidates.csv", index=False)
    (args.output_dir / "selected_prefixes.json").write_bytes(payload_bytes)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
