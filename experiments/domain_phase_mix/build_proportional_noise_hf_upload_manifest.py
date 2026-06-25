# /// script
# dependencies = [
#   "fsspec>=2026.1.0",
#   "gcsfs>=2026.1.0",
#   "huggingface_hub>=0.36.0",
#   "pandas>=2.2.0",
# ]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an HF upload manifest for proportional-noise checkpoints.

The main swarm checkpoint upload manifest intentionally covered parity and
controllability panels. This script builds the missing repeat-proportional panel
needed to run OLMoBaseEval Easy on noise-baseline checkpoints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import pandas as pd
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "Calvin-Xu/marin-data-mixing-swarm-checkpoints"
DEFAULT_RUN_REGISTRY = (
    Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "metric_registry" / "runs.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "proportional_noise_hf_upload_20260623"
)
HF_REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer_config.json")
DEFAULT_PANEL = "proportional_noise"
EXPECTED_STEPS = {
    "60m_1p2b": 4576,
    "300m_6b": 22887,
}


@dataclass(frozen=True)
class UploadRow:
    """One row for upload_swarm_checkpoints_to_hf.py --input-manifest."""

    panel: str
    scale: str
    run_name: str
    source_experiment: str
    checkpoint_root: str
    checkpoint_uri: str
    expected_checkpoint_step: int
    path_in_repo: str
    metadata_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registry", type=Path, default=DEFAULT_RUN_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument(
        "--skip-hf-check",
        action="store_true",
        help="Skip private HF repo listing. GCS readiness is still checked.",
    )
    return parser.parse_args()


def source_experiment_from_root(checkpoint_root: str) -> str:
    marker = "/checkpoints/"
    if marker not in checkpoint_root:
        return ""
    tail = checkpoint_root.split(marker, 1)[1]
    parts = tail.split("/")
    if len(parts) < 4:
        return ""
    return "/".join(parts[:3])


def expected_step(row: pd.Series) -> int:
    for column in ("target_final_checkpoint_step", "expected_checkpoint_step"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return int(float(value))
    scale = str(row["scale"])
    if scale not in EXPECTED_STEPS:
        raise ValueError(f"Missing expected step for scale={scale!r}")
    return EXPECTED_STEPS[scale]


def checkpoint_uri(checkpoint_root: str, step: int) -> str:
    return f"{checkpoint_root.rstrip('/')}/hf/step-{step}"


def safe_run_name(run_name: str) -> str:
    return run_name.replace("/", "__")


def path_in_repo(*, scale: str, panel: str, run_name: str, step: int) -> str:
    return f"checkpoints/{scale}/{panel}/{safe_run_name(run_name)}/hf/step-{step}"


def metadata(row: pd.Series) -> str:
    payload = {
        "row_kind": str(row.get("row_kind", "")),
        "run_id": int(float(row["run_id"])) if pd.notna(row.get("run_id")) else None,
        "wandb_run_id": str(row.get("wandb_run_id", "")),
        "data_seed": _optional_int(row.get("data_seed")),
        "trainer_seed": _optional_int(row.get("trainer_seed")),
        "simulated_epoch_subset_seed": _optional_int(row.get("simulated_epoch_subset_seed")),
    }
    return json.dumps(payload, sort_keys=True)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return int(float(value))


def gcs_required_status(uri: str) -> tuple[bool, list[str]]:
    fs, _, paths = fsspec.get_fs_token_paths(uri)
    root = paths[0].rstrip("/")
    missing: list[str] = []
    for filename in HF_REQUIRED_FILES:
        if not fs.exists(f"{root}/{filename}"):
            missing.append(filename)
    return not missing, missing


def remote_files(repo_id: str) -> set[str]:
    return set(HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset"))


def build_rows(frame: pd.DataFrame, *, panel: str) -> list[UploadRow]:
    rows: list[UploadRow] = []
    for _, row in frame.sort_values(["scale", "run_id"]).iterrows():
        step = expected_step(row)
        root = str(row["checkpoint_root"])
        run_name = str(row["run_name"])
        scale = str(row["scale"])
        uri = checkpoint_uri(root, step)
        rows.append(
            UploadRow(
                panel=panel,
                scale=scale,
                run_name=run_name,
                source_experiment=source_experiment_from_root(root),
                checkpoint_root=root,
                checkpoint_uri=uri,
                expected_checkpoint_step=step,
                path_in_repo=path_in_repo(scale=scale, panel=panel, run_name=run_name, step=step),
                metadata_json=metadata(row),
            )
        )
    return rows


def write_manifest(rows: list[UploadRow], path: Path) -> None:
    pd.DataFrame([asdict(row) for row in rows]).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.run_registry, low_memory=False)
    noise = frame[frame["row_kind"].eq("noise_variable_subset_proportional")].copy()
    if noise.empty:
        raise ValueError("No proportional-noise rows found in run registry")
    if noise["checkpoint_root"].isna().any():
        missing = noise[noise["checkpoint_root"].isna()]["run_name"].tolist()
        raise ValueError(f"Missing checkpoint_root for rows: {missing}")

    rows = build_rows(noise, panel=args.panel)
    seen_paths = [row.path_in_repo for row in rows]
    if len(set(seen_paths)) != len(seen_paths):
        raise ValueError("Duplicate HF paths in generated manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    remote = set() if args.skip_hf_check else remote_files(args.repo_id)
    status_rows: list[dict[str, object]] = []
    ready_rows: list[UploadRow] = []
    for row in rows:
        gcs_ready, missing = gcs_required_status(row.checkpoint_uri)
        remote_required = [f"{row.path_in_repo}/{filename}" for filename in HF_REQUIRED_FILES]
        remote_ready = None if args.skip_hf_check else all(path in remote for path in remote_required)
        status_rows.append(
            {
                **asdict(row),
                "gcs_ready": gcs_ready,
                "gcs_missing_required": ",".join(missing),
                "remote_required_ready": remote_ready,
            }
        )
        if gcs_ready:
            ready_rows.append(row)

    manifest_path = args.output_dir / "proportional_noise_checkpoint_upload_manifest.csv"
    readiness_path = args.output_dir / "proportional_noise_checkpoint_readiness.csv"
    summary_path = args.output_dir / "summary.json"
    write_manifest(ready_rows, manifest_path)
    pd.DataFrame(status_rows).to_csv(readiness_path, index=False)

    status_frame = pd.DataFrame(status_rows)
    summary = {
        "repo_id": args.repo_id,
        "panel": args.panel,
        "total_noise_rows": len(status_rows),
        "gcs_ready_rows": int(status_frame["gcs_ready"].sum()),
        "gcs_missing_rows": int((~status_frame["gcs_ready"]).sum()),
        "hf_check_skipped": args.skip_hf_check,
        "remote_required_ready_rows": (
            None if args.skip_hf_check else int(status_frame["remote_required_ready"].fillna(False).sum())
        ),
        "manifest_rows": len(ready_rows),
        "by_scale": status_frame.groupby("scale").size().to_dict(),
        "gcs_ready_by_scale": status_frame.groupby("scale")["gcs_ready"].sum().astype(int).to_dict(),
        "remote_required_ready_by_scale": (
            None
            if args.skip_hf_check
            else status_frame.groupby("scale")["remote_required_ready"].sum().astype(int).to_dict()
        ),
        "manifest": str(manifest_path),
        "readiness": str(readiness_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
