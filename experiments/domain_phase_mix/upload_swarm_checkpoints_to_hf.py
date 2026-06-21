# /// script
# dependencies = [
#   "fsspec>=2026.1.0",
#   "gcsfs>=2026.1.0",
#   "huggingface_hub>=0.36.0",
#   "pandas>=2.2.0",
#   "tqdm>=4.66.0",
# ]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload selected data-mixing swarm HF checkpoints to a private HF dataset repo.

The script is deliberately restartable. It builds a local CSV manifest, creates
the target dataset repo if needed, and uploads one checkpoint directory at a
time from GCS through a temporary local staging directory. Existing remote files
are skipped when the full expected checkpoint subtree is already present.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Calvin-Xu/marin-data-mixing-swarm-checkpoints"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "swarm_checkpoint_hf_upload_20260620"
)
TWO_PHASE_MANY_CSV = Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "two_phase_many.csv"
PARITY_300M_CSV = (
    Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "qsplit240_300m_6b_completed_vs_60m.csv"
)
PCTRL_300M_CANDIDATE_CSV = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "metric_registry"
    / "proportional_controllability_300m"
    / "proportional_controllability_eval_candidates.csv"
)
PCTRL_60M_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_pctrl60"
PCTRL_60M_GCS_ROOT = f"gs://marin-us-east5/checkpoints/{PCTRL_60M_SOURCE_EXPERIMENT}"
HF_REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer_config.json")
DEFAULT_UPLOAD_ATTEMPTS = 6
DEFAULT_UPLOAD_INITIAL_BACKOFF_SECONDS = 30.0
MAX_UPLOAD_BACKOFF_SECONDS = 300.0
PCTRL_DOMAIN_COUNT = 39
PCTRL_60M_EXPECTED_STEP = 4576


@dataclass(frozen=True)
class CheckpointUploadRow:
    """One checkpoint subtree to upload."""

    panel: str
    scale: str
    run_name: str
    source_experiment: str
    checkpoint_root: str
    checkpoint_uri: str
    expected_checkpoint_step: int
    path_in_repo: str
    metadata_json: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument(
        "--input-manifest",
        help=(
            "Read a prebuilt checkpoint upload manifest from a local path or GCS URI instead of rebuilding it from "
            "local registry CSVs. Use this for Iris jobs because gitignored local CSV inputs are not bundled."
        ),
    )
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument(
        "--upload-attempts",
        type=int,
        default=DEFAULT_UPLOAD_ATTEMPTS,
        help="Maximum per-checkpoint HF upload attempts for retryable commit/LFS failures.",
    )
    parser.add_argument(
        "--upload-initial-backoff-seconds",
        type=float,
        default=DEFAULT_UPLOAD_INITIAL_BACKOFF_SECONDS,
        help="Initial exponential backoff for retryable HF upload failures.",
    )
    parser.add_argument(
        "--skip-remote-verification",
        action="store_true",
        help="Skip the read-only HF tree/LFS size verification pass after upload.",
    )
    parser.add_argument(
        "--only-panel",
        action="append",
        choices=(
            "parity_60m_1p2b",
            "parity_300m_6b",
            "proportional_controllability_300m_6b",
            "proportional_controllability_60m_1p2b",
        ),
        help="Restrict to a panel. Can be passed multiple times.",
    )
    return parser.parse_args()


def _resolve_token(token_arg: str | None, *, required: bool) -> str | None:
    token = token_arg or os.environ.get("HF_TOKEN")
    if required and not token:
        raise RuntimeError("HF token required: pass --token or set HF_TOKEN.")
    return token


def _checkpoint_uri(checkpoint_root: str, expected_step: int) -> str:
    return f"{checkpoint_root.rstrip('/')}/hf/step-{expected_step}"


def _repo_checkpoint_path(panel: str, scale: str, run_name: str, expected_step: int) -> str:
    safe_run_name = run_name.replace("/", "__")
    return f"checkpoints/{scale}/{panel}/{safe_run_name}/hf/step-{expected_step}"


def _source_experiment_from_root(checkpoint_root: str) -> str:
    marker = "/checkpoints/"
    if marker not in checkpoint_root:
        return ""
    tail = checkpoint_root.split(marker, 1)[1]
    parts = tail.split("/")
    if len(parts) < 4:
        return ""
    return "/".join(parts[:3])


def _metadata(**values: object) -> str:
    return json.dumps(values, sort_keys=True)


def _build_60m_parity_rows() -> list[CheckpointUploadRow]:
    frame = pd.read_csv(TWO_PHASE_MANY_CSV, usecols=["wandb_run_id", "source_experiment", "run_name", "run_id"])
    rows: list[CheckpointUploadRow] = []
    for row in frame.sort_values("run_id").itertuples(index=False):
        checkpoint_root = f"gs://marin-us-east5/checkpoints/{row.source_experiment}/{row.wandb_run_id}"
        expected_step = 4576
        rows.append(
            CheckpointUploadRow(
                panel="parity",
                scale="60m_1p2b",
                run_name=str(row.run_name),
                source_experiment=str(row.source_experiment),
                checkpoint_root=checkpoint_root,
                checkpoint_uri=_checkpoint_uri(checkpoint_root, expected_step),
                expected_checkpoint_step=expected_step,
                path_in_repo=_repo_checkpoint_path("parity", "60m_1p2b", str(row.run_name), expected_step),
                metadata_json=_metadata(run_id=int(row.run_id), wandb_run_id=str(row.wandb_run_id)),
            )
        )
    return rows


def _read_manifest(path: str) -> list[CheckpointUploadRow]:
    frame = pd.read_csv(path)
    expected_columns = set(CheckpointUploadRow.__dataclass_fields__)
    actual_columns = set(frame.columns)
    missing = sorted(expected_columns - actual_columns)
    extra = sorted(actual_columns - expected_columns)
    if missing or extra:
        raise ValueError(f"Invalid checkpoint upload manifest columns: missing={missing}, extra={extra}")
    rows = [
        CheckpointUploadRow(
            panel=str(row.panel),
            scale=str(row.scale),
            run_name=str(row.run_name),
            source_experiment=str(row.source_experiment),
            checkpoint_root=str(row.checkpoint_root),
            checkpoint_uri=str(row.checkpoint_uri),
            expected_checkpoint_step=int(row.expected_checkpoint_step),
            path_in_repo=str(row.path_in_repo),
            metadata_json=str(row.metadata_json),
        )
        for row in frame.itertuples(index=False)
    ]
    _validate_manifest(rows)
    return rows


def _east5_checkpoint_root(checkpoint_root: str) -> str:
    if checkpoint_root.startswith("gs://marin-us-central"):
        return checkpoint_root.replace("gs://marin-us-central1/", "gs://marin-us-east5/", 1).replace(
            "gs://marin-us-central2/", "gs://marin-us-east5/", 1
        )
    return checkpoint_root


def _build_300m_parity_rows() -> list[CheckpointUploadRow]:
    frame = pd.read_csv(PARITY_300M_CSV)
    rows: list[CheckpointUploadRow] = []
    for row in frame.sort_values("run_name").itertuples(index=False):
        checkpoint_root = _east5_checkpoint_root(str(row.checkpoint_root))
        expected_step = 22887
        rows.append(
            CheckpointUploadRow(
                panel="parity",
                scale="300m_6b",
                run_name=str(row.run_name),
                source_experiment=_source_experiment_from_root(checkpoint_root),
                checkpoint_root=checkpoint_root,
                checkpoint_uri=_checkpoint_uri(checkpoint_root, expected_step),
                expected_checkpoint_step=expected_step,
                path_in_repo=_repo_checkpoint_path("parity", "300m_6b", str(row.run_name), expected_step),
                metadata_json=_metadata(
                    bpb_300m_6b=float(row.bpb_300m_6b),
                    bpb_60m=float(row.bpb_60m),
                    rank_300m_6b=int(row.rank_300m_6b),
                    rank_60m_within_completed=int(row.rank_60m_within_completed),
                ),
            )
        )
    return rows


def _build_300m_controllability_rows() -> list[CheckpointUploadRow]:
    if not PCTRL_300M_CANDIDATE_CSV.exists():
        raise FileNotFoundError(f"Missing 300M controllability candidate CSV: {PCTRL_300M_CANDIDATE_CSV}")
    frame = pd.read_csv(PCTRL_300M_CANDIDATE_CSV)
    if len(frame) != 117:
        raise ValueError(f"Expected 117 300M controllability rows, found {len(frame)}")
    rows: list[CheckpointUploadRow] = []
    for row in frame.sort_values("run_name").itertuples(index=False):
        expected_step = int(row.expected_checkpoint_step)
        rows.append(
            CheckpointUploadRow(
                panel="proportional_controllability",
                scale="300m_6b",
                run_name=str(row.run_name),
                source_experiment=str(row.source_experiment),
                checkpoint_root=str(row.checkpoint_root),
                checkpoint_uri=_checkpoint_uri(str(row.checkpoint_root), expected_step),
                expected_checkpoint_step=expected_step,
                path_in_repo=_repo_checkpoint_path(
                    "proportional_controllability", "300m_6b", str(row.run_name), expected_step
                ),
                metadata_json=_metadata(
                    intervention_id=str(row.intervention_id),
                    intervention_type=str(row.intervention_type),
                    tv_distance=float(row.tv_distance),
                ),
            )
        )
    return rows


def _expected_60m_controllability_run_names() -> list[str]:
    """Return the compact 117-row 60M controllability panel names."""
    deletion_names = [f"p60_del_{index:02d}" for index in range(PCTRL_DOMAIN_COUNT)]
    tilt_names = [f"p60_tilt_{index:02d}_{sign}" for index in range(PCTRL_DOMAIN_COUNT) for sign in ("m", "p")]
    return [*deletion_names, *tilt_names]


def _discover_60m_controllability_checkpoint_roots() -> dict[str, str]:
    """Discover actual hash-suffixed 60M controllability checkpoint roots."""
    fs, _, _ = fsspec.get_fs_token_paths(PCTRL_60M_GCS_ROOT)
    prefix = PCTRL_60M_GCS_ROOT.removeprefix("gs://").rstrip("/")
    config_pattern = f"{prefix}/*/hf/step-{PCTRL_60M_EXPECTED_STEP}/config.json"
    discovered = fs.glob(config_pattern)
    expected_names = set(_expected_60m_controllability_run_names())
    roots: dict[str, str] = {}
    for config_path in discovered:
        path = str(config_path)
        if not path.startswith("gs://"):
            path = f"gs://{path}"
        checkpoint_root = path.removesuffix(f"/hf/step-{PCTRL_60M_EXPECTED_STEP}/config.json")
        run_leaf = checkpoint_root.rstrip("/").rsplit("/", 1)[-1]
        matched = [name for name in expected_names if run_leaf.startswith(f"{name}-")]
        if len(matched) != 1:
            raise ValueError(f"Could not match 60M controllability checkpoint leaf {run_leaf!r}")
        run_name = matched[0]
        if run_name in roots:
            raise ValueError(
                f"Duplicate checkpoint roots discovered for {run_name}: {roots[run_name]}, {checkpoint_root}"
            )
        roots[run_name] = checkpoint_root

    missing = sorted(expected_names - set(roots))
    extra = sorted(set(roots) - expected_names)
    if missing or extra:
        raise ValueError(f"60M controllability checkpoint discovery mismatch: missing={missing}, extra={extra}")
    return roots


def _metadata_for_60m_controllability_run(run_name: str) -> dict[str, object]:
    """Return compact intervention metadata from the 60M controllability run name."""
    if run_name.startswith("p60_del_"):
        return {
            "intervention_type": "domain_deletion",
            "intervention_index": int(run_name.removeprefix("p60_del_")),
        }
    if run_name.startswith("p60_tilt_"):
        _, _, index, sign = run_name.split("_")
        return {
            "intervention_type": "central_log_tilt",
            "intervention_index": int(index),
            "tilt_sign": {"m": "minus", "p": "plus"}[sign],
        }
    raise ValueError(f"Unknown 60M controllability run name: {run_name}")


def _build_60m_controllability_rows() -> list[CheckpointUploadRow]:
    roots = _discover_60m_controllability_checkpoint_roots()
    rows: list[CheckpointUploadRow] = []
    for run_name in sorted(_expected_60m_controllability_run_names()):
        checkpoint_root = roots[run_name]
        rows.append(
            CheckpointUploadRow(
                panel="proportional_controllability",
                scale="60m_1p2b",
                run_name=run_name,
                source_experiment=PCTRL_60M_SOURCE_EXPERIMENT,
                checkpoint_root=checkpoint_root,
                checkpoint_uri=_checkpoint_uri(checkpoint_root, PCTRL_60M_EXPECTED_STEP),
                expected_checkpoint_step=PCTRL_60M_EXPECTED_STEP,
                path_in_repo=_repo_checkpoint_path(
                    "proportional_controllability", "60m_1p2b", run_name, PCTRL_60M_EXPECTED_STEP
                ),
                metadata_json=_metadata(**_metadata_for_60m_controllability_run(run_name)),
            )
        )
    return rows


def build_manifest(only_panels: Iterable[str] | None) -> list[CheckpointUploadRow]:
    """Build upload rows for the requested panels."""
    requested = set(only_panels or ())
    builders = {
        "parity_60m_1p2b": _build_60m_parity_rows,
        "parity_300m_6b": _build_300m_parity_rows,
        "proportional_controllability_300m_6b": _build_300m_controllability_rows,
        "proportional_controllability_60m_1p2b": _build_60m_controllability_rows,
    }
    if not requested:
        requested = set(builders)
    unknown = requested - set(builders)
    if unknown:
        raise ValueError(f"Unknown panels: {sorted(unknown)}")
    rows: list[CheckpointUploadRow] = []
    for panel_key, builder in builders.items():
        if panel_key in requested:
            rows.extend(builder())
    _validate_manifest(rows)
    return rows


def _validate_manifest(rows: list[CheckpointUploadRow]) -> None:
    if not rows:
        raise ValueError("No upload rows selected")
    repo_paths = [row.path_in_repo for row in rows]
    if len(set(repo_paths)) != len(repo_paths):
        raise ValueError("Duplicate HF repo checkpoint paths")
    checkpoint_uris = [row.checkpoint_uri for row in rows]
    if len(set(checkpoint_uris)) != len(checkpoint_uris):
        raise ValueError("Duplicate source checkpoint URIs")
    central_roots = [row.checkpoint_root for row in rows if row.checkpoint_root.startswith("gs://marin-us-central")]
    if central_roots:
        raise ValueError(f"Central-region checkpoint roots are not allowed: {central_roots[:5]}")


def _write_manifest(rows: list[CheckpointUploadRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _gcs_file_sizes(checkpoint_uri: str) -> dict[str, int]:
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_uri)
    prefix = checkpoint_uri.removeprefix("gs://").rstrip("/")
    raw_infos = fs.find(prefix, detail=True)
    files = {
        path if str(path).startswith("gs://") else f"gs://{path}": int(info["size"]) for path, info in raw_infos.items()
    }
    if not files:
        raise FileNotFoundError(f"No files found under {checkpoint_uri}")
    rels = {Path(file.removeprefix(f"{checkpoint_uri.rstrip('/')}/")).as_posix() for file in files}
    missing_required = [name for name in HF_REQUIRED_FILES if name not in rels]
    if missing_required:
        raise FileNotFoundError(f"{checkpoint_uri} is missing required files: {missing_required}")
    return files


def _copy_checkpoint_to_stage(row: CheckpointUploadRow, file_sizes: dict[str, int], stage_dir: Path) -> Path:
    root = row.checkpoint_uri.rstrip("/")
    local_root = stage_dir / row.run_name
    for source, expected_size in file_sizes.items():
        rel = source.removeprefix(f"{root}/")
        if rel == source:
            raise ValueError(f"File {source} is not under checkpoint root {root}")
        target = local_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with fsspec.open(source, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        actual_size = target.stat().st_size
        if actual_size != expected_size:
            raise OSError(f"Staged file size mismatch for {source}: expected {expected_size}, found {actual_size}")
    return local_root


def _remote_files(api: HfApi, repo_id: str) -> set[str]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(float(attempt * 5))
    assert last_error is not None
    raise RuntimeError(f"Failed to list remote files for {repo_id}") from last_error


def _expected_repo_files(row: CheckpointUploadRow, files: list[str]) -> set[str]:
    root = row.checkpoint_uri.rstrip("/")
    return {f"{row.path_in_repo}/{source.removeprefix(f'{root}/')}" for source in files}


def _retryable_hf_error(exc: HfHubHTTPError) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403, 404}:
        return False
    if status_code in {408, 409, 429}:
        return True
    if status_code is not None and status_code >= 500:
        return True
    message = str(exc).lower()
    return status_code == 400 and "lfs pointer" in message


def _backoff_seconds(attempt: int, initial_backoff_seconds: float) -> float:
    return min(MAX_UPLOAD_BACKOFF_SECONDS, initial_backoff_seconds * (2 ** (attempt - 1)))


def _upload_folder_with_retry(
    api: HfApi,
    *,
    row: CheckpointUploadRow,
    local_root: Path,
    repo_id: str,
    expected_remote: set[str],
    upload_attempts: int,
    upload_initial_backoff_seconds: float,
) -> tuple[str, set[str]]:
    if upload_attempts < 1:
        raise ValueError("--upload-attempts must be >= 1")
    for attempt in range(1, upload_attempts + 1):
        try:
            logger.info(
                "Uploading %s/%s/%s to %s (attempt %d/%d)",
                row.scale,
                row.panel,
                row.run_name,
                row.path_in_repo,
                attempt,
                upload_attempts,
            )
            api.upload_folder(
                folder_path=str(local_root),
                path_in_repo=row.path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {row.scale}/{row.panel}/{row.run_name}",
            )
            return "uploaded", expected_remote
        except HfHubHTTPError as exc:
            fresh_remote = _remote_files(api, repo_id)
            if expected_remote.issubset(fresh_remote):
                logger.warning(
                    "Upload for %s/%s/%s raised %s but all expected remote files are present; continuing.",
                    row.scale,
                    row.panel,
                    row.run_name,
                    exc.__class__.__name__,
                )
                return "uploaded_after_error", fresh_remote
            if not _retryable_hf_error(exc) or attempt == upload_attempts:
                raise RuntimeError(
                    f"HF upload failed for {row.scale}/{row.panel}/{row.run_name} at {row.path_in_repo} "
                    f"after {attempt}/{upload_attempts} attempts"
                ) from exc
            sleep_seconds = _backoff_seconds(attempt, upload_initial_backoff_seconds)
            logger.warning(
                "Retryable HF upload error for %s/%s/%s on attempt %d/%d; sleeping %.1fs before retry: %s",
                row.scale,
                row.panel,
                row.run_name,
                attempt,
                upload_attempts,
                sleep_seconds,
                exc,
            )
            time.sleep(sleep_seconds)

    raise AssertionError("unreachable")


def _remote_file_sizes(api: HfApi, repo_id: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for entry in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True, expand=True):
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if path is None or size is None:
            continue
        sizes[str(path)] = int(size)
    return sizes


def _verify_remote_sizes(api: HfApi, repo_id: str, rows: list[CheckpointUploadRow], output_dir: Path) -> None:
    logger.info("Verifying HF remote file sizes for %d checkpoint rows", len(rows))
    remote_sizes = _remote_file_sizes(api, repo_id)
    missing: list[dict[str, object]] = []
    mismatched: list[dict[str, object]] = []
    checked_count = 0
    for row in tqdm(rows, desc="verify remote files"):
        source_sizes = _gcs_file_sizes(row.checkpoint_uri)
        root = row.checkpoint_uri.rstrip("/")
        for source, expected_size in source_sizes.items():
            rel = source.removeprefix(f"{root}/")
            remote_path = f"{row.path_in_repo}/{rel}"
            remote_size = remote_sizes.get(remote_path)
            checked_count += 1
            if remote_size is None:
                missing.append({"run_name": row.run_name, "path": remote_path, "expected_size": expected_size})
            elif remote_size != expected_size:
                mismatched.append(
                    {
                        "run_name": row.run_name,
                        "path": remote_path,
                        "expected_size": expected_size,
                        "remote_size": remote_size,
                    }
                )

    report = {
        "repo_id": repo_id,
        "row_count": len(rows),
        "checked_file_count": checked_count,
        "missing_count": len(missing),
        "mismatched_count": len(mismatched),
        "missing": missing[:100],
        "mismatched": mismatched[:100],
    }
    report_path = output_dir / "remote_size_verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if missing or mismatched:
        raise RuntimeError(
            f"Remote size verification failed: missing={len(missing)}, mismatched={len(mismatched)}; "
            f"report={report_path}"
        )


def _upload_readme(api: HfApi, repo_id: str, rows: list[CheckpointUploadRow]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        counts[(row.scale, row.panel)] = counts.get((row.scale, row.panel), 0) + 1
    count_lines = "\n".join(f"- `{scale}/{panel}`: {count}" for (scale, panel), count in sorted(counts.items()))
    text = f"""\
---
private: true
tags:
  - marin
  - data-mixing
  - checkpoints
---

# Marin Data-Mixing Swarm Checkpoints

Private checkpoint export for downstream evaluation on non-Marin environments.

## Panels

{count_lines}

Each row is indexed in `manifest/checkpoint_upload_manifest.csv`. Checkpoints
are stored under `checkpoints/<scale>/<panel>/<run_name>/hf/step-<step>/`.
"""
    api.upload_file(
        path_or_fileobj=io.BytesIO(text.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update checkpoint export README",
    )


def upload(
    rows: list[CheckpointUploadRow],
    *,
    repo_id: str,
    token: str,
    private: bool,
    output_dir: Path,
    upload_attempts: int,
    upload_initial_backoff_seconds: float,
    verify_remote: bool,
) -> None:
    """Upload selected checkpoint rows to HF Hub."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    manifest_path = output_dir / "checkpoint_upload_manifest.csv"
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="manifest/checkpoint_upload_manifest.csv",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update checkpoint upload manifest",
    )
    _upload_readme(api, repo_id, rows)

    progress_path = output_dir / "upload_progress.jsonl"
    known_remote = _remote_files(api, repo_id)
    scratch_dir = Path(tempfile.mkdtemp(prefix="swarm-checkpoint-hf-upload-"))
    try:
        for row in tqdm(rows, desc="upload checkpoints"):
            file_sizes = _gcs_file_sizes(row.checkpoint_uri)
            files = list(file_sizes)
            expected_remote = _expected_repo_files(row, files)
            if expected_remote.issubset(known_remote):
                status = "skipped_existing"
            else:
                local_root = _copy_checkpoint_to_stage(row, file_sizes, scratch_dir)
                status, fresh_remote = _upload_folder_with_retry(
                    api,
                    row=row,
                    local_root=local_root,
                    repo_id=repo_id,
                    expected_remote=expected_remote,
                    upload_attempts=upload_attempts,
                    upload_initial_backoff_seconds=upload_initial_backoff_seconds,
                )
                shutil.rmtree(local_root)
                known_remote.update(fresh_remote)
            with progress_path.open("a") as handle:
                handle.write(json.dumps({**asdict(row), "status": status}, sort_keys=True) + "\n")
        if verify_remote:
            _verify_remote_sizes(api, repo_id, rows, output_dir)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    rows = _read_manifest(args.input_manifest) if args.input_manifest else build_manifest(args.only_panel)
    if args.max_checkpoints is not None:
        rows = rows[: args.max_checkpoints]
    output_dir = args.output_dir
    manifest_path = output_dir / "checkpoint_upload_manifest.csv"
    _write_manifest(rows, manifest_path)

    counts_frame = (
        pd.DataFrame([asdict(row) for row in rows]).groupby(["scale", "panel"]).size().reset_index(name="count")
    )
    summary = {
        "repo_id": args.repo_id,
        "row_count": len(rows),
        "counts": counts_frame.to_dict(orient="records"),
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %d upload rows to %s", len(rows), manifest_path)
    if args.dry_run or args.write_manifest_only:
        logger.info("Dry run/write-manifest-only requested; not uploading.")
        return

    token = _resolve_token(args.token, required=True)
    assert token is not None
    upload(
        rows,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        output_dir=output_dir,
        upload_attempts=args.upload_attempts,
        upload_initial_backoff_seconds=args.upload_initial_backoff_seconds,
        verify_remote=not args.skip_remote_verification,
    )


if __name__ == "__main__":
    main()
