# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fsspec>=2026.1.0",
#   "gcsfs>=2026.1.0",
#   "huggingface_hub>=1.2.0",
# ]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and run Stanford SC OLMoBaseEval jobs for uploaded swarm checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import fsspec
from huggingface_hub import HfApi, snapshot_download

DEFAULT_HF_REPO_ID = "Calvin-Xu/marin-data-mixing-swarm-checkpoints"
DEFAULT_WORK_DIR = Path("/juice4/scr4/pinlinxu/olmo_eval_canary")
DEFAULT_OLMO_EVAL_DIR = DEFAULT_WORK_DIR / "OLMo-Eval"
DEFAULT_WRITEBACK_SCRIPT = DEFAULT_WORK_DIR / "scripts" / "write_olmo_eval_wandb.py"
DEFAULT_KEY_PREFIX = "olmo_base_eval/easy_bpb"
DEFAULT_EXPERIMENT_GROUP = "olmo_base_eval_sc_716"
DEFAULT_TASK_PREP_WORKERS = 1
OLMO_EVAL_TASK_PREP_WORKERS = "OLMO_EVAL_TASK_PREP_WORKERS"
OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT = "OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT"
CHECKPOINT_REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer_config.json")
BASIC_SKILLS_SUBTASKS = (
    "arithmetic",
    "coding",
    "common_knowledge",
    "logical_reasoning",
    "string_operations",
    "pattern",
)
FULL_SUITE_TASKS = (
    "olmobase:easy:qa:bpb",
    "olmobase:easy:math:bpb",
    "olmobase:easy:code:bpb",
)
SMOKE_TASKS = ("arc_easy:olmo3base:bpb",)
UPLOAD_MANIFEST_COLUMNS = {
    "panel",
    "scale",
    "run_name",
    "source_experiment",
    "checkpoint_root",
    "checkpoint_uri",
    "expected_checkpoint_step",
    "path_in_repo",
    "metadata_json",
}
SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class EvalManifestRow:
    """One HF checkpoint to evaluate and mirror back to W&B."""

    index: int
    scale: str
    panel: str
    run_name: str
    source_experiment: str
    checkpoint_root: str
    expected_checkpoint_step: int
    hf_repo_id: str
    hf_checkpoint_path: str
    wandb_run_id: str
    output_name: str
    metadata_json: str


def safe_segment(value: str) -> str:
    """Return a stable filesystem-safe segment."""
    cleaned = SAFE_SEGMENT_RE.sub("_", value.strip()).strip("_.-")
    return cleaned or "unknown"


def read_text(path: str) -> str:
    """Read local or fsspec-supported text files."""
    if path.startswith("gs://"):
        with fsspec.open(path, "rt") as handle:
            return handle.read()
    return Path(path).read_text()


def write_csv(rows: list[EvalManifestRow], path: Path) -> None:
    """Write the SC eval manifest."""
    if not rows:
        raise ValueError("No manifest rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_eval_manifest(path: Path) -> list[EvalManifestRow]:
    """Read an SC eval manifest."""
    with path.open(newline="") as handle:
        rows = [
            EvalManifestRow(
                index=int(row["index"]),
                scale=row["scale"],
                panel=row["panel"],
                run_name=row["run_name"],
                source_experiment=row["source_experiment"],
                checkpoint_root=row["checkpoint_root"],
                expected_checkpoint_step=int(row["expected_checkpoint_step"]),
                hf_repo_id=row["hf_repo_id"],
                hf_checkpoint_path=row["hf_checkpoint_path"],
                wandb_run_id=row["wandb_run_id"],
                output_name=row["output_name"],
                metadata_json=row["metadata_json"],
            )
            for row in csv.DictReader(handle)
        ]
    validate_eval_manifest(rows)
    return rows


def _wandb_run_id_from_upload_row(upload_row: dict[str, str]) -> str:
    metadata = json.loads(upload_row["metadata_json"])
    if isinstance(metadata, dict) and metadata.get("wandb_run_id"):
        return str(metadata["wandb_run_id"])
    checkpoint_root = upload_row["checkpoint_root"].rstrip("/")
    leaf = checkpoint_root.rsplit("/", 1)[-1]
    if not leaf:
        raise ValueError(f"Cannot derive W&B run id from checkpoint_root={checkpoint_root!r}")
    return leaf


def _output_name(index: int, upload_row: dict[str, str]) -> str:
    return safe_segment(f"{index:03d}_{upload_row['scale']}_{upload_row['panel']}_{upload_row['run_name']}")


def rows_from_upload_manifest(text: str, *, hf_repo_id: str, start_index: int) -> list[EvalManifestRow]:
    """Convert checkpoint-upload rows to OLMoBaseEval rows."""
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise ValueError("Upload manifest is empty")
    actual = set(reader.fieldnames)
    missing = sorted(UPLOAD_MANIFEST_COLUMNS - actual)
    if missing:
        raise ValueError(f"Upload manifest missing columns: {missing}")

    rows: list[EvalManifestRow] = []
    for offset, upload_row in enumerate(reader):
        index = start_index + offset
        rows.append(
            EvalManifestRow(
                index=index,
                scale=upload_row["scale"],
                panel=upload_row["panel"],
                run_name=upload_row["run_name"],
                source_experiment=upload_row["source_experiment"],
                checkpoint_root=upload_row["checkpoint_root"],
                expected_checkpoint_step=int(upload_row["expected_checkpoint_step"]),
                hf_repo_id=hf_repo_id,
                hf_checkpoint_path=upload_row["path_in_repo"],
                wandb_run_id=_wandb_run_id_from_upload_row(upload_row),
                output_name=_output_name(index, upload_row),
                metadata_json=upload_row["metadata_json"],
            )
        )
    return rows


def build_eval_manifest(input_manifests: list[str], *, hf_repo_id: str) -> list[EvalManifestRow]:
    """Build a concatenated SC eval manifest from checkpoint-upload manifests."""
    rows: list[EvalManifestRow] = []
    for manifest in input_manifests:
        rows.extend(rows_from_upload_manifest(read_text(manifest), hf_repo_id=hf_repo_id, start_index=len(rows)))
    validate_eval_manifest(rows)
    return rows


def validate_eval_manifest(rows: list[EvalManifestRow]) -> None:
    """Validate manifest invariants that make Slurm array indexing safe."""
    if not rows:
        raise ValueError("No eval rows")
    expected_indexes = list(range(len(rows)))
    actual_indexes = [row.index for row in rows]
    if actual_indexes != expected_indexes:
        raise ValueError("Eval manifest indexes must be contiguous and ordered from 0")
    for attr in ("output_name", "hf_checkpoint_path", "wandb_run_id"):
        values = [getattr(row, attr) for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {attr} values in eval manifest")
    central = [row.checkpoint_root for row in rows if row.checkpoint_root.startswith("gs://marin-us-central")]
    if central:
        raise ValueError(f"Central-region checkpoint roots are not allowed in this manifest: {central[:5]}")


def manifest_summary(rows: list[EvalManifestRow]) -> dict[str, object]:
    """Return compact manifest counts for validation and logging."""
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.scale}/{row.panel}"
        counts[key] = counts.get(key, 0) + 1
    return {"row_count": len(rows), "by_scale_panel": dict(sorted(counts.items()))}


def audit_hf_paths(rows: list[EvalManifestRow]) -> dict[str, object]:
    """Check whether each checkpoint path exists in the HF dataset repo."""
    api = HfApi()
    repo_ids = sorted({row.hf_repo_id for row in rows})
    repo_files: dict[str, set[str]] = {}
    for repo_id in repo_ids:
        repo_files[repo_id] = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    missing: list[str] = []
    for row in rows:
        required = f"{row.hf_checkpoint_path}/model.safetensors"
        if required not in repo_files[row.hf_repo_id]:
            missing.append(required)
    return {
        "row_count": len(rows),
        "missing_count": len(missing),
        "missing": missing[:100],
    }


def olmo_eval_git_sha(olmo_eval_dir: Path) -> str | None:
    """Return OLMo-Eval git SHA when available."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=olmo_eval_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def downloaded_model_root_for_row(row: EvalManifestRow, work_dir: Path) -> Path:
    """Return the per-row downloaded checkpoint root, rejecting unsafe names."""
    if not row.output_name or row.output_name in {".", ".."}:
        raise ValueError(f"Unsafe empty or relative output_name: {row.output_name!r}")
    if "/" in row.output_name or "\\" in row.output_name:
        raise ValueError(f"Unsafe output_name with path separator: {row.output_name!r}")
    download_root = (work_dir / "downloaded_models").resolve()
    model_root = (download_root / row.output_name).resolve()
    if model_root == download_root:
        raise ValueError(f"Unsafe output_name resolves to shared download root: {row.output_name!r}")
    try:
        model_root.relative_to(download_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe output_name escapes download root: {row.output_name!r}") from exc
    return model_root


def model_dir_for_row(row: EvalManifestRow, work_dir: Path) -> Path:
    """Return the local model path after snapshot download."""
    return downloaded_model_root_for_row(row, work_dir) / row.hf_checkpoint_path


def missing_checkpoint_files(model_dir: Path) -> list[str]:
    """Return required checkpoint files that are absent from a local model dir."""
    return [name for name in CHECKPOINT_REQUIRED_FILES if not (model_dir / name).is_file()]


def assert_checkpoint_complete(row: EvalManifestRow, work_dir: Path) -> Path:
    """Return the local checkpoint path, or raise if staging is incomplete."""
    model_dir = model_dir_for_row(row, work_dir)
    missing = missing_checkpoint_files(model_dir)
    if missing:
        raise FileNotFoundError(
            f"Checkpoint for {row.output_name} is missing {missing}: {model_dir}. "
            "Run stage-checkpoints before local-only evaluation."
        )
    return model_dir


def download_checkpoint(row: EvalManifestRow, work_dir: Path, *, revision: str | None = None) -> Path:
    """Download one checkpoint subtree from the HF dataset repo."""
    local_root = downloaded_model_root_for_row(row, work_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {}
    if revision is not None:
        kwargs["revision"] = revision
    snapshot_download(
        repo_id=row.hf_repo_id,
        repo_type="dataset",
        allow_patterns=[f"{row.hf_checkpoint_path}/**"],
        local_dir=local_root,
        **kwargs,
    )
    return assert_checkpoint_complete(row, work_dir)


def resolve_checkpoint(
    row: EvalManifestRow,
    work_dir: Path,
    *,
    checkpoint_mode: Literal["download", "local-only"],
    revision: str | None = None,
) -> Path:
    """Resolve the checkpoint for a run according to the selected access mode."""
    if checkpoint_mode == "local-only":
        return assert_checkpoint_complete(row, work_dir)
    if checkpoint_mode == "download":
        return download_checkpoint(row, work_dir, revision=revision)
    raise AssertionError(f"Unhandled checkpoint mode: {checkpoint_mode}")


def selected_rows(
    rows: list[EvalManifestRow],
    *,
    start_index: int | None,
    end_index: int | None,
    indexes: list[int] | None,
) -> list[EvalManifestRow]:
    """Return a bounded subset of manifest rows for staging or prewarming."""
    if indexes:
        by_index = {row.index: row for row in rows}
        missing = [index for index in indexes if index not in by_index]
        if missing:
            raise IndexError(f"Manifest indexes out of range: {missing}")
        return [by_index[index] for index in indexes]
    start = 0 if start_index is None else start_index
    end = len(rows) - 1 if end_index is None else end_index
    if start < 0 or end < start or end >= len(rows):
        raise IndexError(f"Invalid row range {start}..{end} for {len(rows)} rows")
    return rows[start : end + 1]


def stage_checkpoints(args: argparse.Namespace) -> None:
    """Stage manifest checkpoints into the local SC work dir before array launch."""
    rows = selected_rows(
        read_eval_manifest(args.manifest),
        start_index=args.start_index,
        end_index=args.end_index,
        indexes=args.index,
    )
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    def stage_one(row: EvalManifestRow) -> dict[str, object]:
        staged_dir = model_dir_for_row(row, args.work_dir)
        if not missing_checkpoint_files(staged_dir):
            return {
                "index": row.index,
                "output_name": row.output_name,
                "model_dir": str(staged_dir),
                "status": "already_staged",
            }
        if args.only_missing is False:
            shutil.rmtree(downloaded_model_root_for_row(row, args.work_dir), ignore_errors=True)
        model_dir = download_checkpoint(row, args.work_dir, revision=args.hf_revision)
        return {
            "index": row.index,
            "output_name": row.output_name,
            "model_dir": str(model_dir),
            "status": "staged",
        }

    results: list[dict[str, object]] = []
    if args.max_workers == 1:
        for row in rows:
            result = stage_one(row)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_row = {executor.submit(stage_one, row): row for row in rows}
            for future in concurrent.futures.as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to stage checkpoint for index={row.index} {row.output_name}") from exc
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"status": "complete", "staged_count": len(results)}, sort_keys=True))


def clean_incomplete_dataset_cache(work_dir: Path, *, apply: bool) -> dict[str, object]:
    """Remove Hugging Face datasets cache directories left by interrupted builders."""
    datasets_cache = work_dir / "hf_home" / "datasets"
    incomplete_dirs = sorted(datasets_cache.glob("**/*.incomplete")) if datasets_cache.exists() else []
    removed: list[str] = []
    for path in incomplete_dirs:
        if apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        removed.append(str(path))
    return {
        "apply": apply,
        "datasets_cache": str(datasets_cache),
        "incomplete_count": len(incomplete_dirs),
        "incomplete_dirs": removed[:100],
    }


def clean_dataset_cache(args: argparse.Namespace) -> None:
    """CLI wrapper for removing interrupted datasets-cache builders."""
    print(json.dumps(clean_incomplete_dataset_cache(args.work_dir, apply=args.apply), indent=2, sort_keys=True))


def offline_cache_report(work_dir: Path) -> dict[str, object]:
    """Return a conservative readiness report for the shared HF cache root."""
    hf_home = work_dir / "hf_home"
    required_dirs = {
        "hf_home": hf_home,
        "hub": hf_home / "hub",
        "datasets": hf_home / "datasets",
        "modules": hf_home / "modules",
    }
    missing_dirs = [name for name, path in required_dirs.items() if not path.is_dir()]
    incomplete_dirs = sorted((hf_home / "datasets").glob("**/*.incomplete")) if (hf_home / "datasets").exists() else []
    basic_skills_root: Path | None = None
    basic_skills_missing: list[str] = []
    try:
        basic_skills_root = basic_skills_snapshot_root(work_dir)
    except FileNotFoundError as exc:
        basic_skills_missing.append(str(exc))
    return {
        "ready": not missing_dirs and not incomplete_dirs and not basic_skills_missing,
        "missing_dirs": missing_dirs,
        "incomplete_count": len(incomplete_dirs),
        "incomplete_dirs": [str(path) for path in incomplete_dirs[:100]],
        "basic_skills_root": None if basic_skills_root is None else str(basic_skills_root),
        "basic_skills_missing": basic_skills_missing,
    }


def verify_offline_cache(args: argparse.Namespace) -> None:
    """Fail fast unless the shared HF cache root is safe for offline fanout."""
    report = offline_cache_report(args.work_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        raise RuntimeError("HF offline cache is not ready for fanout")


def cache_environment(work_dir: Path) -> dict[str, str]:
    """Return HF cache environment variables shared by prewarm and array jobs."""
    hf_home = work_dir / "hf_home"
    return {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HF_MODULES_CACHE": str(hf_home / "modules"),
        "HF_ALLOW_CODE_EVAL": "1",
    }


def stable_eval_environment() -> dict[str, str]:
    """Return env vars that make OLMo-Eval task preparation fanout-safe."""
    return {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
        OLMO_EVAL_TASK_PREP_WORKERS: str(DEFAULT_TASK_PREP_WORKERS),
    }


def basic_skills_snapshot_root(work_dir: Path) -> Path:
    """Return the prewarmed basic-skills snapshot root, or raise with missing files."""
    hub_root = work_dir / "hf_home" / "hub" / "datasets--allenai--basic-skills"
    ref = hub_root / "refs" / "main"
    if not ref.is_file():
        raise FileNotFoundError(str(ref))
    snapshot = hub_root / "snapshots" / ref.read_text().strip()
    missing = [
        str(snapshot / subset / "validation.json")
        for subset in BASIC_SKILLS_SUBTASKS
        if not (snapshot / subset / "validation.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return snapshot


def ensure_basic_skills_snapshot(work_dir: Path) -> Path:
    """Populate and validate the hub snapshot layout needed by offline basic-skills."""
    snapshot_download(
        repo_id="allenai/basic-skills",
        repo_type="dataset",
        cache_dir=str(work_dir / "hf_home" / "hub"),
        allow_patterns=[f"{subset}/validation.json" for subset in BASIC_SKILLS_SUBTASKS],
    )
    return basic_skills_snapshot_root(work_dir)


def worker_environment(work_dir: Path) -> dict[str, str]:
    """Return an eval-worker environment with stable cache roots."""
    env = os.environ.copy()
    env.update(cache_environment(work_dir))
    env.update({key: value for key, value in stable_eval_environment().items() if key not in env})
    if env.get("HF_HUB_OFFLINE") == "1" or env.get("HF_DATASETS_OFFLINE") == "1":
        env["UV_OFFLINE"] = "1"
    return env


def task_args(suite: Literal["full", "smoke"]) -> list[str]:
    """Return OLMo-Eval task CLI arguments."""
    tasks = FULL_SUITE_TASKS if suite == "full" else SMOKE_TASKS
    args: list[str] = []
    for task in tasks:
        args.extend(["-t", task])
    return args


def run_olmo_eval(
    *,
    row: EvalManifestRow,
    model_dir: Path,
    output_dir: Path,
    olmo_eval_dir: Path,
    suite: Literal["full", "smoke"],
    limit: int | None,
    env: dict[str, str] | None = None,
) -> Path:
    """Run OLMo-Eval for one checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "olmo-eval",
        "run",
        "--harness",
        "default",
        "-o",
        "provider.kind=hf",
        "--num-gpus",
        "1",
        "--parallelism",
        "1",
        "-m",
        str(model_dir),
        *task_args(suite),
    ]
    if limit is not None:
        if suite != "smoke":
            raise ValueError("--limit is supported only for smoke suite to avoid ambiguous multi-task task options")
        command.extend(["-o", f"limit={limit}"])
    command.extend(
        [
            "--experiment-name",
            f"olmo_sc_{row.output_name}_{suite}",
            "--experiment-group",
            DEFAULT_EXPERIMENT_GROUP,
            "--output-dir",
            str(output_dir),
            "--no-save-predictions",
            "--no-save-requests",
        ]
    )
    subprocess.run(command, cwd=olmo_eval_dir, check=True, env=env)
    metrics_json = output_dir / "metrics.json"
    if not metrics_json.is_file():
        raise FileNotFoundError(f"OLMo-Eval did not produce metrics.json: {metrics_json}")
    return metrics_json


def run_writeback(
    *,
    row: EvalManifestRow,
    metrics_json: Path,
    writeback_script: Path,
    output_dir: Path,
    key_prefix: str,
    apply: bool,
) -> Path:
    """Mirror one OLMo-Eval metrics file to the original W&B run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--script",
        str(writeback_script),
        "--metrics-json",
        str(metrics_json),
        "--target-run-name",
        row.run_name,
        "--target-wandb-run-id",
        row.wandb_run_id,
        "--checkpoint-root",
        row.checkpoint_root,
        "--hf-repo",
        row.hf_repo_id,
        "--slurm-job-id",
        os.environ.get("SLURM_JOB_ID", "local"),
        "--olmo-eval-git-sha",
        os.environ.get("OLMO_EVAL_GIT_SHA", ""),
        "--key-prefix",
        key_prefix,
        "--output-dir",
        str(output_dir),
    ]
    if apply:
        command.append("--apply")
    subprocess.run(command, check=True)
    manifest = output_dir / "wandb_writeback_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"W&B writeback did not produce manifest: {manifest}")
    return manifest


def prewarm_datasets(args: argparse.Namespace) -> None:
    """Run OLMo-Eval once to build the datasets cache before fanout."""
    cache_env = cache_environment(args.work_dir)
    env = os.environ.copy()
    env.update(cache_env)
    env.update(stable_eval_environment())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "UV_OFFLINE"):
        env.pop(key, None)
        os.environ.pop(key, None)
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_MODULES_CACHE"):
        Path(cache_env[key]).mkdir(parents=True, exist_ok=True)
    basic_skills_root = ensure_basic_skills_snapshot(args.work_dir)
    rows = read_eval_manifest(args.manifest)
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"Manifest index {args.index} out of range for {len(rows)} rows")
    row = rows[args.index]
    model_dir = resolve_checkpoint(
        row,
        args.work_dir,
        checkpoint_mode=args.checkpoint_mode,
        revision=args.hf_revision,
    )
    output_dir = args.output_dir or (args.work_dir / "outputs" / f"dataset_prewarm_{row.output_name}")
    metrics_json = run_olmo_eval(
        row=row,
        model_dir=model_dir,
        output_dir=output_dir,
        olmo_eval_dir=args.olmo_eval_dir,
        suite=args.suite,
        limit=args.limit,
        env=env,
    )
    print(
        json.dumps(
            {
                "status": "prewarmed",
                "row": asdict(row),
                "metrics_json": str(metrics_json),
                "suite": args.suite,
                "limit": args.limit,
                "basic_skills_root": str(basic_skills_root),
            },
            sort_keys=True,
        )
    )


def run_one(args: argparse.Namespace) -> None:
    """Run one manifest row end-to-end."""
    rows = read_eval_manifest(args.manifest)
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"Manifest index {args.index} out of range for {len(rows)} rows")
    row = rows[args.index]
    work_dir = args.work_dir
    status_dir = work_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{row.output_name}.json"
    output_dir = work_dir / "outputs" / row.output_name
    writeback_dir = work_dir / "wandb_writeback" / row.output_name

    existing_metrics = output_dir / "metrics.json"
    existing_writeback = writeback_dir / "wandb_writeback_manifest.json"
    if args.skip_existing and existing_metrics.is_file() and existing_writeback.is_file():
        prior_status = {}
        if status_path.is_file():
            try:
                prior_status = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                prior_status = {}
        prior_was_applied = bool(prior_status.get("writeback_apply"))
        if args.writeback_apply and not prior_was_applied:
            print(
                json.dumps(
                    {
                        "status": "rerunning_existing_dry_run_for_apply",
                        "output_name": row.output_name,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            status_path.write_text(
                json.dumps(
                    {
                        "status": "skipped_existing",
                        "row": asdict(row),
                        "writeback_apply": prior_was_applied,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(json.dumps({"status": "skipped_existing", "output_name": row.output_name}, sort_keys=True))
            return

    try:
        model_dir = resolve_checkpoint(
            row,
            work_dir,
            checkpoint_mode=args.checkpoint_mode,
            revision=args.hf_revision,
        )
        metrics_json = run_olmo_eval(
            row=row,
            model_dir=model_dir,
            output_dir=output_dir,
            olmo_eval_dir=args.olmo_eval_dir,
            suite=args.suite,
            limit=args.limit,
            env=worker_environment(work_dir),
        )
        writeback_manifest = run_writeback(
            row=row,
            metrics_json=metrics_json,
            writeback_script=args.writeback_script,
            output_dir=writeback_dir,
            key_prefix=args.key_prefix,
            apply=args.writeback_apply,
        )
        status = {
            "status": "succeeded",
            "row": asdict(row),
            "metrics_json": str(metrics_json),
            "writeback_manifest": str(writeback_manifest),
            "writeback_apply": args.writeback_apply,
            "suite": args.suite,
            "limit": args.limit,
        }
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    finally:
        if args.cleanup_model:
            try:
                shutil.rmtree(downloaded_model_root_for_row(row, work_dir), ignore_errors=True)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "status": "cleanup_model_failed",
                            "output_name": row.output_name,
                            "error": repr(exc),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    print(json.dumps(status, sort_keys=True))


def write_sbatch(args: argparse.Namespace) -> None:
    """Write a Slurm array script for the manifest."""
    rows = read_eval_manifest(args.manifest)
    datasets_offline = bool(getattr(args, "datasets_offline", False))
    if args.offline and args.checkpoint_mode != "local-only":
        raise ValueError("--offline requires --checkpoint-mode local-only so array workers do not call the Hub")
    if args.checkpoint_mode == "local-only" and args.cleanup_model:
        raise ValueError("--checkpoint-mode local-only cannot be combined with --cleanup-model")
    runtime_manifest = getattr(args, "runtime_manifest", None) or args.manifest
    array_start = 0 if getattr(args, "array_start_index", None) is None else args.array_start_index
    array_max = len(rows) - 1 if getattr(args, "array_end_index", None) is None else args.array_end_index
    if array_start < 0 or array_max < array_start or array_max >= len(rows):
        raise IndexError(f"Invalid Slurm array range {array_start}..{array_max} for {len(rows)} rows")
    if args.array_concurrency < 1:
        raise ValueError("--array-concurrency must be >= 1")
    task_prep_workers = getattr(args, "task_prep_workers", DEFAULT_TASK_PREP_WORKERS)
    if task_prep_workers < 1:
        raise ValueError("--task-prep-workers must be >= 1")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    limit_arg = "" if args.limit is None else f" --limit {args.limit}"
    writeback_apply_arg = " --writeback-apply" if args.writeback_apply else ""
    cleanup_arg = " --cleanup-model" if args.cleanup_model else ""
    skip_existing_arg = " --skip-existing" if args.skip_existing else ""
    basic_skills_exports = ""
    if args.suite == "full" and (args.offline or datasets_offline):
        basic_skills_subsets = " ".join(BASIC_SKILLS_SUBTASKS)
        basic_skills_exports = f"""
basic_skills_ref_file="$HF_HUB_CACHE/datasets--allenai--basic-skills/refs/main"
if [[ -f "$basic_skills_ref_file" ]]; then
  basic_skills_snapshot_dir="$HF_HUB_CACHE/datasets--allenai--basic-skills/snapshots"
  export OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT="$basic_skills_snapshot_dir/$(cat "$basic_skills_ref_file")"
else
  echo "Missing prewarmed basic-skills snapshot ref: $basic_skills_ref_file" >&2
  exit 2
fi
for subset in {basic_skills_subsets}; do
  if [[ ! -f "$OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT/$subset/validation.json" ]]; then
    echo "Missing prewarmed basic-skills file: $OLMO_EVAL_BASIC_SKILLS_LOCAL_ROOT/$subset/validation.json" >&2
    exit 2
  fi
done
"""
    offline_exports = ""
    if args.offline:
        offline_exports = """
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
"""
    elif datasets_offline:
        offline_exports = """
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
"""
    revision_arg = "" if args.hf_revision is None else f" --hf-revision {args.hf_revision}"
    text = f"""#!/bin/bash
#SBATCH --job-name={args.job_name}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --array={array_start}-{array_max}%{args.array_concurrency}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --output={args.work_dir}/slurm/%x-%A_%a.out
#SBATCH --error={args.work_dir}/slurm/%x-%A_%a.err

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR={args.work_dir}/uv_cache
export UV_PYTHON_INSTALL_DIR={args.work_dir}/uv_python
export HF_HOME={args.work_dir}/hf_home
export HF_HUB_CACHE={args.work_dir}/hf_home/hub
export HUGGINGFACE_HUB_CACHE={args.work_dir}/hf_home/hub
export TRANSFORMERS_CACHE={args.work_dir}/hf_home/transformers
export HF_DATASETS_CACHE={args.work_dir}/hf_home/datasets
export HF_MODULES_CACHE={args.work_dir}/hf_home/modules
export HF_ALLOW_CODE_EVAL=1
export TMPDIR={args.work_dir}/tmp
export HF_TOKEN="${{HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
export OLMO_EVAL_TASK_PREP_WORKERS={task_prep_workers}
{offline_exports.rstrip()}
if [[ -z "${{WANDB_API_KEY:-}}" ]] && ! grep -q "wandb" "$HOME/.netrc" 2>/dev/null; then
  echo "WANDB_API_KEY is not set and $HOME/.netrc has no W&B credentials" >&2
  exit 2
fi
mkdir -p \\
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \\
  "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$TMPDIR" \\
  "{args.work_dir}/slurm"
{basic_skills_exports.rstrip()}

cd "{args.olmo_eval_dir}"
export OLMO_EVAL_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "slurm_array_task_id=${{SLURM_ARRAY_TASK_ID}}"
echo "cuda_visible_devices=${{CUDA_VISIBLE_DEVICES:-unset}}"
nvidia-smi || true

uv run --script "{args.worker_script}" run-one \\
  --manifest "{runtime_manifest}" \\
  --index "$SLURM_ARRAY_TASK_ID" \\
  --work-dir "{args.work_dir}" \\
  --olmo-eval-dir "{args.olmo_eval_dir}" \\
  --writeback-script "{args.writeback_script}" \\
  --suite {args.suite} \\
  --checkpoint-mode {args.checkpoint_mode} \\
  --key-prefix "{args.key_prefix}"{limit_arg}{revision_arg}{writeback_apply_arg}{cleanup_arg}{skip_existing_arg}

echo "finished_at=$(date -Is)"
"""
    output.write_text(text)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument("--input-manifest", action="append", required=True)
    build.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--expected-count", type=int)

    audit = subparsers.add_parser("audit-manifest")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--expected-count", type=int)
    audit.add_argument("--check-hf", action="store_true")

    stage = subparsers.add_parser("stage-checkpoints")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    stage.add_argument("--start-index", type=int)
    stage.add_argument("--end-index", type=int)
    stage.add_argument("--index", type=int, action="append")
    stage.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    stage.add_argument("--max-workers", type=int, default=4)
    stage.add_argument("--hf-revision")

    clean = subparsers.add_parser("clean-dataset-cache")
    clean.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    clean.add_argument("--apply", action="store_true")

    verify_cache = subparsers.add_parser("verify-offline-cache")
    verify_cache.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)

    prewarm = subparsers.add_parser("prewarm-datasets")
    prewarm.add_argument("--manifest", type=Path, required=True)
    prewarm.add_argument("--index", type=int, default=0)
    prewarm.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    prewarm.add_argument("--olmo-eval-dir", type=Path, default=DEFAULT_OLMO_EVAL_DIR)
    prewarm.add_argument("--output-dir", type=Path)
    prewarm.add_argument("--suite", choices=("full", "smoke"), default="full")
    prewarm.add_argument("--limit", type=int)
    prewarm.add_argument("--checkpoint-mode", choices=("download", "local-only"), default="local-only")
    prewarm.add_argument("--hf-revision")

    sbatch = subparsers.add_parser("write-sbatch")
    sbatch.add_argument("--manifest", type=Path, required=True)
    sbatch.add_argument("--runtime-manifest", type=Path)
    sbatch.add_argument("--array-start-index", type=int)
    sbatch.add_argument("--array-end-index", type=int)
    sbatch.add_argument("--output", type=Path, required=True)
    sbatch.add_argument("--worker-script", type=Path, required=True)
    sbatch.add_argument("--writeback-script", type=Path, default=DEFAULT_WRITEBACK_SCRIPT)
    sbatch.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    sbatch.add_argument("--olmo-eval-dir", type=Path, default=DEFAULT_OLMO_EVAL_DIR)
    sbatch.add_argument("--array-concurrency", type=int, default=32)
    sbatch.add_argument("--task-prep-workers", type=int, default=DEFAULT_TASK_PREP_WORKERS)
    sbatch.add_argument("--job-name", default="olmo-easy-716")
    sbatch.add_argument("--account", default="nlp")
    sbatch.add_argument("--partition", default="sc-loprio")
    sbatch.add_argument("--cpus-per-task", type=int, default=8)
    sbatch.add_argument("--mem", default="64G")
    sbatch.add_argument("--time", default="12:00:00")
    sbatch.add_argument("--suite", choices=("full", "smoke"), default="full")
    sbatch.add_argument("--limit", type=int)
    sbatch.add_argument("--key-prefix", default=DEFAULT_KEY_PREFIX)
    sbatch.add_argument("--writeback-apply", action="store_true")
    sbatch.add_argument("--cleanup-model", action=argparse.BooleanOptionalAction, default=False)
    sbatch.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    sbatch.add_argument("--checkpoint-mode", choices=("download", "local-only"), default="download")
    sbatch.add_argument("--hf-revision")
    sbatch.add_argument("--offline", action="store_true")
    sbatch.add_argument(
        "--datasets-offline",
        action="store_true",
        help=(
            "Run eval workers with datasets/transformers/uv offline while leaving HF Hub online for "
            "per-row checkpoint snapshot_download."
        ),
    )

    run = subparsers.add_parser("run-one")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    run.add_argument("--olmo-eval-dir", type=Path, default=DEFAULT_OLMO_EVAL_DIR)
    run.add_argument("--writeback-script", type=Path, default=DEFAULT_WRITEBACK_SCRIPT)
    run.add_argument("--suite", choices=("full", "smoke"), default="full")
    run.add_argument("--limit", type=int)
    run.add_argument("--key-prefix", default=DEFAULT_KEY_PREFIX)
    run.add_argument("--writeback-apply", action="store_true")
    run.add_argument("--cleanup-model", action="store_true")
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--checkpoint-mode", choices=("download", "local-only"), default="download")
    run.add_argument("--hf-revision")
    return parser.parse_args()


def main() -> None:
    """Run the selected subcommand."""
    args = parse_args()
    if args.command == "build-manifest":
        rows = build_eval_manifest(args.input_manifest, hf_repo_id=args.hf_repo_id)
        if args.expected_count is not None and len(rows) != args.expected_count:
            raise ValueError(f"Expected {args.expected_count} rows, found {len(rows)}")
        write_csv(rows, args.output)
        print(json.dumps(manifest_summary(rows), indent=2, sort_keys=True))
        return
    if args.command == "audit-manifest":
        rows = read_eval_manifest(args.manifest)
        if args.expected_count is not None and len(rows) != args.expected_count:
            raise ValueError(f"Expected {args.expected_count} rows, found {len(rows)}")
        report = manifest_summary(rows)
        if args.check_hf:
            report["hf"] = audit_hf_paths(rows)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "stage-checkpoints":
        stage_checkpoints(args)
        return
    if args.command == "clean-dataset-cache":
        clean_dataset_cache(args)
        return
    if args.command == "verify-offline-cache":
        verify_offline_cache(args)
        return
    if args.command == "prewarm-datasets":
        prewarm_datasets(args)
        return
    if args.command == "write-sbatch":
        if args.runtime_manifest is None:
            args.runtime_manifest = args.manifest
        write_sbatch(args)
        print(
            json.dumps({"sbatch": str(args.output), "row_count": len(read_eval_manifest(args.manifest))}, sort_keys=True)
        )
        return
    if args.command == "run-one":
        run_one(args)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
