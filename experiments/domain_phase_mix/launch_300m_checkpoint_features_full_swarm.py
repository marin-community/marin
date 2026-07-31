# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas", "pyarrow"]
# ///
"""Launch MDE-style checkpoint-feature scoring for the full 300M swarm.

This scales the proportional canary to all eligible 300M/6B signal checkpoints.
It emits the same three feature surfaces per checkpoint:

* raw-text per-byte losses in ``scored_documents.parquet``;
* teacher-forced GSM8K/HumanEval request loglikelihoods;
* MCQ smooth-proxy per-choice loglikelihoods.

These are checkpoint-feature/log-mixture surrogate features, not true
single-domain MDE vertex-expert features.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import fsspec
import pandas as pd
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, InputName, executor_main, output_path_of, this_output_path

from experiments.domain_phase_mix.launch_300m_checkpoint_features_canary import (
    DEFAULT_MCQ_REQUEST_CACHE_URI,
    DEFAULT_TEACHER_FORCED_REQUEST_CACHE_URI,
    DEFAULT_TEXT_BUNDLES,
    MCQ_REQUEST_FEATURES_PARQUET,
    MCQ_SURFACE,
    SIGNAL_MATRIX_CSV,
    TEACHER_FORCED_REQUEST_FEATURES_PARQUET,
    TEACHER_FORCED_SURFACE,
    TEXT_BUNDLE_CHOICES,
    TEXT_FEATURE_SURFACE,
    CheckpointFeatureCanarySpec,
    _build_state_row_from_values,
    _load_state_rows,
    _parse_text_bundles,
    _read_csv,
    _require_nonempty_request_cache,
    _string_value,
    build_feature_steps,
    build_text_feature_datasets,
)
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    _executor_prefix,
    _slug,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_checkpoint_features_full_swarm"
STATE_CSV = OUTPUT_DIR / "300m_checkpoint_features_full_swarm_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_checkpoint_features_full_swarm_launch_manifest.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_checkpoint_features_full_swarm_20260528"
DEFAULT_MAX_CONCURRENT = 16
DEFAULT_MAX_DOCS_PER_DATASET = 512
EXPECTED_SIGNAL_ROWS = 242
SUMMARY_JSON = "checkpoint_feature_full_swarm_summary.json"
STATE_OUTPUT_CSV = "300m_checkpoint_features_full_swarm_state.csv"

SURFACE_ARTIFACTS = {
    TEXT_FEATURE_SURFACE: "scored_documents.parquet",
    TEACHER_FORCED_SURFACE: TEACHER_FORCED_REQUEST_FEATURES_PARQUET,
    MCQ_SURFACE: MCQ_REQUEST_FEATURES_PARQUET,
}


@dataclass(frozen=True)
class CollectCheckpointFeatureFullSwarmConfig:
    """Config for collecting full-swarm checkpoint-feature output paths."""

    output_path: str
    state_rows_json: str
    surface_output_paths: dict[str, InputName]


def _expected_checkpoint_step(row: pd.Series) -> int:
    for column in ("expected_checkpoint_step", "target_final_checkpoint_step"):
        if column not in row:
            continue
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return int(value)
    return 22_887


def _region_local_checkpoint_root(checkpoint_root: str, region: str) -> str:
    if not checkpoint_root.startswith("gs://marin-us-"):
        return checkpoint_root
    _bucket, _, path = checkpoint_root.removeprefix("gs://").partition("/")
    return f"gs://marin-{region}/{path}"


def _state_rows_from_matrix(
    *,
    matrix_csv: str | Path,
    text_bundle_keys: tuple[str, ...],
    text_dataset_names: tuple[str, ...],
    max_docs_per_dataset: int | None,
    max_eval_instances: int | None,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    allow_partial: bool,
) -> list[CheckpointFeatureCanarySpec]:
    frame = _read_csv(matrix_csv)
    required = {"run_name", "checkpoint_root", "source_experiment", "cohort"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Full-swarm checkpoint-feature matrix {matrix_csv} is missing columns: {missing}")
    if len(frame) != EXPECTED_SIGNAL_ROWS and not allow_partial:
        raise ValueError(
            f"Expected {EXPECTED_SIGNAL_ROWS} full-swarm signal rows in {matrix_csv}, found {len(frame)}. "
            "Pass --allow-partial only for debugging."
        )

    rows: list[CheckpointFeatureCanarySpec] = []
    roots: list[str] = []
    for _, row in frame.iterrows():
        checkpoint_root = _region_local_checkpoint_root(
            _string_value(row.get("checkpoint_root")).rstrip("/"),
            default_tpu_region,
        )
        if not checkpoint_root:
            continue
        roots.append(checkpoint_root)
        run_name = _string_value(row.get("run_name"))
        spec = _build_state_row_from_values(
            run_name=run_name,
            registry_key=_string_value(row.get("registry_run_key", row.get("registry_key"))),
            source_experiment=_string_value(row.get("source_experiment")),
            cohort=_string_value(row.get("cohort")),
            checkpoint_root=checkpoint_root,
            expected_checkpoint_step=_expected_checkpoint_step(row),
            text_bundle_keys=text_bundle_keys,
            text_dataset_names=text_dataset_names,
            max_docs_per_dataset=max_docs_per_dataset,
            max_eval_instances=max_eval_instances,
            default_tpu_type=default_tpu_type,
            default_tpu_region=default_tpu_region,
            default_tpu_zone=default_tpu_zone,
        )
        rows.append(replace(spec, step_name=f"checkpoint_feature_full_swarm/{_slug(run_name)}"))

    duplicates = sorted(root for root in set(roots) if roots.count(root) > 1)
    if duplicates:
        raise ValueError(f"Duplicate checkpoint roots in {matrix_csv}: {duplicates[:10]}")
    return rows


def _apply_requested_subset(
    rows: list[CheckpointFeatureCanarySpec], include_run_names: list[str]
) -> list[CheckpointFeatureCanarySpec]:
    if not include_run_names:
        return rows
    include = set(include_run_names)
    selected = [row for row in rows if row.run_name in include]
    missing = sorted(include - {row.run_name for row in selected})
    if missing:
        raise ValueError(f"Requested run names not present in checkpoint-feature state: {missing}")
    return selected


def _write_local_outputs(rows: list[CheckpointFeatureCanarySpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records([asdict(row) for row in rows])
    frame.to_csv(STATE_CSV, index=False)
    frame[frame["launch_decision"].eq("launch")].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _surface_output_key(row: CheckpointFeatureCanarySpec, surface: str) -> str:
    return f"{_slug(row.run_name)}::{surface}"


def build_full_swarm_feature_steps(
    *,
    name_prefix: str,
    state_rows: list[CheckpointFeatureCanarySpec],
    datasets,
    teacher_forced_request_cache_uri: str,
    mcq_request_cache_uri: str,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build checkpoint-feature steps for every full-swarm launch row."""
    steps: list[ExecutorStep] = []
    surface_outputs: dict[str, InputName] = {}
    for row in state_rows:
        row_steps, row_outputs = build_feature_steps(
            name_prefix=name_prefix,
            spec=row,
            datasets=datasets,
            teacher_forced_request_cache_uri=teacher_forced_request_cache_uri,
            mcq_request_cache_uri=mcq_request_cache_uri,
        )
        steps.extend(row_steps)
        for surface, output_path in row_outputs.items():
            surface_outputs[_surface_output_key(row, surface)] = output_path
    return steps, surface_outputs


def collect_checkpoint_feature_full_swarm(config: CollectCheckpointFeatureFullSwarmConfig) -> None:
    """Collect full-swarm checkpoint-feature output locations."""
    state_rows = [CheckpointFeatureCanarySpec(**row) for row in json.loads(config.state_rows_json)]
    rows_by_slug = {_slug(row.run_name): row for row in state_rows}
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)

    surface_rows = []
    for key, surface_output_path in sorted(config.surface_output_paths.items()):
        run_slug, surface = key.split("::", maxsplit=1)
        state_row = rows_by_slug[run_slug]
        surface_rows.append(
            {
                "run_name": state_row.run_name,
                "registry_key": state_row.registry_key,
                "checkpoint_root": state_row.checkpoint_root,
                "surface": surface,
                "output_path": str(surface_output_path),
                "primary_artifact": SURFACE_ARTIFACTS[surface],
            }
        )

    summary = {
        "state_rows": [asdict(row) for row in state_rows],
        "surfaces": surface_rows,
        "semantics": (
            "checkpoint-feature/log-mixture surrogate full-swarm extraction; "
            "not true single-domain MDE vertex-expert features"
        ),
    }
    with fsspec.open(os.path.join(output_path, SUMMARY_JSON), "wt") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[CheckpointFeatureCanarySpec],
    surface_output_paths: dict[str, InputName],
) -> ExecutorStep:
    """Build the final full-swarm checkpoint-feature collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_checkpoint_feature_full_swarm",
        description=f"Collect full-swarm checkpoint-feature outputs for {len(state_rows)} checkpoints",
        fn=collect_checkpoint_feature_full_swarm,
        config=CollectCheckpointFeatureFullSwarmConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            surface_output_paths=surface_output_paths,
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--matrix-csv", default=str(SIGNAL_MATRIX_CSV))
    parser.add_argument("--state-csv")
    parser.add_argument("--include-run-name", action="append", default=[])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--text-bundle", action="append", choices=TEXT_BUNDLE_CHOICES, default=[])
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-docs-per-dataset", type=int, default=DEFAULT_MAX_DOCS_PER_DATASET)
    parser.add_argument("--max-eval-instances", type=int)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--teacher-forced-request-cache-uri", default=DEFAULT_TEACHER_FORCED_REQUEST_CACHE_URI)
    parser.add_argument("--mcq-request-cache-uri", default=DEFAULT_MCQ_REQUEST_CACHE_URI)
    parser.add_argument(
        "--skip-request-cache-check",
        action="store_true",
        help="Skip pre-submit existence checks for smooth-proxy request-cache JSONL files.",
    )
    return parser.parse_known_args()


def _validate_live_state_source(*, dry_run: bool, state_csv: str | None, matrix_csv: str) -> None:
    if dry_run or os.getenv("CI") is not None:
        return
    if state_csv:
        if not str(state_csv).startswith("gs://"):
            raise ValueError(
                "Live full-swarm checkpoint-feature launches require --state-csv to be a gs:// path. "
                "Local state CSVs are excluded from Iris parent bundles."
            )
        return
    if str(matrix_csv).startswith("gs://"):
        return
    raise ValueError(
        "Live full-swarm checkpoint-feature launches must pass --state-csv gs://... or --matrix-csv gs://... . "
        "The default matrix CSV is local and excluded from Iris parent bundles."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    _validate_live_state_source(dry_run=args.dry_run, state_csv=args.state_csv, matrix_csv=args.matrix_csv)

    text_bundle_keys = _parse_text_bundles(args.text_bundle)
    if not args.skip_request_cache_check:
        _require_nonempty_request_cache(args.teacher_forced_request_cache_uri)
        _require_nonempty_request_cache(args.mcq_request_cache_uri)
    datasets = build_text_feature_datasets(text_bundle_keys)
    dataset_names = tuple(datasets)

    if args.state_csv is None:
        state_rows = _state_rows_from_matrix(
            matrix_csv=args.matrix_csv,
            text_bundle_keys=text_bundle_keys,
            text_dataset_names=dataset_names,
            max_docs_per_dataset=args.max_docs_per_dataset,
            max_eval_instances=args.max_eval_instances,
            default_tpu_type=args.tpu_type,
            default_tpu_region=args.tpu_region,
            default_tpu_zone=args.tpu_zone,
            allow_partial=args.allow_partial,
        )
    else:
        state_rows = _load_state_rows(args.state_csv)
    state_rows = _apply_requested_subset(state_rows, args.include_run_name)
    _write_local_outputs(state_rows)

    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Selected text feature bundles: %s", "+".join(text_bundle_keys or DEFAULT_TEXT_BUNDLES))
    logger.info("Selected %d text feature datasets", len(dataset_names))
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info(
        "Prepared %d launch checkpoints, %d total rows, and %d feature steps",
        launch_count,
        len(state_rows),
        launch_count * 3,
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    feature_steps, surface_output_paths = build_full_swarm_feature_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        datasets=datasets,
        teacher_forced_request_cache_uri=args.teacher_forced_request_cache_uri,
        mcq_request_cache_uri=args.mcq_request_cache_uri,
    )
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        surface_output_paths=surface_output_paths,
    )
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[*feature_steps, collect_step],
        description=f"{args.name_prefix}: 300M full-swarm checkpoint-feature extraction",
    )


if __name__ == "__main__":
    main()
