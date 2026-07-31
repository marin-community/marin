# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Marin-native OLMoBaseEval Easy Table 9 BPB from a CSV manifest.

This is for checkpoint coverage sweeps where the training graph has already
completed and we want Table-9 parity results for a set of HF-exported
checkpoints. Checkpoint paths in the manifest are prefix-relative to
``gs://marin-us-east5`` so parent and child reads stay region-local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import RESULTS_FILENAME, olmo_base_eval_step
from marin.execution import InputName
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, output_path_of, this_output_path

logger = logging.getLogger(__name__)

REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")
DEFAULT_MAX_CONCURRENT = 16
DEFAULT_WANDB_GROUP = "olmo_base_eval_table9_extra_checkpoint_coverage"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# The Table-9 evaluator was parity-tested on v6e-8. Keep it in us-east5 and
# pin to the v6e zone while reading checkpoints from the regional east5 bucket.
RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], zone="us-east5-b", disk="80g")


@dataclass(frozen=True)
class Table9ManifestTarget:
    """One HF-exported checkpoint to evaluate."""

    eval_name: str
    checkpoint: str
    panel: str
    scale: str
    run_name: str
    source_experiment: str
    checkpoint_root: str
    expected_checkpoint_step: int
    method: str


@dataclass(frozen=True)
class WriteManifestConfig:
    """Executor config for writing the submitted target manifest."""

    output_path: str
    targets_json: str


@dataclass(frozen=True)
class CollectResultsConfig:
    """Executor config for collecting Table-9 result JSON files into one CSV."""

    output_path: str
    targets_json: str
    results_by_eval_name: dict[str, InputName]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch OLMoBaseEval Table-9 BPB for a checkpoint manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument(
        "--list-only", action="store_true", help="Validate and summarize manifest targets without launching."
    )
    return parser.parse_known_args()


def _required_value(row: dict[str, str], key: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"Manifest row is missing required {key}: {row}")
    return value


def _load_targets(manifest_path: str, manifest_sha256: str) -> list[Table9ManifestTarget]:
    if SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise ValueError("--manifest-sha256 must be a lowercase SHA-256 digest")
    with fsspec.open(manifest_path, mode="rb") as f:
        manifest_bytes = f.read()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != manifest_sha256:
        raise ValueError(f"Manifest SHA-256 changed: {actual_sha256} != {manifest_sha256}")
    with io.StringIO(manifest_bytes.decode()) as manifest_text:
        reader = csv.DictReader(manifest_text)
        targets = [
            Table9ManifestTarget(
                eval_name=_required_value(row, "eval_name"),
                checkpoint=_required_value(row, "checkpoint"),
                panel=_required_value(row, "panel"),
                scale=_required_value(row, "scale"),
                run_name=_required_value(row, "run_name"),
                source_experiment=_required_value(row, "source_experiment"),
                checkpoint_root=_required_value(row, "checkpoint_root"),
                expected_checkpoint_step=int(_required_value(row, "expected_checkpoint_step")),
                method=row.get("method", "").strip(),
            )
            for row in reader
        ]

    if not targets:
        raise ValueError(f"No targets found in {manifest_path}")

    duplicate_names = sorted(
        {target.eval_name for target in targets if sum(t.eval_name == target.eval_name for t in targets) > 1}
    )
    if duplicate_names:
        raise ValueError(f"Duplicate eval names in {manifest_path}: {duplicate_names[:10]}")

    bad_checkpoints = [target.checkpoint for target in targets if not target.checkpoint.startswith("checkpoints/")]
    if bad_checkpoints:
        raise ValueError(
            "Manifest checkpoints must be prefix-relative paths under checkpoints/: " f"{bad_checkpoints[:5]}"
        )

    central_paths = [
        target.checkpoint_root
        for target in targets
        if "marin-us-central" in target.checkpoint_root or "us-central" in target.checkpoint
    ]
    if central_paths:
        raise ValueError(f"Central-region checkpoint paths are not allowed: {central_paths[:5]}")

    return targets


def write_manifest(config: WriteManifestConfig) -> None:
    """Write the exact target manifest submitted to the executor graph."""
    targets = [asdict(Table9ManifestTarget(**target)) for target in json.loads(config.targets_json)]
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, "table9_eval_manifest.json"), "w") as f:
        json.dump({"target_count": len(targets), "targets": targets}, f, indent=2, sort_keys=True)


def collect_results(config: CollectResultsConfig) -> None:
    """Collect result JSONs into one CSV with macro and component BPBs."""
    targets = [Table9ManifestTarget(**target) for target in json.loads(config.targets_json)]
    rows: list[dict[str, Any]] = []
    for target in targets:
        with fsspec.open(config.results_by_eval_name[target.eval_name], "r") as f:
            result = json.load(f)
        rows.append(
            {
                **asdict(target),
                "table9_macro_bpb": result["table9_macro_bpb"],
                **{f"table9/{component}/bpb": value for component, value in result["table9_components"].items()},
            }
        )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, "table9_eval_results.csv"), "w") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _step(target: Table9ManifestTarget, *, wandb_group: str):
    return olmo_base_eval_step(
        name=target.eval_name,
        checkpoint=InputName.hardcoded(target.checkpoint),
        request_set_dir=REQUEST_SET_DIR,
        resource_config=RESOURCES,
        wandb_group=wandb_group,
        provenance={
            "evaluator": "marin-native-table9-bpb",
            "panel": target.panel,
            "scale": target.scale,
            "source_run_name": target.run_name,
            "source_experiment": target.source_experiment,
            "checkpoint_root": target.checkpoint_root,
            "method": target.method,
        },
    )


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    targets = _load_targets(args.manifest, args.manifest_sha256)
    if args.list_only:
        by_panel_scale: dict[tuple[str, str], int] = {}
        for target in targets:
            key = (target.panel, target.scale)
            by_panel_scale[key] = by_panel_scale.get(key, 0) + 1
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "manifest_sha256": args.manifest_sha256,
                    "target_count": len(targets),
                    "by_panel_scale": {f"{panel}/{scale}": count for (panel, scale), count in by_panel_scale.items()},
                    "first_targets": [asdict(target) for target in targets[:5]],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    targets_json = json.dumps([asdict(target) for target in targets], sort_keys=True)
    manifest_step = ExecutorStep(
        name=f"evaluation/olmo_base_eval_table9/manifest_checkpoint_coverage_{args.manifest_sha256[:12]}_manifest",
        description=f"Write Table-9 eval manifest for {len(targets)} completed checkpoints",
        fn=write_manifest,
        config=WriteManifestConfig(output_path=this_output_path(), targets_json=targets_json),
    )

    eval_steps = [_step(target, wandb_group=args.wandb_group) for target in targets]
    results_by_eval_name = {
        target.eval_name: output_path_of(step, RESULTS_FILENAME)
        for target, step in zip(targets, eval_steps, strict=True)
    }
    collect_step = ExecutorStep(
        name=f"evaluation/olmo_base_eval_table9/manifest_checkpoint_coverage_{args.manifest_sha256[:12]}_collect",
        description=f"Collect Table-9 eval results for {len(targets)} completed checkpoints",
        fn=collect_results,
        config=CollectResultsConfig(
            output_path=this_output_path(),
            targets_json=targets_json,
            results_by_eval_name=results_by_eval_name,
        ),
    )

    logger.info(
        "Launching OLMoBaseEval Table-9 for %d manifest checkpoints with max_concurrent=%d.",
        len(targets),
        args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[manifest_step, *eval_steps, collect_step],
        description=f"OLMoBaseEval Table-9 manifest coverage: {args.manifest}",
    )


if __name__ == "__main__":
    main()
