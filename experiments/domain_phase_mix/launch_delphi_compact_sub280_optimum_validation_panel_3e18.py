# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the Compact sub-280 raw-optimum learning curve at Delphi 3e18."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.domain_phase_mix import launch_delphi_compact_optimum_path_validation_panel_3e18 as compact_base
from experiments.domain_phase_mix import launch_delphi_hpr_optimum_validation_panel_3e18 as panel_base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_compact_sub280_optimum_validation_panel_20260721"
PANEL_TAG = "delphi-3e18-compact-sub280-optimum-validation"
EXPECTED_RUNS = 140
RUN_ID_BASE = 7_222_000
DEFAULT_MAX_CONCURRENT = 56
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--source-panel-sha256", required=True)
    parser.add_argument("--analysis-output-path", default=augmented.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=augmented.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=augmented.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != augmented.DEFAULT_TPU_REGION or args.tpu_zone != augmented.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {augmented.DEFAULT_TPU_REGION}/{augmented.DEFAULT_TPU_ZONE}")
    if args.max_concurrent < 1 or args.max_concurrent > DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    if SHA256_PATTERN.fullmatch(args.source_panel_sha256) is None:
        raise ValueError("--source-panel-sha256 must be a lowercase SHA-256 digest")
    if not args.source_panel.startswith("gs://marin-us-east5/"):
        raise ValueError("The source panel must be stored in marin-us-east5")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs, metadata = panel_base.load_source_panel(
        source_panel=args.source_panel,
        source_panel_sha256=args.source_panel_sha256,
        expected_runs=EXPECTED_RUNS,
        experiment_name=EXPERIMENT_NAME,
        run_id_base=RUN_ID_BASE,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    compact_base.validate_metadata(
        metadata,
        expected_runs=EXPECTED_RUNS,
        candidate_kind_prefix="compact_raw_optimum_sub280_",
    )
    run_specs = compact_base.canonical_run_specs(
        run_specs,
        metadata,
        experiment_name=EXPERIMENT_NAME,
        run_name_prefix="crslowv",
        panel_source="compact_sub280_raw_optimum_learning_curve",
    )
    if args.dry_run:
        output_path = f"/tmp/{Path(EXPERIMENT_NAME).name}-launch-dry-run"
        panel_base.save_panel_manifest(
            panel_base.SavePanelManifestConfig(
                output_path=output_path,
                experiment_name=EXPERIMENT_NAME,
                source_panel=args.source_panel,
                source_panel_sha256=args.source_panel_sha256,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
            )
        )
        logger.info("Validated %d frozen Compact sub-280 raw optima", len(run_specs))
        return

    validation_steps = augmented._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        steps = compact_base.build_steps(
            run_specs=run_specs,
            metadata=metadata,
            analysis_output_path=args.analysis_output_path,
            source_panel=args.source_panel,
            source_panel_sha256=args.source_panel_sha256,
            validation_configs=validation_configs,
            experiment_name=EXPERIMENT_NAME,
            panel_tag=PANEL_TAG,
            wandb_group="olmo_base_eval_table9_delphi_compact_sub280_optimum_validation_20260721",
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built %d training and %d native Table-9 steps; skipping launch in CI",
            EXPECTED_RUNS,
            EXPECTED_RUNS,
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            f"{EXPERIMENT_NAME}: {EXPECTED_RUNS} Compact sub-280 raw optima at Delphi 3e18 "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
