# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a frozen hybrid phase-ordering validation panel at Delphi 3e18."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.domain_phase_mix import launch_delphi_hpr_optimum_validation_panel_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 56


def build_launch_artifacts(
    *,
    run_specs: list[augmented.DelphiSwarmRunSpec],
    metadata: list[base.PanelMetadata],
    experiment_name: str,
    analysis_output_path: str,
    source_panel: str,
    source_panel_sha256: str,
    validation_configs,
) -> base.LaunchArtifacts:
    """Build the shared Delphi training plus native Table-9 graph."""
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for run_spec, row_metadata in zip(run_specs, metadata, strict=True):
        resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        training_step = ExecutorStep(
            name=f"{experiment_name}/{run_spec.run_name}",
            fn=remote(
                augmented.run_delphi_swarm_training,
                resources=resources,
                env_vars={augmented.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
            resources=resources,
            config=augmented.DelphiSwarmTrainingConfig(
                analysis_output_path=analysis_output_path,
                output_path=this_output_path(),
                run_spec=run_spec,
                validation_configs=validation_configs,
                wandb_tags=(
                    "delphi-3e18-hybrid-phase-ordering-validation",
                    f"fit-source={row_metadata.fit_source}",
                    f"target={row_metadata.target}",
                    f"policy={row_metadata.policy_class}",
                    f"kind={row_metadata.candidate_kind}",
                ),
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=augmented.TABLE9_REQUEST_SET_DIR,
                resource_config=augmented.TABLE9_EVAL_RESOURCES,
                wandb_group=f"olmo_base_eval_table9_{Path(experiment_name).name}",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": Path(experiment_name).name,
                    "scale": "3e18",
                    **asdict(row_metadata),
                },
            )
        )
    manifest_step = ExecutorStep(
        name=f"{experiment_name}/manifest",
        fn=base.save_panel_manifest,
        config=base.SavePanelManifestConfig(
            output_path=this_output_path(),
            experiment_name=experiment_name,
            source_panel=source_panel,
            source_panel_sha256=source_panel_sha256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
        ),
    )
    return base.LaunchArtifacts(manifest_step, training_steps, eval_steps)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--source-panel-sha256", required=True)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id-base", type=int, required=True)
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
    if args.expected_runs < 1 or args.run_id_base < 1:
        raise ValueError("Expected runs and run ID base must be positive")
    if base.SHA256_PATTERN.fullmatch(args.source_panel_sha256) is None:
        raise ValueError("--source-panel-sha256 must be a lowercase SHA-256 digest")
    if not args.source_panel.startswith("gs://marin-us-east5/"):
        raise ValueError("The source panel must be stored in marin-us-east5")
    if not args.experiment_name.startswith("pinlin_calvin_xu/data_mixture/"):
        raise ValueError("The experiment name must remain under pinlin_calvin_xu/data_mixture")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs, metadata = base.load_source_panel(
        source_panel=args.source_panel,
        source_panel_sha256=args.source_panel_sha256,
        expected_runs=args.expected_runs,
        experiment_name=args.experiment_name,
        run_id_base=args.run_id_base,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        output_path = f"/tmp/{Path(args.experiment_name).name}-launch-dry-run"
        base.save_panel_manifest(
            base.SavePanelManifestConfig(
                output_path=output_path,
                experiment_name=args.experiment_name,
                source_panel=args.source_panel,
                source_panel_sha256=args.source_panel_sha256,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
            )
        )
        logger.info("Validated %d frozen hybrid-panel rows", len(run_specs))
        return

    validation_steps = augmented._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            metadata=metadata,
            experiment_name=args.experiment_name,
            analysis_output_path=args.analysis_output_path,
            source_panel=args.source_panel,
            source_panel_sha256=args.source_panel_sha256,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built hybrid graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.experiment_name}: {len(run_specs)} 3e18-fit hybrid phase-ordering candidates "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
