# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the frozen Compact Retained State raw-optimum path panel at Delphi 3e18."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, replace
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
from experiments.domain_phase_mix import launch_delphi_hpr_optimum_validation_panel_3e18 as panel_base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_compact_optimum_path_validation_panel_20260721"
PANEL_TAG = "delphi-3e18-compact-optimum-path-validation"
EXPECTED_RUNS = 15
RUN_ID_BASE = 7_221_000
MAX_CONCURRENT = EXPECTED_RUNS
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--source-panel-sha256", required=True)
    parser.add_argument("--analysis-output-path", default=augmented.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=augmented.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=augmented.DEFAULT_TPU_ZONE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def canonical_run_specs(
    run_specs: list[augmented.DelphiSwarmRunSpec],
    metadata: list[panel_base.PanelMetadata],
    *,
    experiment_name: str = EXPERIMENT_NAME,
    run_name_prefix: str = "crsv",
    panel_source: str = "compact_raw_optimum_path",
) -> list[augmented.DelphiSwarmRunSpec]:
    canonical: list[augmented.DelphiSwarmRunSpec] = []
    for run_order, (run_spec, item) in enumerate(zip(run_specs, metadata, strict=True)):
        canonical.append(
            replace(
                run_spec,
                run_name=f"{run_name_prefix}_{run_order:03d}_{item.candidate_id}",
                source_experiment=experiment_name,
                panel_source=panel_source,
            )
        )
    return canonical


def validate_metadata(
    metadata: list[panel_base.PanelMetadata],
    *,
    expected_runs: int = EXPECTED_RUNS,
    candidate_kind_prefix: str = "compact_raw_optimum_",
) -> None:
    if len(metadata) != expected_runs:
        raise ValueError(f"Expected {expected_runs} candidates, found {len(metadata)}")
    if {item.target for item in metadata} != {"uncheatable", "table9"}:
        raise ValueError("The panel must contain both proposal targets")
    if {item.policy_class for item in metadata} != {"two_phase"}:
        raise ValueError("The panel must contain only two-phase policies")
    if {item.fit_source for item in metadata} != {"delphi_3e18"}:
        raise ValueError("Every proposal must be fit exclusively on Delphi 3e18 evidence")
    if not all(item.candidate_kind.startswith(candidate_kind_prefix) for item in metadata):
        raise ValueError("Unexpected candidate kind in Compact optimum panel")


def build_steps(
    *,
    run_specs: list[augmented.DelphiSwarmRunSpec],
    metadata: list[panel_base.PanelMetadata],
    analysis_output_path: str,
    source_panel: str,
    source_panel_sha256: str,
    validation_configs,
    experiment_name: str = EXPERIMENT_NAME,
    panel_tag: str = PANEL_TAG,
    wandb_group: str = "olmo_base_eval_table9_delphi_compact_optimum_path_validation_20260721",
) -> list[ExecutorStep]:
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for run_spec, item in zip(run_specs, metadata, strict=True):
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
                    panel_tag,
                    "model=compact_retained_state",
                    f"target={item.target}",
                    f"policy={item.policy_class}",
                    f"kind={item.candidate_kind}",
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
                wandb_group=wandb_group,
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": Path(experiment_name).name,
                    "scale": "3e18",
                    "model": "compact_retained_state",
                    **asdict(item),
                },
            )
        )

    manifest_step = ExecutorStep(
        name=f"{experiment_name}/manifest",
        fn=panel_base.save_panel_manifest,
        config=panel_base.SavePanelManifestConfig(
            output_path=this_output_path(),
            experiment_name=experiment_name,
            source_panel=source_panel,
            source_panel_sha256=source_panel_sha256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
        ),
    )
    return [manifest_step, *training_steps, *eval_steps]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != augmented.DEFAULT_TPU_REGION or args.tpu_zone != augmented.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {augmented.DEFAULT_TPU_REGION}/{augmented.DEFAULT_TPU_ZONE}")
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
    validate_metadata(metadata)
    run_specs = canonical_run_specs(run_specs, metadata)
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
        logger.info("Validated %d frozen Compact raw-optimum policies", len(run_specs))
        return

    validation_steps = augmented._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        steps = build_steps(
            run_specs=run_specs,
            metadata=metadata,
            analysis_output_path=args.analysis_output_path,
            source_panel=args.source_panel,
            source_panel_sha256=args.source_panel_sha256,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d training and %d native Table-9 steps; skipping launch in CI", EXPECTED_RUNS, EXPECTED_RUNS)
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=MAX_CONCURRENT),
        steps=steps,
        description=(
            f"{EXPERIMENT_NAME}: {EXPECTED_RUNS} Compact raw-optimum path candidates at Delphi 3e18 "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
