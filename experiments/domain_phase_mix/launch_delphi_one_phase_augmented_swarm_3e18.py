# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the missing one-phase aliases of the Delphi 3e18 augmented fit panel.

The source is the exact 280-row two-phase panel. Each schedule is collapsed to
its realized phase-fraction-weighted aggregate and tied across both phases.
Rows already tied in the source panel are exact policy aliases and are recorded
as reused; only genuinely new coordinates are trained. Every new checkpoint is
followed by the Marin-native OLMoBaseEval Table-9 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix.launch_delphi_augmented_swarm_3e18 import (
    DEFAULT_ANALYSIS_OUTPUT_PATH,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_SOURCE_PANEL,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_ZONE,
    HF_HUB_DISABLE_XET_ENV_VAR,
    PHASE_NAMES,
    PHASE_SCHEDULE,
    SOURCE_PANEL_SHA256,
    TABLE9_EVAL_RESOURCES,
    TABLE9_REQUEST_SET_DIR,
    DelphiSwarmRunSpec,
    DelphiSwarmTrainingConfig,
    _default_validation_sets,
    _weight_diagnostics,
    load_source_panel,
    run_delphi_swarm_training,
)
from experiments.domain_phase_mix.launch_delphi_augmented_swarm_3e18 import (
    LOCAL_ARTIFACT_DIR as TWO_PHASE_LOCAL_ARTIFACT_DIR,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_one_phase_augmented_swarm_3e18_20260715"
LOCAL_ARTIFACT_DIR = TWO_PHASE_LOCAL_ARTIFACT_DIR.parent / "delphi_one_phase_augmented_swarm_3e18_20260715"
EXPECTED_TOTAL_ROWS = 280
EXPECTED_REUSED_ROWS = 42
EXPECTED_SCHEDULED_ROWS = 238
PHASE_TIED_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SaveOnePhaseManifestConfig:
    """Configuration for the one-phase audit manifest."""

    output_path: str
    source_panel: str
    analysis_output_path: str
    run_specs_json: str
    reused_source_names_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved one-phase manifest, training, and Table-9 graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def _source_is_phase_tied(run_spec: DelphiSwarmRunSpec) -> bool:
    return (
        max(
            abs(run_spec.phase_weights["phase_0"][domain] - run_spec.phase_weights["phase_1"][domain])
            for domain in DOMAIN_NAMES
        )
        <= PHASE_TIED_TOLERANCE
    )


def _collapse_run_spec(run_spec: DelphiSwarmRunSpec) -> DelphiSwarmRunSpec:
    phase_1_start_step = PHASE_SCHEDULE.phases[1].get_start_step_aligned(
        run_spec.train_steps,
        run_spec.batch_size,
        2048,
    )
    phase_0_fraction = phase_1_start_step / run_spec.train_steps
    phase_fractions = {"phase_0": phase_0_fraction, "phase_1": 1.0 - phase_0_fraction}
    aggregate = {
        domain: sum(
            phase_fractions[phase_name] * run_spec.phase_weights[phase_name][domain] for phase_name in PHASE_NAMES
        )
        for domain in DOMAIN_NAMES
    }
    total = sum(aggregate.values())
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"{run_spec.source_run_name} aggregate weights sum to {total}, expected 1")
    tied_weights = {phase_name: dict(aggregate) for phase_name in PHASE_NAMES}
    max_epoch, q95_epoch, phase_tv = _weight_diagnostics(tied_weights)
    return replace(
        run_spec,
        run_name=f"singleavg_{run_spec.run_name}",
        phase_0_fraction=phase_fractions["phase_0"],
        phase_1_fraction=phase_fractions["phase_1"],
        phase_weights=tied_weights,
        max_simulated_epoch=max_epoch,
        q95_simulated_epoch=q95_epoch,
        mean_phase_tv_to_proportional=phase_tv,
    )


def build_one_phase_specs(source_specs: list[DelphiSwarmRunSpec]) -> tuple[list[DelphiSwarmRunSpec], set[str]]:
    """Collapse all source schedules and identify exact phase-tied aliases."""
    if len(source_specs) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} source rows, found {len(source_specs)}")
    collapsed_specs = [_collapse_run_spec(spec) for spec in source_specs]
    reused_source_names = {spec.source_run_name for spec in source_specs if _source_is_phase_tied(spec)}
    if len(reused_source_names) != EXPECTED_REUSED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_REUSED_ROWS} phase-tied aliases, found {len(reused_source_names)}: "
            f"{sorted(reused_source_names)}"
        )
    scheduled_count = sum(spec.source_run_name not in reused_source_names for spec in collapsed_specs)
    if scheduled_count != EXPECTED_SCHEDULED_ROWS:
        raise ValueError(f"Expected {EXPECTED_SCHEDULED_ROWS} scheduled rows, found {scheduled_count}")
    scheduled_specs = [spec for spec in collapsed_specs if spec.source_run_name not in reused_source_names]
    scheduled_coordinates = {
        tuple(spec.phase_weights["phase_0"][domain].hex() for domain in DOMAIN_NAMES) for spec in scheduled_specs
    }
    if len(scheduled_coordinates) != EXPECTED_SCHEDULED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SCHEDULED_ROWS} distinct scheduled coordinates, found {len(scheduled_coordinates)}"
        )
    return collapsed_specs, reused_source_names


def save_one_phase_manifest(config: SaveOnePhaseManifestConfig) -> None:
    """Persist all 280 identities and mark exact aliases versus new runs."""
    run_specs = [DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    reused_source_names = set(json.loads(config.reused_source_names_json))
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    manifest_fields = [
        "run_order",
        "run_name",
        "source_run_name",
        "source_experiment",
        "panel_source",
        "disposition",
        "reused_two_phase_run_name",
        "data_seed",
        "trainer_seed",
        "phase_0_fraction",
        "phase_1_fraction",
        "realized_train_tokens",
        "max_simulated_epoch",
        "q95_simulated_epoch",
    ]
    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(manifest_buffer, fieldnames=manifest_fields)
    writer.writeheader()
    for spec in run_specs:
        reused = spec.source_run_name in reused_source_names
        writer.writerow(
            {
                "run_order": spec.run_order,
                "run_name": spec.run_name,
                "source_run_name": spec.source_run_name,
                "source_experiment": spec.source_experiment,
                "panel_source": spec.panel_source,
                "disposition": "reused_exact_phase_tied_alias" if reused else "scheduled_new_training",
                "reused_two_phase_run_name": spec.run_name.removeprefix("singleavg_") if reused else "",
                "data_seed": spec.data_seed,
                "trainer_seed": spec.trainer_seed,
                "phase_0_fraction": spec.phase_0_fraction,
                "phase_1_fraction": spec.phase_1_fraction,
                "realized_train_tokens": spec.realized_train_tokens,
                "max_simulated_epoch": spec.max_simulated_epoch,
                "q95_simulated_epoch": spec.q95_simulated_epoch,
            }
        )
    with fs.open(os.path.join(config.output_path, "training_manifest.csv"), "w") as handle:
        handle.write(manifest_buffer.getvalue())

    weights_buffer = io.StringIO(newline="")
    weights_writer = csv.DictWriter(
        weights_buffer,
        fieldnames=["run_name", "source_run_name", "disposition", "phase", "domain", "weight"],
    )
    weights_writer.writeheader()
    for spec in run_specs:
        disposition = (
            "reused_exact_phase_tied_alias" if spec.source_run_name in reused_source_names else "scheduled_new_training"
        )
        for phase_name in PHASE_NAMES:
            for domain in DOMAIN_NAMES:
                weights_writer.writerow(
                    {
                        "run_name": spec.run_name,
                        "source_run_name": spec.source_run_name,
                        "disposition": disposition,
                        "phase": phase_name,
                        "domain": domain,
                        "weight": spec.phase_weights[phase_name][domain],
                    }
                )
    with fs.open(os.path.join(config.output_path, "phase_weights.csv"), "w") as handle:
        handle.write(weights_buffer.getvalue())

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source_panel": config.source_panel,
        "source_panel_sha256": SOURCE_PANEL_SHA256,
        "analysis_output_path": config.analysis_output_path,
        "policy_class": "single_phase_tied",
        "collapse_rule": {
            "phase_0": run_specs[0].phase_0_fraction,
            "phase_1": run_specs[0].phase_1_fraction,
        },
        "total_rows": len(run_specs),
        "reused_exact_phase_tied_aliases": len(reused_source_names),
        "scheduled_new_training": len(run_specs) - len(reused_source_names),
        "native_table9_scheduled_for_new_training": True,
        "pairing": "New rows retain the source run_id, data seed, trainer seed, and Delphi configuration.",
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[DelphiSwarmRunSpec],
    reused_source_names: set[str],
    analysis_output_path: str,
    source_panel: str,
    validation_configs,
) -> LaunchArtifacts:
    """Build 238 new matched trainings while retaining a 280-row manifest."""
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        if run_spec.source_run_name in reused_source_names:
            continue
        resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        training_step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
            fn=remote(
                run_delphi_swarm_training,
                resources=resources,
                env_vars={HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
            resources=resources,
            config=DelphiSwarmTrainingConfig(
                analysis_output_path=analysis_output_path,
                output_path=this_output_path(),
                run_spec=run_spec,
                validation_configs=validation_configs,
                wandb_tags=("delphi-3e18-augmented-swarm", "fit-panel", "single-phase-ablation"),
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=TABLE9_REQUEST_SET_DIR,
                resource_config=TABLE9_EVAL_RESOURCES,
                wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_augmented_swarm",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_3e18_one_phase_augmented_fit_swarm",
                    "policy_class": "single_phase_tied",
                    "source_run_name": run_spec.source_run_name,
                    "swarm_run_name": run_spec.run_name,
                    "panel_source": run_spec.panel_source,
                },
            )
        )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_one_phase_manifest,
        config=SaveOnePhaseManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            reused_source_names_json=json.dumps(sorted(reused_source_names)),
        ),
    )
    if len(training_steps) != EXPECTED_SCHEDULED_ROWS or len(eval_steps) != EXPECTED_SCHEDULED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SCHEDULED_ROWS} train/eval pairs, found " f"{len(training_steps)}/{len(eval_steps)}"
        )
    return LaunchArtifacts(manifest_step=manifest_step, training_steps=training_steps, eval_steps=eval_steps)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--analysis-output-path", default=DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    if args.max_concurrent < 1 or args.max_concurrent > DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    source_specs = load_source_panel(
        source_panel=args.source_panel,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    run_specs, reused_source_names = build_one_phase_specs(source_specs)
    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        save_one_phase_manifest(
            SaveOnePhaseManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR),
                source_panel=args.source_panel,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                reused_source_names_json=json.dumps(sorted(reused_source_names)),
            )
        )
        logger.info(
            "Wrote %d one-phase rows (%d reused, %d scheduled) under %s",
            len(run_specs),
            len(reused_source_names),
            len(run_specs) - len(reused_source_names),
            LOCAL_ARTIFACT_DIR,
        )
        return

    validation_steps = _default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            reused_source_names=reused_source_names,
            analysis_output_path=args.analysis_output_path,
            source_panel=args.source_panel,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built one-phase graph with %d new training and %d native Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: one-phase ablation of the exact 280-row Delphi 3e18 augmented fit panel; "
            f"reuse {len(reused_source_names)} exact aliases and train {len(artifacts.training_steps)} new rows"
        ),
    )


if __name__ == "__main__":
    main()
