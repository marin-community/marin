# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the frozen 120-row adversarial surrogate stress panel at Delphi 3e18.

The source panel is content-addressed and selected without consulting any
historical 3e18 target values. Every row runs the standard smooth training
evaluations and then the Marin-native OLMoBaseEval Table-9 evaluator. Stable
names and seeds make the graph safe to resubmit after partial failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUT_DIR / "delphi_3e18_adversarial_stress_panel_20260716"
DEFAULT_SOURCE_PANEL = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_adversarial_stress_panel_20260716/"
    "source/launcher_source_panel-1694c9ddaec95fee.csv"
)
SOURCE_PANEL_SHA256 = "1694c9ddaec95feef33d7f0f8175506497d4d0464c770dc9499e4f06261c4eb5"
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_adversarial_stress_panel_20260716"
EXPECTED_RUNS = 120
EXPECTED_TARGET_COUNTS = {"uncheatable": 60, "table9": 60}
EXPECTED_POLICY_COUNTS = {"single_phase_tied": 12, "two_phase": 108}
EXPECTED_SELECTION_COUNTS = {"baseline_ranked": 40, "challenger_ranked": 40, "high_disagreement": 40}
RUN_ID_BASE = 7_161_000
DEFAULT_MAX_CONCURRENT = 56


@dataclass(frozen=True)
class StressPanelMetadata:
    candidate_id: str
    target: str
    policy_class: str
    selection_stratum: str
    proposal_models: str


@dataclass(frozen=True)
class SaveStressPanelManifestConfig:
    output_path: str
    source_panel: str
    source_panel_sha256: str
    analysis_output_path: str
    run_specs_json: str
    metadata_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def load_source_panel(
    *,
    source_panel: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[augmented.DelphiSwarmRunSpec], list[StressPanelMetadata]]:
    """Load and strictly validate the content-addressed stress panel."""
    with fsspec.open(source_panel, "rb") as handle:
        source_bytes = handle.read()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SOURCE_PANEL_SHA256:
        raise ValueError(f"Source panel SHA-256 changed: {source_sha256} != {SOURCE_PANEL_SHA256}")
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} rows, found {len(rows)}")

    candidate_ids = [row["candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Stress panel contains duplicate candidate IDs")
    target_counts = Counter(row["target"] for row in rows)
    policy_counts = Counter(row["policy_class"] for row in rows)
    selection_counts = Counter(row["selection_stratum"] for row in rows)
    if dict(target_counts) != EXPECTED_TARGET_COUNTS:
        raise ValueError(f"Target counts changed: {dict(target_counts)} != {EXPECTED_TARGET_COUNTS}")
    if dict(policy_counts) != EXPECTED_POLICY_COUNTS:
        raise ValueError(f"Policy counts changed: {dict(policy_counts)} != {EXPECTED_POLICY_COUNTS}")
    if dict(selection_counts) != EXPECTED_SELECTION_COUNTS:
        raise ValueError(f"Selection counts changed: {dict(selection_counts)} != {EXPECTED_SELECTION_COUNTS}")

    scaling_fits = augmented._read_scaling_fits(analysis_output_path)
    candidate = augmented._candidate_for_budget(scaling_fits=scaling_fits)
    realized_train_tokens = candidate.train_steps * augmented.TARGET_BATCH_SIZE * augmented.SEQ_LEN_DELPHI
    if realized_train_tokens > augmented.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError("Resolved Delphi token budget exceeds the fixed simulated-epoch budget")
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(augmented.completed_adamh_heuristic.vocab_size))
    tensor_parallel_size = augmented._tensor_parallel_size(candidate.model_config.hidden_dim, augmented.TARGET_TPU_TYPE)

    run_specs = []
    metadata = []
    for run_order, row in enumerate(rows):
        candidate_id = row["candidate_id"]
        phase_weights = augmented._phase_weights_from_row(row, source_run_name=candidate_id)
        max_epoch, q95_epoch, phase_tv = augmented._weight_diagnostics(phase_weights)
        run_specs.append(
            augmented.DelphiSwarmRunSpec(
                run_order=run_order,
                run_id=RUN_ID_BASE + run_order,
                run_name=f"stress_{run_order:03d}_{candidate_id}",
                source_run_name=candidate_id,
                source_experiment=EXPERIMENT_NAME,
                panel_source="adversarial_stress",
                target_flops=augmented.TARGET_FLOPS,
                tpu_type=augmented.TARGET_TPU_TYPE,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                batch_size=augmented.TARGET_BATCH_SIZE,
                train_steps=candidate.train_steps,
                realized_train_tokens=realized_train_tokens,
                expected_checkpoint_step=candidate.train_steps - 1,
                model_hidden_dim=int(candidate.model_config.hidden_dim),
                model_layers=int(candidate.model_config.num_layers),
                non_embedding_params=non_embedding_params,
                total_trainable_params=total_params,
                tensor_parallel_size=tensor_parallel_size,
                data_seed=RUN_ID_BASE + run_order,
                trainer_seed=0,
                phase_boundary=augmented.PHASE_BOUNDARIES[0],
                phase_0_fraction=augmented.PHASE_FRACTIONS["phase_0"],
                phase_1_fraction=augmented.PHASE_FRACTIONS["phase_1"],
                simulated_epoch_target_budget=augmented.SIMULATED_EPOCH_TARGET_BUDGET,
                available_top_level_tokens=augmented.TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv,
                phase_weights=phase_weights,
            )
        )
        metadata.append(
            StressPanelMetadata(
                candidate_id=candidate_id,
                target=row["target"],
                policy_class=row["policy_class"],
                selection_stratum=row["selection_stratum"],
                proposal_models=row["proposal_models"],
            )
        )
    return run_specs, metadata


def save_stress_panel_manifest(config: SaveStressPanelManifestConfig) -> None:
    """Persist source provenance and fully resolved training configurations."""
    run_specs = [augmented.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    metadata = [StressPanelMetadata(**item) for item in json.loads(config.metadata_json)]
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(spec) for spec in run_specs], handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "panel_metadata.json"), "w") as handle:
        json.dump([asdict(item) for item in metadata], handle, indent=2, sort_keys=True)
    with (
        fsspec.open(config.source_panel, "r") as source,
        fs.open(os.path.join(config.output_path, "source_panel.csv"), "w") as destination,
    ):
        destination.write(source.read())
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "n_runs": len(run_specs),
        "target_counts": dict(Counter(item.target for item in metadata)),
        "policy_counts": dict(Counter(item.policy_class for item in metadata)),
        "selection_counts": dict(Counter(item.selection_stratum for item in metadata)),
        "target_flops": augmented.TARGET_FLOPS,
        "native_table9_scheduled": True,
        "selection_uses_historical_heldout_targets": False,
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[augmented.DelphiSwarmRunSpec],
    metadata: list[StressPanelMetadata],
    analysis_output_path: str,
    source_panel: str,
    validation_configs,
) -> LaunchArtifacts:
    """Build the 120-train plus 120-native-Table-9 graph."""
    training_steps = []
    eval_steps = []
    for run_spec, row_metadata in zip(run_specs, metadata, strict=True):
        resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        training_step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
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
                    "delphi-3e18-adversarial-stress",
                    "frozen-surrogate-stress-panel",
                    f"target={row_metadata.target}",
                    f"policy={row_metadata.policy_class}",
                    f"selection={row_metadata.selection_stratum}",
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
                wandb_group="olmo_base_eval_table9_delphi_3e18_adversarial_stress_20260716",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_3e18_adversarial_stress_panel_20260716",
                    "scale": "3e18",
                    "candidate_id": row_metadata.candidate_id,
                    "target": row_metadata.target,
                    "policy_class": row_metadata.policy_class,
                    "selection_stratum": row_metadata.selection_stratum,
                    "proposal_models": row_metadata.proposal_models,
                },
            )
        )
    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_stress_panel_manifest,
        config=SaveStressPanelManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            source_panel_sha256=SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
        ),
    )
    return LaunchArtifacts(manifest_step, training_steps, eval_steps)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=DEFAULT_SOURCE_PANEL)
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
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs, metadata = load_source_panel(
        source_panel=args.source_panel,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        save_stress_panel_manifest(
            SaveStressPanelManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR / "launch_dry_run"),
                source_panel=args.source_panel,
                source_panel_sha256=SOURCE_PANEL_SHA256,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
            )
        )
        logger.info("Validated %d frozen stress-panel rows", len(run_specs))
        return

    validation_steps = augmented._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            metadata=metadata,
            analysis_output_path=args.analysis_output_path,
            source_panel=args.source_panel,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built stress graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: frozen 120-row adversarial surrogate stress panel at Delphi 3e18 "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
