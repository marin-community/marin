# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a content-addressed HPR optimum panel at Delphi 3e18.

The candidate policies may be fit at 300M or Delphi 3e18, but every row is
trained with the same Delphi 3e18 configuration and receives the standard
smooth evaluations plus Marin-native OLMoBaseEval Table-9.
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

FIT_SOURCES = ("300m", "delphi_3e18")
DEFAULT_MAX_CONCURRENT = 56
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PanelMetadata:
    candidate_id: str
    target: str
    policy_class: str
    candidate_kind: str
    fit_source: str
    aggregate_kl_coefficient: float | None
    phase_information_budget: float | None


@dataclass(frozen=True)
class SavePanelManifestConfig:
    output_path: str
    experiment_name: str
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


def optional_float(value: str) -> float | None:
    if not value or value.lower() == "nan":
        return None
    return float(value)


def load_source_panel(
    *,
    source_panel: str,
    source_panel_sha256: str,
    expected_runs: int,
    experiment_name: str,
    run_id_base: int,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[augmented.DelphiSwarmRunSpec], list[PanelMetadata]]:
    """Load and strictly validate one frozen HPR source panel."""
    with fsspec.open(source_panel, "rb") as handle:
        source_bytes = handle.read()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != source_panel_sha256:
        raise ValueError(f"Source panel SHA-256 changed: {actual_sha256} != {source_panel_sha256}")
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    if len(rows) != expected_runs:
        raise ValueError(f"Expected {expected_runs} rows, found {len(rows)}")
    if not rows:
        raise ValueError("HPR panel is empty")

    candidate_ids = [row["candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("HPR panel contains duplicate candidate IDs")
    fit_sources = {row["fit_source"] for row in rows}
    if len(fit_sources) != 1 or next(iter(fit_sources)) not in FIT_SOURCES:
        raise ValueError(f"Invalid fit-source set: {fit_sources}")

    scaling_fits = augmented._read_scaling_fits(analysis_output_path)
    candidate = augmented._candidate_for_budget(scaling_fits=scaling_fits)
    realized_train_tokens = candidate.train_steps * augmented.TARGET_BATCH_SIZE * augmented.SEQ_LEN_DELPHI
    if realized_train_tokens > augmented.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError("Resolved Delphi token budget exceeds the fixed simulated-epoch budget")
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(augmented.completed_adamh_heuristic.vocab_size))
    tensor_parallel_size = augmented._tensor_parallel_size(candidate.model_config.hidden_dim, augmented.TARGET_TPU_TYPE)

    run_specs: list[augmented.DelphiSwarmRunSpec] = []
    metadata: list[PanelMetadata] = []
    for run_order, row in enumerate(rows):
        candidate_id = row["candidate_id"]
        phase_weights = augmented._phase_weights_from_row(row, source_run_name=candidate_id)
        max_epoch, q95_epoch, phase_tv = augmented._weight_diagnostics(phase_weights)
        run_specs.append(
            augmented.DelphiSwarmRunSpec(
                run_order=run_order,
                run_id=run_id_base + run_order,
                run_name=f"hprv_{run_order:03d}_{candidate_id}",
                source_run_name=candidate_id,
                source_experiment=experiment_name,
                panel_source=f"hpr_{row['fit_source']}_to_3e18",
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
                data_seed=run_id_base + run_order,
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
            PanelMetadata(
                candidate_id=candidate_id,
                target=row["target"],
                policy_class=row["policy_class"],
                candidate_kind=row["candidate_kind"],
                fit_source=row["fit_source"],
                aggregate_kl_coefficient=optional_float(row["aggregate_kl_coefficient"]),
                phase_information_budget=optional_float(row["phase_information_budget"]),
            )
        )
    return run_specs, metadata


def save_panel_manifest(config: SavePanelManifestConfig) -> None:
    """Persist source provenance and fully resolved training configurations."""
    run_specs = [augmented.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    metadata = [PanelMetadata(**item) for item in json.loads(config.metadata_json)]
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
        "experiment_name": config.experiment_name,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "n_runs": len(run_specs),
        "fit_source_counts": dict(Counter(item.fit_source for item in metadata)),
        "target_counts": dict(Counter(item.target for item in metadata)),
        "policy_counts": dict(Counter(item.policy_class for item in metadata)),
        "candidate_kind_counts": dict(Counter(item.candidate_kind for item in metadata)),
        "target_flops": augmented.TARGET_FLOPS,
        "native_table9_scheduled": True,
        "selection_uses_3e18_heldout_targets": False,
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[augmented.DelphiSwarmRunSpec],
    metadata: list[PanelMetadata],
    experiment_name: str,
    analysis_output_path: str,
    source_panel: str,
    source_panel_sha256: str,
    validation_configs,
) -> LaunchArtifacts:
    """Build the train plus native-Table-9 graph."""
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
                    "delphi-3e18-hpr-optimum-validation",
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
        fn=save_panel_manifest,
        config=SavePanelManifestConfig(
            output_path=this_output_path(),
            experiment_name=experiment_name,
            source_panel=source_panel,
            source_panel_sha256=source_panel_sha256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
        ),
    )
    return LaunchArtifacts(manifest_step, training_steps, eval_steps)


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
    if SHA256_PATTERN.fullmatch(args.source_panel_sha256) is None:
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

    run_specs, metadata = load_source_panel(
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
        save_panel_manifest(
            SavePanelManifestConfig(
                output_path=output_path,
                experiment_name=args.experiment_name,
                source_panel=args.source_panel,
                source_panel_sha256=args.source_panel_sha256,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
            )
        )
        logger.info("Validated %d frozen HPR-panel rows", len(run_specs))
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
            "Built HPR graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.experiment_name}: {len(run_specs)} HPR optimum candidates at Delphi 3e18 "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
