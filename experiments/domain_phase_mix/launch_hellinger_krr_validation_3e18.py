# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate two frozen content-Hellinger KRR proposals at Delphi 3e18.

The source panel contains one two-phase proposal optimized for Uncheatable and
one optimized for Table-9. Both were selected by the same fit-only protocol:
nested-CV Hellinger KRR on the canonical 280-row Delphi fit swarm, followed by
averaging the top 64 members of a frozen candidate bank after the unconstrained
continuous optimum failed a training-derived support gate.

Both checkpoints receive the standard smooth Uncheatable evaluation and native
OLMoBaseEval Table-9, so target-matched and cross-target outcomes are complete.
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
from levanter.data.text.datasets import DatasetComponent
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
LOCAL_ARTIFACT_DIR = (
    SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs" / "hellinger_krr_delphi_3e18_validation_20260727"
)
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/hellinger_krr_delphi_3e18_validation_20260727"
PANEL_ID = "hellinger_krr_delphi_3e18_validation_20260727"
DEFAULT_SOURCE_PANEL = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "hellinger_krr_delphi_3e18_validation_20260727/"
    "source/validation_panel-bd6040e58d66f330.csv"
)
SOURCE_PANEL_SHA256 = "bd6040e58d66f3309d1c0aace3231838f3042580e4c1d483081347ec6fac58de"
EXPECTED_TARGETS = {"uncheatable_bpb", "table9_macro_bpb"}
EXPECTED_RUNS = 2
EXPECTED_CANDIDATE_KIND = "top_64_average"
MAX_SUPPORT_DISTANCE = 0.041858 + 1e-6
DEFAULT_MAX_CONCURRENT = 2
REQUIRED_BUCKET_PREFIX = "gs://marin-us-east5/"


@dataclass(frozen=True)
class SaveManifestConfig:
    """Persist the source panel and fully resolved run specs."""

    output_path: str
    source_panel: str
    source_panel_sha256: str
    analysis_output_path: str
    run_specs_json: str
    proposal_metadata_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--analysis-output-path", default=augmented.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=augmented.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=augmented.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def source_rows(source_panel: str) -> list[dict[str, str]]:
    """Load and validate the immutable two-proposal panel."""
    with fsspec.open(source_panel, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_PANEL_SHA256:
        raise ValueError(f"Source panel digest changed: {digest} != {SOURCE_PANEL_SHA256}")
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} proposals, found {len(rows)}")
    targets = {row["proposal_target"] for row in rows}
    if targets != EXPECTED_TARGETS:
        raise ValueError(f"Target set changed: {sorted(targets)}")
    if {row["candidate_kind"] for row in rows} != {EXPECTED_CANDIDATE_KIND}:
        raise ValueError("The panel no longer contains only frozen top-64 proposals")
    if len({row["run_name"] for row in rows}) != EXPECTED_RUNS:
        raise ValueError("Run names are not unique")
    if len({row["data_seed"] for row in rows}) != EXPECTED_RUNS:
        raise ValueError("Data seeds are not unique")
    for row in rows:
        support = float(row["nearest_fit_hellinger_sq"])
        if support > MAX_SUPPORT_DISTANCE:
            raise ValueError(f"{row['run_name']} support distance {support} exceeds the frozen q95 gate")
    return rows


def build_run_specs(
    rows: list[dict[str, str]],
    *,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[augmented.DelphiSwarmRunSpec]:
    scaling_fits = augmented._read_scaling_fits(analysis_output_path)
    candidate = augmented._candidate_for_budget(scaling_fits=scaling_fits)
    realized_train_tokens = candidate.train_steps * augmented.TARGET_BATCH_SIZE * augmented.SEQ_LEN_DELPHI
    if realized_train_tokens > augmented.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError("Resolved training tokens exceed the simulated-epoch target budget")
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(augmented.completed_adamh_heuristic.vocab_size))
    tensor_parallel_size = augmented._tensor_parallel_size(candidate.model_config.hidden_dim, augmented.TARGET_TPU_TYPE)

    run_specs = []
    for run_order, row in enumerate(rows):
        phase_weights = augmented._phase_weights_from_row(row, source_run_name=row["run_name"])
        max_epoch, q95_epoch, phase_tv = augmented._weight_diagnostics(phase_weights)
        if abs(max_epoch - float(row["nominal_0p8_max_simulated_epoch"])) > 2e-6:
            raise ValueError(f"{row['run_name']} max epoch changed: {max_epoch}")
        pairwise_phase_tv = 0.5 * sum(
            abs(phase_weights["phase_1"][domain] - phase_weights["phase_0"][domain]) for domain in augmented.DOMAIN_NAMES
        )
        if abs(pairwise_phase_tv - float(row["phase_tv"])) > 1e-8:
            raise ValueError(f"{row['run_name']} pairwise phase TV changed: {pairwise_phase_tv}")
        if abs(phase_tv - float(row["mean_phase_tv_to_proportional"])) > 1e-8:
            raise ValueError(f"{row['run_name']} proportional phase TV changed: {phase_tv}")
        run_specs.append(
            augmented.DelphiSwarmRunSpec(
                run_order=run_order,
                run_id=int(row["run_id"]),
                run_name=row["run_name"],
                source_run_name=row["run_name"],
                source_experiment=row["source_experiment"],
                panel_source=row["panel_source"],
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
                data_seed=int(row["data_seed"]),
                trainer_seed=int(row["trainer_seed"]),
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
    return run_specs


def save_manifest(config: SaveManifestConfig) -> None:
    run_specs = [augmented.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    metadata = json.loads(config.proposal_metadata_json)
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(spec) for spec in run_specs], handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "proposal_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (
        fsspec.open(config.source_panel, "r") as source,
        fs.open(os.path.join(config.output_path, "source_panel.csv"), "w") as destination,
    ):
        destination.write(source.read())
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "panel_id": PANEL_ID,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "n_runs": len(run_specs),
        "proposal_targets": dict(Counter(item["proposal_target"] for item in metadata)),
        "candidate_kind": EXPECTED_CANDIDATE_KIND,
        "target_flops": augmented.TARGET_FLOPS,
        "native_table9_scheduled_for_all": True,
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    rows: list[dict[str, str]],
    run_specs: list[augmented.DelphiSwarmRunSpec],
    source_panel: str,
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
) -> LaunchArtifacts:
    training_steps = []
    eval_steps = []
    for row, run_spec in zip(rows, run_specs, strict=True):
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
                    "issue-7067",
                    "delphi-3e18-hellinger-krr-validation",
                    "two-phase",
                    f"proposal-target={row['proposal_target']}",
                    EXPECTED_CANDIDATE_KIND,
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
                wandb_group="olmo_base_eval_table9_hellinger_krr_validation",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": PANEL_ID,
                    "scale": "3e18",
                    "source_run_name": run_spec.source_run_name,
                    "proposal_target": row["proposal_target"],
                    "candidate_kind": row["candidate_kind"],
                },
            )
        )
    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_manifest,
        config=SaveManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            source_panel_sha256=SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            proposal_metadata_json=json.dumps(rows, sort_keys=True),
        ),
    )
    return LaunchArtifacts(manifest_step, training_steps, eval_steps)


def write_local_dry_run(
    *,
    rows: list[dict[str, str]],
    run_specs: list[augmented.DelphiSwarmRunSpec],
    source_panel: str,
    analysis_output_path: str,
) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_manifest(
        SaveManifestConfig(
            output_path=str(LOCAL_ARTIFACT_DIR),
            source_panel=source_panel,
            source_panel_sha256=SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            proposal_metadata_json=json.dumps(rows, sort_keys=True),
        )
    )


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
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required east5 prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    if not args.source_panel.startswith(REQUIRED_BUCKET_PREFIX):
        raise ValueError(f"Source panel must be in east5: {args.source_panel}")

    rows = source_rows(args.source_panel)
    run_specs = build_run_specs(
        rows,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        write_local_dry_run(
            rows=rows,
            run_specs=run_specs,
            source_panel=args.source_panel,
            analysis_output_path=args.analysis_output_path,
        )
        logger.info("Wrote %d dry-run specs under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = augmented._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            rows=rows,
            run_specs=run_specs,
            source_panel=args.source_panel,
            analysis_output_path=args.analysis_output_path,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built Hellinger KRR graph with %d training and %d Table-9 steps",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=f"{EXPERIMENT_NAME}: two frozen content-Hellinger KRR proposals at Delphi 3e18",
    )


if __name__ == "__main__":
    main()
