# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate fixed-aggregate phase-information sweeps at the Delphi 3e18 rung."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.defaults import default_validation_sets
from experiments.domain_phase_mix import launch_delphi_uncheatable_optimized_mixtures as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
DEFAULT_MANIFEST = str(
    REFERENCE_OUTPUT_DIR / "decoupled_phase_information_validation_panel_20260712" / "selected_candidate_manifest.csv"
)
DEFAULT_LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUT_DIR / "delphi_decoupled_phase_information_validation_3e18_20260712"
DEFAULT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_decoupled_phase_information_validation_3e18_20260712"
TARGET_FLOPS = 3e18
PANEL_DATA_SEED = 690300
RUN_ID_BASE = 712_000
MAX_ALLOWED_CONCURRENT = 56
WANDB_SERIES_TAG = "delphi-decoupled-phase-information"
GITHUB_ISSUE = 6611
REQUIRED_MANIFEST_COLUMNS = {
    "candidate",
    "objective",
    "anchor_tag",
    "family",
    "phase_information_budget",
    "max_simulated_epoch",
    "source_csv",
}


class PanelMixtureKey(str):
    """String key with the value interface used by the shared Delphi launcher."""

    @property
    def value(self) -> str:
        return str(self)


@dataclass(frozen=True)
class PanelTrainingConfig:
    """Primitive-only config for one manifest-driven training child."""

    experiment_name: str
    analysis_output_path: str
    candidate: str
    source_csv: str
    objective: str
    family: str
    phase_information_budget: float
    expected_max_simulated_epoch: float
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    output_path: str
    run_id: int
    run_name: str
    data_seed: int


@dataclass(frozen=True)
class SavePanelManifestConfig:
    """Persist the exact selected rows and launch invariants."""

    output_path: str
    rows_json: str
    experiment_name: str
    analysis_output_path: str
    max_concurrent: int


def target_metric(objective: str) -> str:
    if objective == "table9":
        return base.TABLE9_TARGET_METRIC
    if objective == "uncheatable":
        return "eval/uncheatable_eval/bpb"
    raise ValueError(f"Unknown objective: {objective}")


def panel_source(config: PanelTrainingConfig) -> tuple[PanelMixtureKey, base.MixtureSource]:
    key = PanelMixtureKey(config.candidate)
    typed_key = cast(base.DelphiValidationMixture, key)
    source = base.MixtureSource(
        key=typed_key,
        display_name=(
            f"decoupled phase information {config.family} {config.objective} "
            f"epsilon={config.phase_information_budget:g}"
        ),
        source_csv=config.source_csv,
        github_issue=GITHUB_ISSUE,
        target_metric=target_metric(config.objective),
        method=f"decoupled-phase-information-{config.family}",
        wandb_series_tag=WANDB_SERIES_TAG,
        expected_max_simulated_epoch=config.expected_max_simulated_epoch,
        data_seed_override=config.data_seed,
    )
    return key, source


def run_panel_training(config: PanelTrainingConfig) -> None:
    """Register one manifest source and delegate to the shared Delphi trainer."""
    key, source = panel_source(config)
    typed_key = cast(base.DelphiValidationMixture, key)
    base.MIXTURE_SOURCES[typed_key] = source
    base.EXPERIMENT_NAME = config.experiment_name
    base.run_delphi_optimized_training(
        base.DelphiOptimizedTrainingConfig(
            analysis_output_path=config.analysis_output_path,
            target_flops=config.target_flops,
            tpu_type=config.tpu_type,
            tpu_region=config.tpu_region,
            tpu_zone=config.tpu_zone,
            batch_size=config.batch_size,
            mixture=typed_key,
            label=base.LABEL,
            output_path=config.output_path,
            run_id=config.run_id,
            run_name=config.run_name,
            data_seed=config.data_seed,
            train_tokens_override=None,
            trainer_seed=0,
            validation_configs=configured_validation_sets(),
        )
    )


def configured_validation_sets():
    validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
    return {name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()}


def save_panel_manifest(config: SavePanelManifestConfig) -> None:
    """Write exact candidate provenance beside the executor graph."""
    rows = json.loads(config.rows_json)
    summary = {
        "experiment_name": config.experiment_name,
        "analysis_output_path": config.analysis_output_path,
        "target_flops": TARGET_FLOPS,
        "data_seed": PANEL_DATA_SEED,
        "trainer_seed": 0,
        "phase_fractions": dict(base.PHASE_FRACTIONS),
        "max_concurrent": config.max_concurrent,
        "candidate_count": len(rows),
        "repeat_count_per_candidate": 1,
        "native_table9_eval_for_every_candidate": True,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "selected_candidate_manifest.json"), "w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def read_manifest(path: str) -> list[dict[str, str]]:
    with fsspec.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no candidates")
    candidates = [row["candidate"] for row in rows]
    if len(candidates) != len(set(candidates)):
        raise ValueError("Panel manifest has duplicate candidates")
    return rows


def local_source_path(manifest: str, candidate: str) -> str:
    return str(Path(manifest).parent / "mixtures" / f"{candidate}.csv")


def validate_rows(rows: list[dict[str, str]], manifest: str, *, remote_sources: bool) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for row in rows:
        source_csv = row["source_csv"] if remote_sources else local_source_path(manifest, row["candidate"])
        config = PanelTrainingConfig(
            experiment_name=DEFAULT_EXPERIMENT_NAME,
            analysis_output_path="",
            candidate=row["candidate"],
            source_csv=source_csv,
            objective=row["objective"],
            family=row["family"],
            phase_information_budget=float(row["phase_information_budget"]),
            expected_max_simulated_epoch=float(row["max_simulated_epoch"]),
            target_flops=TARGET_FLOPS,
            tpu_type="v5p-8",
            tpu_region=base.DEFAULT_TPU_REGION,
            tpu_zone=base.DEFAULT_TPU_ZONE,
            batch_size=1,
            output_path="",
            run_id=0,
            run_name=row["candidate"],
            data_seed=PANEL_DATA_SEED,
        )
        key, source = panel_source(config)
        phase_weights, mixture_diagnostics = base._read_phase_weights(source)
        base._validate_runtime_phase_weights(phase_weights, run_name=key.value)
        diagnostics.append(asdict(mixture_diagnostics))
    return diagnostics


def write_local_dry_run(
    rows: list[dict[str, str]],
    diagnostics: list[dict[str, object]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_candidates.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (output_dir / "mixture_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    summary = {
        "candidate_count": len(rows),
        "objectives": sorted({row["objective"] for row in rows}),
        "families": sorted({row["family"] for row in rows}),
        "target_flops": TARGET_FLOPS,
        "data_seed": PANEL_DATA_SEED,
        "repeat_count_per_candidate": 1,
        "max_concurrent": args.max_concurrent,
        "native_table9_eval_for_every_candidate": True,
    }
    (output_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def build_steps(
    rows: list[dict[str, str]],
    *,
    analysis_output_path: str,
    experiment_name: str,
    max_concurrent: int,
) -> list[ExecutorStep]:
    tpu_type, batch_size = base.TARGET_BUDGETS[TARGET_FLOPS]
    scaling_fits = base._read_scaling_fits(analysis_output_path)
    scale_candidate = base._candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=TARGET_FLOPS,
        batch_size=batch_size,
    )
    steps: list[ExecutorStep] = [
        ExecutorStep(
            name=f"{experiment_name}/manifest",
            fn=save_panel_manifest,
            config=SavePanelManifestConfig(
                output_path=this_output_path(),
                rows_json=json.dumps(rows, sort_keys=True),
                experiment_name=experiment_name,
                analysis_output_path=analysis_output_path,
                max_concurrent=max_concurrent,
            ),
        )
    ]
    for index, row in enumerate(rows):
        run_name = f"{row['candidate']}_{base._slug(TARGET_FLOPS)}"
        training_step = ExecutorStep(
            name=f"{experiment_name}/{run_name}",
            fn=run_panel_training,
            resources=ResourceConfig.with_tpu(
                tpu_type,
                regions=[base.DEFAULT_TPU_REGION],
                zone=base.DEFAULT_TPU_ZONE,
            ),
            config=PanelTrainingConfig(
                experiment_name=experiment_name,
                analysis_output_path=analysis_output_path,
                candidate=row["candidate"],
                source_csv=row["source_csv"],
                objective=row["objective"],
                family=row["family"],
                phase_information_budget=float(row["phase_information_budget"]),
                expected_max_simulated_epoch=float(row["max_simulated_epoch"]),
                target_flops=TARGET_FLOPS,
                tpu_type=tpu_type,
                tpu_region=base.DEFAULT_TPU_REGION,
                tpu_zone=base.DEFAULT_TPU_ZONE,
                batch_size=batch_size,
                output_path=this_output_path(),
                run_id=RUN_ID_BASE + index,
                run_name=run_name,
                data_seed=PANEL_DATA_SEED,
            ),
        )
        eval_step = olmo_base_eval_step(
            name=f"t9_{run_name}",
            checkpoint=training_step / f"hf/step-{scale_candidate.train_steps - 1}",
            request_set_dir=base.TABLE9_REQUEST_SET_DIR,
            resource_config=base.TABLE9_EVAL_RESOURCES,
            wandb_group="olmo_base_eval_table9_decoupled_phase_information",
            provenance={
                "evaluator": "marin-native-table9-bpb",
                "panel": "delphi_decoupled_phase_information",
                "scale": base._slug(TARGET_FLOPS),
                "source_run_name": run_name,
                "mixture": row["candidate"],
                "method": f"decoupled-phase-information-{row['family']}",
            },
        )
        steps.extend([training_step, eval_step])
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--local-artifact-dir", type=Path, default=DEFAULT_LOCAL_ARTIFACT_DIR)
    parser.add_argument("--max-concurrent", type=int, default=MAX_ALLOWED_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-remote-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not 1 <= args.max_concurrent <= MAX_ALLOWED_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {MAX_ALLOWED_CONCURRENT}]")
    if args.dry_run and args.validate_remote_only:
        raise ValueError("--dry-run and --validate-remote-only are mutually exclusive")
    expected_prefix = marin_prefix_for_region(base.DEFAULT_TPU_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    rows = read_manifest(args.manifest)
    diagnostics = validate_rows(rows, args.manifest, remote_sources=args.validate_remote_only or not args.dry_run)
    if args.dry_run or args.validate_remote_only:
        write_local_dry_run(rows, diagnostics, args.local_artifact_dir, args)
        source_kind = "remote" if args.validate_remote_only else "local"
        logger.info(
            "Validated %d candidates from %s sources and wrote artifacts to %s",
            len(rows),
            source_kind,
            args.local_artifact_dir,
        )
        return

    steps = build_steps(
        rows,
        analysis_output_path=args.analysis_output_path,
        experiment_name=args.experiment_name,
        max_concurrent=args.max_concurrent,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{args.experiment_name}: fixed-aggregate phase-information validation at 3e18",
    )


if __name__ == "__main__":
    main()
