# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep tokens per parameter for fixed-architecture Uncheatable phase policies.

The 3e18 Delphi architecture is held fixed while the token horizon varies. Each
row starts from scratch with an 80/20 WSD schedule and token-aware AdamH
hyperparameters recomputed for its realized token budget.
"""

from __future__ import annotations

import argparse
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
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
DEFAULT_LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUT_DIR / "delphi_fixed_n_tpp_phase_sweep_20260712"
DEFAULT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_fixed_n_tpp_phase_sweep_20260712"
TARGET_ARCHITECTURE_FLOPS = 3e18
DEFAULT_TOKENS_PER_PARAMETER = (10.0, 20.0)
PANEL_DATA_SEED = 714_000
RUN_ID_BASE = 714_000
MAX_ALLOWED_CONCURRENT = 56
DEFAULT_MAX_CONCURRENT = 12
WANDB_SERIES_TAG = "delphi-fixed-n-tpp-phase-sweep"
GITHUB_ISSUE = 6611
UNCHEATABLE_TARGET_METRIC = "eval/uncheatable_eval/bpb"
DECOUPLED_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_validation_20260712/mixtures"
)
PROPORTIONAL_MIXTURE_GCS = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_baseline_noise_validation_mixtures_20260703/mixtures/proportional.csv"
)


@dataclass(frozen=True)
class PolicySpec:
    """One fixed phase policy in the TPP sweep."""

    key: str
    display_name: str
    source_csv: str
    method: str
    family: str
    phase_information_budget: float | None
    expected_max_simulated_epoch: float


POLICIES = (
    PolicySpec(
        key="unch_effexp_e0p005",
        display_name="Uncheatable effective-exposure DSP, epsilon_phase=0.005",
        source_csv=f"{DECOUPLED_MIXTURE_GCS_DIR}/dphase_unch05_eff_e0p005.csv",
        method="effective-exposure-dsp-decoupled-phase-information",
        family="effective-exposure-dsp",
        phase_information_budget=0.005,
        expected_max_simulated_epoch=12.918367,
    ),
    PolicySpec(
        key="unch_effexp_e0p05",
        display_name="Uncheatable effective-exposure DSP, epsilon_phase=0.05",
        source_csv=f"{DECOUPLED_MIXTURE_GCS_DIR}/dphase_unch05_eff_e0p05.csv",
        method="effective-exposure-dsp-decoupled-phase-information",
        family="effective-exposure-dsp",
        phase_information_budget=0.05,
        expected_max_simulated_epoch=12.918367,
    ),
    PolicySpec(
        key="unch_effexp_e0p1",
        display_name="Uncheatable effective-exposure DSP, epsilon_phase=0.1",
        source_csv=f"{DECOUPLED_MIXTURE_GCS_DIR}/dphase_unch05_eff_e0p1.csv",
        method="effective-exposure-dsp-decoupled-phase-information",
        family="effective-exposure-dsp",
        phase_information_budget=0.1,
        expected_max_simulated_epoch=12.918367,
    ),
    PolicySpec(
        key="unch_effexp_e0p2",
        display_name="Uncheatable effective-exposure DSP, epsilon_phase=0.2",
        source_csv=f"{DECOUPLED_MIXTURE_GCS_DIR}/dphase_unch05_eff_e0p2.csv",
        method="effective-exposure-dsp-decoupled-phase-information",
        family="effective-exposure-dsp",
        phase_information_budget=0.2,
        expected_max_simulated_epoch=12.918367,
    ),
    PolicySpec(
        key="unch_effexp_tied",
        display_name="Uncheatable effective-exposure DSP aggregate-matched tied control",
        source_csv=f"{DECOUPLED_MIXTURE_GCS_DIR}/dphase_unch05_tied.csv",
        method="effective-exposure-dsp-aggregate-matched-tied-control",
        family="aggregate-matched-tied-control",
        phase_information_budget=0.0,
        expected_max_simulated_epoch=12.918367,
    ),
    PolicySpec(
        key="proportional",
        display_name="Proportional baseline",
        source_csv=PROPORTIONAL_MIXTURE_GCS,
        method="proportional-baseline",
        family="proportional",
        phase_information_budget=None,
        expected_max_simulated_epoch=0.905353,
    ),
)


class PanelMixtureKey(str):
    """String key with the value interface used by the shared launcher."""

    @property
    def value(self) -> str:
        return str(self)


@dataclass(frozen=True)
class FixedTppTrainingConfig:
    """Primitive-only config for one fixed-architecture training child."""

    experiment_name: str
    analysis_output_path: str
    policy_key: str
    display_name: str
    source_csv: str
    method: str
    family: str
    phase_information_budget: float | None
    expected_max_simulated_epoch: float
    tokens_per_parameter: float
    train_tokens: int
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
    """Persist the exact fixed-N run matrix and launch invariants."""

    output_path: str
    rows_json: str
    experiment_name: str
    analysis_output_path: str
    max_concurrent: int


def panel_source(config: FixedTppTrainingConfig) -> tuple[PanelMixtureKey, base.MixtureSource]:
    key = PanelMixtureKey(config.policy_key)
    typed_key = cast(base.DelphiValidationMixture, key)
    source = base.MixtureSource(
        key=typed_key,
        display_name=config.display_name,
        source_csv=config.source_csv,
        github_issue=GITHUB_ISSUE,
        target_metric=UNCHEATABLE_TARGET_METRIC,
        method=config.method,
        wandb_series_tag=WANDB_SERIES_TAG,
        expected_max_simulated_epoch=config.expected_max_simulated_epoch,
        data_seed_override=config.data_seed,
    )
    return key, source


def _configured_validation_sets():
    validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
    return {name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()}


def run_fixed_tpp_training(config: FixedTppTrainingConfig) -> None:
    """Register one policy and delegate to the shared fixed-model trainer."""
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
            train_tokens_override=config.train_tokens,
            trainer_seed=0,
            validation_configs=_configured_validation_sets(),
        )
    )


def save_panel_manifest(config: SavePanelManifestConfig) -> None:
    """Write exact run provenance beside the executor graph."""
    rows = json.loads(config.rows_json)
    summary = {
        "experiment_name": config.experiment_name,
        "analysis_output_path": config.analysis_output_path,
        "architecture_target_flops": TARGET_ARCHITECTURE_FLOPS,
        "data_seed": PANEL_DATA_SEED,
        "trainer_seed": 0,
        "phase_fractions": dict(base.PHASE_FRACTIONS),
        "simulated_epoch_target_budget": base.SIMULATED_EPOCH_TARGET_BUDGET,
        "max_concurrent": config.max_concurrent,
        "run_count": len(rows),
        "policy_count": len(POLICIES),
        "native_table9_eval_for_every_run": True,
        "primary_metric": UNCHEATABLE_TARGET_METRIC,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_manifest.json"), "w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def planned_rows(
    *,
    analysis_output_path: str,
    tokens_per_parameter: tuple[float, ...],
) -> list[dict[str, object]]:
    tpu_type, batch_size = base.TARGET_BUDGETS[TARGET_ARCHITECTURE_FLOPS]
    scaling_fits = base._read_scaling_fits(analysis_output_path)
    scale_candidate = base._candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=TARGET_ARCHITECTURE_FLOPS,
        batch_size=batch_size,
    )
    params = int(scale_candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size))
    rows: list[dict[str, object]] = []
    for tpp in tokens_per_parameter:
        requested_train_tokens = round(params * tpp)
        train_steps = round(requested_train_tokens / (batch_size * base.SEQ_LEN_DELPHI))
        realized_train_tokens = train_steps * batch_size * base.SEQ_LEN_DELPHI
        if realized_train_tokens > base.SIMULATED_EPOCH_TARGET_BUDGET:
            raise ValueError(
                f"TPP={tpp:g} realizes {realized_train_tokens} tokens, exceeding simulated-epoch target "
                f"budget {base.SIMULATED_EPOCH_TARGET_BUDGET}"
            )
        optimizer = completed_adamh_heuristic.build_optimizer_config(batch_size, realized_train_tokens)
        for policy in POLICIES:
            run_name = f"{policy.key}_tpp{tpp:g}"
            rows.append(
                {
                    "run_name": run_name,
                    "policy_key": policy.key,
                    "display_name": policy.display_name,
                    "source_csv": policy.source_csv,
                    "method": policy.method,
                    "family": policy.family,
                    "phase_information_budget": policy.phase_information_budget,
                    "expected_max_simulated_epoch": policy.expected_max_simulated_epoch,
                    "tokens_per_parameter_target": tpp,
                    "tokens_per_parameter_realized": realized_train_tokens / params,
                    "requested_train_tokens": requested_train_tokens,
                    "realized_train_tokens": realized_train_tokens,
                    "train_steps": train_steps,
                    "expected_checkpoint_step": train_steps - 1,
                    "architecture_target_flops": TARGET_ARCHITECTURE_FLOPS,
                    "actual_approximate_flops": 6 * params * realized_train_tokens,
                    "total_trainable_params": params,
                    "model_hidden_dim": int(scale_candidate.model_config.hidden_dim),
                    "model_layers": int(scale_candidate.model_config.num_layers),
                    "batch_size": batch_size,
                    "sequence_length": base.SEQ_LEN_DELPHI,
                    "tpu_type": tpu_type,
                    "data_seed": PANEL_DATA_SEED,
                    "trainer_seed": 0,
                    "optimizer": asdict(optimizer),
                    "phase_fractions": dict(base.PHASE_FRACTIONS),
                }
            )
    return rows


def validate_policy_sources() -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    aggregate_weights: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        config = FixedTppTrainingConfig(
            experiment_name=DEFAULT_EXPERIMENT_NAME,
            analysis_output_path="",
            policy_key=policy.key,
            display_name=policy.display_name,
            source_csv=policy.source_csv,
            method=policy.method,
            family=policy.family,
            phase_information_budget=policy.phase_information_budget,
            expected_max_simulated_epoch=policy.expected_max_simulated_epoch,
            tokens_per_parameter=0.0,
            train_tokens=1,
            target_flops=TARGET_ARCHITECTURE_FLOPS,
            tpu_type="v5p-8",
            tpu_region=base.DEFAULT_TPU_REGION,
            tpu_zone=base.DEFAULT_TPU_ZONE,
            batch_size=1,
            output_path="",
            run_id=0,
            run_name=policy.key,
            data_seed=PANEL_DATA_SEED,
        )
        key, source = panel_source(config)
        phase_weights, mixture_diagnostics = base._read_phase_weights(source)
        base._validate_runtime_phase_weights(phase_weights, run_name=key.value)
        aggregate_weights[policy.key] = {
            domain: sum(base.PHASE_FRACTIONS[phase] * phase_weights[phase][domain] for phase in base.PHASE_NAMES)
            for domain in base.DOMAIN_NAMES
        }
        diagnostics.append({"policy_key": policy.key, **asdict(mixture_diagnostics)})

    tied = aggregate_weights["unch_effexp_tied"]
    for policy in POLICIES:
        if policy.family not in {"effective-exposure-dsp", "aggregate-matched-tied-control"}:
            continue
        max_difference = max(abs(aggregate_weights[policy.key][domain] - tied[domain]) for domain in base.DOMAIN_NAMES)
        if max_difference > 1e-9:
            raise ValueError(
                f"{policy.key} is not aggregate-matched to unch_effexp_tied: max difference={max_difference}"
            )
    return diagnostics


def write_local_dry_run(
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (output_dir / "mixture_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    summary = {
        "architecture_target_flops": TARGET_ARCHITECTURE_FLOPS,
        "tokens_per_parameter": list(args.tokens_per_parameter),
        "run_count": len(rows),
        "policy_count": len(POLICIES),
        "data_seed": PANEL_DATA_SEED,
        "phase_fractions": dict(base.PHASE_FRACTIONS),
        "primary_metric": UNCHEATABLE_TARGET_METRIC,
        "native_table9_eval_for_every_run": True,
        "max_concurrent": args.max_concurrent,
    }
    (output_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def build_steps(
    rows: list[dict[str, object]],
    *,
    analysis_output_path: str,
    experiment_name: str,
    max_concurrent: int,
) -> list[ExecutorStep]:
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
        run_name = str(row["run_name"])
        training_step = ExecutorStep(
            name=f"{experiment_name}/{run_name}",
            fn=run_fixed_tpp_training,
            resources=ResourceConfig.with_tpu(
                str(row["tpu_type"]),
                regions=[base.DEFAULT_TPU_REGION],
                zone=base.DEFAULT_TPU_ZONE,
            ),
            config=FixedTppTrainingConfig(
                experiment_name=experiment_name,
                analysis_output_path=analysis_output_path,
                policy_key=str(row["policy_key"]),
                display_name=str(row["display_name"]),
                source_csv=str(row["source_csv"]),
                method=str(row["method"]),
                family=str(row["family"]),
                phase_information_budget=(
                    None if row["phase_information_budget"] is None else float(row["phase_information_budget"])
                ),
                expected_max_simulated_epoch=float(row["expected_max_simulated_epoch"]),
                tokens_per_parameter=float(row["tokens_per_parameter_target"]),
                train_tokens=int(row["realized_train_tokens"]),
                target_flops=TARGET_ARCHITECTURE_FLOPS,
                tpu_type=str(row["tpu_type"]),
                tpu_region=base.DEFAULT_TPU_REGION,
                tpu_zone=base.DEFAULT_TPU_ZONE,
                batch_size=int(row["batch_size"]),
                output_path=this_output_path(),
                run_id=RUN_ID_BASE + index,
                run_name=run_name,
                data_seed=PANEL_DATA_SEED,
            ),
        )
        eval_step = olmo_base_eval_step(
            name=f"t9_{run_name}",
            checkpoint=training_step / f"hf/step-{int(row['expected_checkpoint_step'])}",
            request_set_dir=base.TABLE9_REQUEST_SET_DIR,
            resource_config=base.TABLE9_EVAL_RESOURCES,
            wandb_group="olmo_base_eval_table9_fixed_n_tpp_phase_sweep",
            provenance={
                "evaluator": "marin-native-table9-bpb",
                "panel": "delphi_fixed_n_tpp_phase_sweep",
                "source_run_name": run_name,
                "policy": str(row["policy_key"]),
                "method": str(row["method"]),
                "tokens_per_parameter": str(row["tokens_per_parameter_realized"]),
                "primary_metric": UNCHEATABLE_TARGET_METRIC,
            },
        )
        steps.extend([training_step, eval_step])
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--local-artifact-dir", type=Path, default=DEFAULT_LOCAL_ARTIFACT_DIR)
    parser.add_argument(
        "--tokens-per-parameter",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOKENS_PER_PARAMETER),
    )
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not 1 <= args.max_concurrent <= MAX_ALLOWED_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {MAX_ALLOWED_CONCURRENT}]")
    tokens_per_parameter = tuple(float(value) for value in args.tokens_per_parameter)
    if len(tokens_per_parameter) != len(set(tokens_per_parameter)):
        raise ValueError("--tokens-per-parameter values must be unique")
    if any(value <= 0 for value in tokens_per_parameter):
        raise ValueError("--tokens-per-parameter values must be positive")

    expected_prefix = marin_prefix_for_region(base.DEFAULT_TPU_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    rows = planned_rows(
        analysis_output_path=args.analysis_output_path,
        tokens_per_parameter=tokens_per_parameter,
    )
    diagnostics = validate_policy_sources()
    if args.dry_run:
        write_local_dry_run(rows, diagnostics, args.local_artifact_dir, args)
        logger.info(
            "Validated %d policies and wrote %d planned runs to %s", len(POLICIES), len(rows), args.local_artifact_dir
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
        description=(
            f"{args.experiment_name}: fixed 3e18 Delphi architecture at TPP "
            f"{', '.join(f'{value:g}' for value in tokens_per_parameter)}"
        ),
    )


if __name__ == "__main__":
    main()
