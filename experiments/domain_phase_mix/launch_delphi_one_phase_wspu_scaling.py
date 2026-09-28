# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale the frozen 3e18 WSPU optima at their OLMix comparator seeds.

Both one-phase mixtures use cap 6, the independently selected winner in the
original epoch-cap sweep. The frozen model, token, batch, and seed ladder runs
on v6e because the matching v5p capacity is degraded.
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
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.defaults import default_validation_sets
from experiments.domain_phase_mix import launch_delphi_baseline_mixtures as scaling
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_one_phase_weibull_softplus_epoch_cap_sweep_3e18 as wspu
from experiments.domain_phase_mix import launch_delphi_uncheatable_optimized_mixtures as base
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_one_phase_wspu_scaling_v6e_20260906"
LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_one_phase_wspu_scaling_v6e_20260906"
    / "launch_dry_run"
)
TRAIN_TPU_REGION = "us-east5"
TRAIN_TPU_ZONE = "us-east5-b"
RUN_ID_BASE = 7_371_000
EXPECTED_RUN_COUNT = 8
MAX_CONCURRENT = EXPECTED_RUN_COUNT
UNCHEATABLE_TARGET_METRIC = "eval/uncheatable_eval/bpb"
WANDB_SERIES_TAG = "delphi-one-phase-wspu-scaling"
FP32_ADAMH_PERSISTENT_BYTES_PER_PARAMETER = 12

TARGET_BUDGETS: dict[float, tuple[str, int]] = {
    3e18: ("v6e-8", 128),
    2e19: ("v6e-16", 128),
    3e20: ("v6e-32", 256),
    1e21: ("v6e-64", 512),
}
V6E_DEVICE_COUNTS = {"v6e-8": 8, "v6e-16": 16, "v6e-32": 32, "v6e-64": 64}

POLICY_CANDIDATES = {
    "uncheatable": "wspu_uncheatable_cap06",
    "table9": "wspu_table9_cap06",
}

# These are the data seeds of the exact OLMix series used in the paper-facing
# scaling comparison, not a newly generated seed sequence.
OLMIX_COMPARATOR_SEEDS: dict[str, dict[float, int]] = {
    "uncheatable": {3e18: 666_200, 2e19: 666_202, 3e20: 666_204, 1e21: 666_206},
    "table9": {3e18: 662_009, 2e19: 662_001, 3e20: 662_003, 1e21: 662_005},
}
OLMIX_COMPARATOR_SERIES = {
    "uncheatable": "olmix_onephase_uncheatable_d001_kl005_cap4",
    "table9": "olmix_onephase_table9_d001_kl0p005_cap4",
}


class ScalingMixtureKey(str):
    """String key accepted by the shared optimized-mixture trainer."""

    @property
    def value(self) -> str:
        return str(self)


@dataclass(frozen=True)
class FrozenPolicy:
    """One selected 3e18 WSPU policy and its embedded runtime weights."""

    candidate_id: str
    target: str
    target_metric: str
    epoch_cap: int
    max_materialized_epoch: float
    weights_csv: str
    weights_sha256: str


@dataclass(frozen=True)
class ScalingTrainingConfig:
    """Primitive-only configuration for one scaling child."""

    experiment_name: str
    analysis_output_path: str
    candidate_id: str
    target: str
    target_metric: str
    epoch_cap: int
    max_materialized_epoch: float
    weights_csv: str
    weights_sha256: str
    target_flops: float
    tpu_type: str
    batch_size: int
    output_path: str
    run_id: int
    run_name: str
    data_seed: int
    validation_configs: dict[str, DatasetComponent] | None


@dataclass(frozen=True)
class SaveScalingManifestConfig:
    """Persist the frozen policies and resolved ladder rows."""

    output_path: str
    analysis_output_path: str
    candidate_weights_path: str
    candidate_weights_sha256: str
    rows_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved manifest, training, and Table-9 evaluation graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def _weight_csv(candidate: sweep.CandidateMixture) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["domain", "phase_0_weight", "phase_1_weight", "simulated_epochs"],
    )
    writer.writeheader()
    for domain in base.DOMAIN_NAMES:
        weight = candidate.weights[domain]
        writer.writerow(
            {
                "domain": domain,
                "phase_0_weight": weight,
                "phase_1_weight": weight,
                "simulated_epochs": base.SIMULATED_EPOCH_TARGET_BUDGET * weight / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain],
            }
        )
    return buffer.getvalue()


def selected_policies() -> tuple[FrozenPolicy, ...]:
    """Load the SHA-pinned candidate table and return the two frozen policies."""
    candidates, _ = sweep.load_candidate_mixtures(
        wspu.DEFAULT_CANDIDATE_WEIGHTS,
        wspu.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=wspu.SWEEP_DEFINITION,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    policies = []
    for target, candidate_id in POLICY_CANDIDATES.items():
        candidate = by_id[candidate_id]
        weights_csv = _weight_csv(candidate)
        policies.append(
            FrozenPolicy(
                candidate_id=candidate_id,
                target=target,
                target_metric=base.TABLE9_TARGET_METRIC if target == "table9" else UNCHEATABLE_TARGET_METRIC,
                epoch_cap=candidate.epoch_cap,
                max_materialized_epoch=candidate.max_materialized_epoch,
                weights_csv=weights_csv,
                weights_sha256=hashlib.sha256(weights_csv.encode()).hexdigest(),
            )
        )
    return tuple(policies)


def _register_policy(config: ScalingTrainingConfig) -> tuple[ScalingMixtureKey, base.MixtureSource]:
    key = ScalingMixtureKey(config.candidate_id)
    typed_key = cast(base.DelphiValidationMixture, key)
    source = base.MixtureSource(
        key=typed_key,
        display_name=f"WSPU {config.target} cap {config.epoch_cap}",
        source_csv=f"embedded://{config.weights_sha256}/{config.candidate_id}.csv",
        github_issue=6611 if config.target == "table9" else 6609,
        target_metric=config.target_metric,
        method="weibull_softplus_unscaled",
        wandb_series_tag=WANDB_SERIES_TAG,
        expected_max_simulated_epoch=config.max_materialized_epoch,
        data_seed_override=config.data_seed,
    )
    base.MIXTURE_SOURCES[typed_key] = source
    base._EMBEDDED_MIXTURE_WEIGHT_CSVS[typed_key] = config.weights_csv
    return key, source


def run_scaling_training(config: ScalingTrainingConfig) -> None:
    """Register the embedded policy and delegate to the shared Delphi trainer."""
    if hashlib.sha256(config.weights_csv.encode()).hexdigest() != config.weights_sha256:
        raise ValueError(f"Embedded weights changed for {config.candidate_id}")
    key, _ = _register_policy(config)
    base.EXPERIMENT_NAME = config.experiment_name
    base.run_delphi_optimized_training(
        base.DelphiOptimizedTrainingConfig(
            analysis_output_path=config.analysis_output_path,
            target_flops=config.target_flops,
            tpu_type=config.tpu_type,
            tpu_region=TRAIN_TPU_REGION,
            tpu_zone=TRAIN_TPU_ZONE,
            batch_size=config.batch_size,
            mixture=cast(base.DelphiValidationMixture, key),
            label=base.LABEL,
            output_path=config.output_path,
            run_id=config.run_id,
            run_name=config.run_name,
            data_seed=config.data_seed,
            train_tokens_override=None,
            trainer_seed=0,
            validation_configs=config.validation_configs,
        )
    )


def planned_rows(analysis_output_path: str) -> list[dict[str, object]]:
    """Resolve the frozen eight-run ladder against the canonical scaling fit."""
    canonical_batches = {target_flops: batch_size for target_flops, (_, batch_size) in scaling.TARGET_BUDGETS.items()}
    selected_batches = {target_flops: batch_size for target_flops, (_, batch_size) in TARGET_BUDGETS.items()}
    if selected_batches != canonical_batches:
        raise ValueError(f"Scaling budgets or batches changed: {canonical_batches} != frozen {selected_batches}")
    scaling_fits = base._read_scaling_fits(analysis_output_path)
    policies = selected_policies()
    rows: list[dict[str, object]] = []
    for target_flops, (tpu_type, batch_size) in TARGET_BUDGETS.items():
        candidate = base._candidate_for_budget(
            scaling_fits=scaling_fits,
            target_flops=target_flops,
            batch_size=batch_size,
        )
        non_embedding_params = int(candidate.model_config.total_trainable_params(0))
        total_params = int(candidate.model_config.total_trainable_params(scaling.completed_adamh_heuristic.vocab_size))
        realized_train_tokens = candidate.train_steps * batch_size * base.SEQ_LEN_DELPHI
        device_count = V6E_DEVICE_COUNTS[tpu_type]
        if batch_size % device_count != 0:
            raise ValueError(f"Batch {batch_size} is not divisible by {device_count} devices on {tpu_type}")
        if candidate.model_config.hidden_dim % device_count != 0:
            raise ValueError(
                f"Hidden dim {candidate.model_config.hidden_dim} cannot use the validated data-axis sharding "
                f"on {tpu_type}"
            )
        persistent_state_bytes = total_params * FP32_ADAMH_PERSISTENT_BYTES_PER_PARAMETER
        persistent_state_gib_per_device = persistent_state_bytes / device_count / 2**30
        for policy in policies:
            run_order = len(rows)
            data_seed = OLMIX_COMPARATOR_SEEDS[policy.target][target_flops]
            run_name = f"{policy.candidate_id}_{base._slug(target_flops)}_seed{data_seed}"
            rows.append(
                {
                    "run_order": run_order,
                    "run_id": RUN_ID_BASE + run_order,
                    "run_name": run_name,
                    "candidate_id": policy.candidate_id,
                    "target": policy.target,
                    "target_metric": policy.target_metric,
                    "epoch_cap": policy.epoch_cap,
                    "max_materialized_epoch": policy.max_materialized_epoch,
                    "weights_csv": policy.weights_csv,
                    "weights_sha256": policy.weights_sha256,
                    "target_flops": target_flops,
                    "tpu_type": tpu_type,
                    "tpu_region": TRAIN_TPU_REGION,
                    "tpu_zone": TRAIN_TPU_ZONE,
                    "batch_size": batch_size,
                    "device_count": device_count,
                    "examples_per_device": batch_size // device_count,
                    "estimated_persistent_state_gib_per_device": persistent_state_gib_per_device,
                    "train_steps": candidate.train_steps,
                    "realized_train_tokens": realized_train_tokens,
                    "expected_checkpoint_step": candidate.train_steps - 1,
                    "model_hidden_dim": int(candidate.model_config.hidden_dim),
                    "model_layers": int(candidate.model_config.num_layers),
                    "non_embedding_params": non_embedding_params,
                    "total_trainable_params": total_params,
                    "tensor_parallel_size": base._tensor_parallel_size(candidate.model_config.hidden_dim, tpu_type),
                    "data_seed": data_seed,
                    "trainer_seed": 0,
                    "olmix_comparator_series": OLMIX_COMPARATOR_SERIES[policy.target],
                }
            )
    if len(rows) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} scaling rows, found {len(rows)}")
    return rows


def save_scaling_manifest(config: SaveScalingManifestConfig) -> None:
    """Write exact run and policy provenance beside the executor graph."""
    rows = json.loads(config.rows_json)
    manifest_rows = [{key: value for key, value in row.items() if key != "weights_csv"} for row in rows]
    policy_weights = {row["candidate_id"]: row["weights_csv"] for row in rows}
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_manifest.json"), "w") as handle:
        json.dump(manifest_rows, handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "policy_weights.json"), "w") as handle:
        json.dump(policy_weights, handle, indent=2, sort_keys=True)
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "analysis_output_path": config.analysis_output_path,
        "candidate_weights_path": config.candidate_weights_path,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "source_selection": {
            "uncheatable": "cap 6, the observed winner in the original frozen sweep",
            "table9": "cap 6, the observed winner in the original frozen sweep",
        },
        "target_flops": list(TARGET_BUDGETS),
        "run_count": len(rows),
        "max_concurrent": MAX_CONCURRENT,
        "trainer_seed": 0,
        "native_table9_eval_for_every_run": True,
        "inline_uncheatable_eval_for_every_run": True,
        "canonical_model_and_batch_ladder": True,
        "olmix_comparator_seed_matched": True,
        "accelerator_generation_matched_to_olmix": False,
        "v6e_capacity_migration": True,
        "exact_3e18_v6e_configuration_previously_succeeded": True,
        "training_region": TRAIN_TPU_REGION,
        "training_zone": TRAIN_TPU_ZONE,
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _training_config(
    row: dict[str, object],
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent] | None,
) -> ScalingTrainingConfig:
    return ScalingTrainingConfig(
        experiment_name=EXPERIMENT_NAME,
        analysis_output_path=analysis_output_path,
        candidate_id=str(row["candidate_id"]),
        target=str(row["target"]),
        target_metric=str(row["target_metric"]),
        epoch_cap=int(row["epoch_cap"]),
        max_materialized_epoch=float(row["max_materialized_epoch"]),
        weights_csv=str(row["weights_csv"]),
        weights_sha256=str(row["weights_sha256"]),
        target_flops=float(row["target_flops"]),
        tpu_type=str(row["tpu_type"]),
        batch_size=int(row["batch_size"]),
        output_path=this_output_path(),
        run_id=int(row["run_id"]),
        run_name=str(row["run_name"]),
        data_seed=int(row["data_seed"]),
        validation_configs=validation_configs,
    )


def build_launch_artifacts(
    rows: list[dict[str, object]],
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent] | None,
) -> LaunchArtifacts:
    """Build independent trainings and one native Table-9 eval per checkpoint."""
    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_scaling_manifest,
        config=SaveScalingManifestConfig(
            output_path=this_output_path(),
            analysis_output_path=analysis_output_path,
            candidate_weights_path=str(wspu.DEFAULT_CANDIDATE_WEIGHTS),
            candidate_weights_sha256=wspu.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
            rows_json=json.dumps(rows, sort_keys=True),
        ),
    )
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for row in rows:
        run_name = str(row["run_name"])
        resources = ResourceConfig.with_tpu(
            str(row["tpu_type"]),
            regions=[TRAIN_TPU_REGION],
            zone=TRAIN_TPU_ZONE,
        )
        training_step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}/{run_name}",
            fn=remote(
                run_scaling_training,
                resources=resources,
                env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
            resources=resources,
            config=_training_config(row, analysis_output_path, validation_configs),
        )
        eval_steps.append(
            olmo_base_eval_step(
                name=(
                    f"t9_w{'u' if row['target'] == 'uncheatable' else 't'}_"
                    f"{base._slug(float(row['target_flops']))}_s{row['data_seed']}"
                ),
                checkpoint=training_step / f"hf/step-{int(row['expected_checkpoint_step'])}",
                request_set_dir=base.TABLE9_REQUEST_SET_DIR,
                resource_config=base.TABLE9_EVAL_RESOURCES,
                wandb_group="olmo_base_eval_table9_delphi_one_phase_wspu_scaling",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_one_phase_wspu_scaling",
                    "scale": base._slug(float(row["target_flops"])),
                    "source_run_name": run_name,
                    "mixture": str(row["candidate_id"]),
                    "method": "weibull-softplus-unscaled",
                    "data_seed": str(row["data_seed"]),
                    "olmix_comparator_series": str(row["olmix_comparator_series"]),
                },
            )
        )
        training_steps.append(training_step)
    return LaunchArtifacts(
        manifest_step=manifest_step,
        training_steps=training_steps,
        eval_steps=eval_steps,
    )


def _write_local_dry_run(rows: list[dict[str, object]], analysis_output_path: str) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_scaling_manifest(
        SaveScalingManifestConfig(
            output_path=str(LOCAL_ARTIFACT_DIR),
            analysis_output_path=analysis_output_path,
            candidate_weights_path=str(wspu.DEFAULT_CANDIDATE_WEIGHTS),
            candidate_weights_sha256=wspu.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
            rows_json=json.dumps(rows, sort_keys=True),
        )
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    expected_prefix = marin_prefix_for_region(TRAIN_TPU_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    rows = planned_rows(args.analysis_output_path)
    if args.dry_run:
        _write_local_dry_run(rows, args.analysis_output_path)
        logger.info("Wrote %d frozen WSPU scaling rows under %s", len(rows), LOCAL_ARTIFACT_DIR)
        return

    with executor_context():
        validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
        validation_configs = {
            name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
        }
        artifacts = build_launch_artifacts(rows, args.analysis_output_path, validation_configs)
    if os.getenv("CI") is not None:
        logger.info("Built eight WSPU scaling trainings and evaluations; skipping launch in CI")
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=MAX_CONCURRENT),
        steps=artifacts.steps,
        description=("Delphi one-phase WSPU v6e scaling: cap-6 Uncheatable and Table 9 at matched OLMix data seeds"),
    )


if __name__ == "__main__":
    main()
