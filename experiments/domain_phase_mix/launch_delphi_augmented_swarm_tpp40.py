# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the exact 280-row Delphi fit swarm at total-parameter TPP 40.

The model architecture and 80/20 WSD schedule are the same as the Delphi 3e18
fit swarm. Only the training horizon changes. The token-aware AdamH schedule is
recomputed for the longer horizon, while source coordinates and random seeds
remain matched to the original 280-row panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, replace

from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

TOTAL_TOKENS_PER_PARAMETER = 40.0
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_20260803"
LOCAL_ARTIFACT_DIR = base.REFERENCE_OUTPUT_DIR / "delphi_augmented_swarm_tpp40_20260803" / "launch_dry_run"
DEFAULT_MAX_CONCURRENT = 56
STEPS_PER_EVAL = 5000
PERMANENT_CHECKPOINT_INTERVAL = None
HORIZON_FIELDS = frozenset(
    {
        "target_flops",
        "train_steps",
        "realized_train_tokens",
        "expected_checkpoint_step",
    }
)


def _panel_hash(run_specs: list[base.DelphiSwarmRunSpec]) -> str:
    payload = [
        {
            "source_run_name": spec.source_run_name,
            "panel_source": spec.panel_source,
            "phase_weights": spec.phase_weights,
        }
        for spec in run_specs
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _fixed_identity_hash(run_specs: list[base.DelphiSwarmRunSpec]) -> str:
    payload = []
    for spec in run_specs:
        row = asdict(spec)
        for field in HORIZON_FIELDS:
            row.pop(field)
        payload.append(row)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_run_specs(
    *,
    source_panel: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[base.DelphiSwarmRunSpec], dict[str, object]]:
    """Resolve the fixed architecture and replace only its training horizon."""
    source_specs = base.load_source_panel(
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )
    total_params = {spec.total_trainable_params for spec in source_specs}
    if len(total_params) != 1:
        raise ValueError(f"Source panel has inconsistent parameter counts: {sorted(total_params)}")
    total_trainable_params = total_params.pop()
    requested_train_tokens = round(total_trainable_params * TOTAL_TOKENS_PER_PARAMETER)
    tokens_per_step = base.TARGET_BATCH_SIZE * base.SEQ_LEN_DELPHI
    train_steps = round(requested_train_tokens / tokens_per_step)
    realized_train_tokens = train_steps * tokens_per_step
    if realized_train_tokens > base.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError(
            f"TPP-40 realizes {realized_train_tokens} tokens, exceeding simulated-epoch target "
            f"budget {base.SIMULATED_EPOCH_TARGET_BUDGET}"
        )
    realized_tpp = realized_train_tokens / total_trainable_params
    actual_approximate_flops = 6 * total_trainable_params * realized_train_tokens

    run_specs = [
        replace(
            spec,
            target_flops=actual_approximate_flops,
            train_steps=train_steps,
            realized_train_tokens=realized_train_tokens,
            expected_checkpoint_step=train_steps - 1,
        )
        for spec in source_specs
    ]
    if _panel_hash(run_specs) != _panel_hash(source_specs):
        raise ValueError("TPP-40 materialization changed a source coordinate")
    source_identity_hash = _fixed_identity_hash(source_specs)
    if _fixed_identity_hash(run_specs) != source_identity_hash:
        raise ValueError("TPP-40 materialization changed a fixed source identity field")
    data_seeds_matched = [spec.data_seed for spec in run_specs] == [spec.data_seed for spec in source_specs]
    if not data_seeds_matched:
        raise ValueError("TPP-40 materialization changed source data seeds")
    late_phase_start_step = base.PHASE_SCHEDULE.phases[1].get_start_step_aligned(
        train_steps,
        base.TARGET_BATCH_SIZE,
        base.MIXTURE_BLOCK_SIZE,
    )
    audit: dict[str, object] = {
        "experiment_name": EXPERIMENT_NAME,
        "source_panel": source_panel,
        "source_panel_sha256": base.SOURCE_PANEL_SHA256,
        "run_count": len(run_specs),
        "source_coordinate_hash": _panel_hash(source_specs),
        "fixed_identity_hash": source_identity_hash,
        "architecture_target_flops": base.TARGET_FLOPS,
        "actual_approximate_flops_per_run": actual_approximate_flops,
        "total_trainable_params": total_trainable_params,
        "non_embedding_params": run_specs[0].non_embedding_params,
        "requested_train_tokens": requested_train_tokens,
        "realized_train_tokens": realized_train_tokens,
        "train_steps": train_steps,
        "expected_checkpoint_step": train_steps - 1,
        "target_total_tokens_per_parameter": TOTAL_TOKENS_PER_PARAMETER,
        "realized_total_tokens_per_parameter": realized_tpp,
        "realized_non_embedding_tokens_per_parameter": realized_train_tokens / run_specs[0].non_embedding_params,
        "phase_fractions": base.PHASE_FRACTIONS,
        "realized_late_phase_start_step": late_phase_start_step,
        "realized_late_phase_start_fraction": late_phase_start_step / train_steps,
        "data_seeds_matched_to_original_swarm": data_seeds_matched,
        "steps_per_eval": STEPS_PER_EVAL,
        "temporary_checkpoint_interval_minutes": 10,
        "permanent_checkpoint_interval": PERMANENT_CHECKPOINT_INTERVAL,
        "native_table9_scheduled": True,
    }
    return run_specs, audit


def _write_local_dry_run(
    *,
    source_panel: str,
    analysis_output_path: str,
    run_specs: list[base.DelphiSwarmRunSpec],
    audit: dict[str, object],
) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    base.save_delphi_swarm_manifest(
        base.SaveDelphiSwarmManifestConfig(
            output_path=str(LOCAL_ARTIFACT_DIR),
            experiment_name=EXPERIMENT_NAME,
            source_panel=source_panel,
            source_panel_sha256=base.SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            architecture_target_flops=base.TARGET_FLOPS,
        )
    )
    (LOCAL_ARTIFACT_DIR / "launch_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=base.DEFAULT_SOURCE_PANEL)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {base.DEFAULT_TPU_REGION}/{base.DEFAULT_TPU_ZONE}")
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")

    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs, audit = build_run_specs(
        source_panel=args.source_panel,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        _write_local_dry_run(
            source_panel=args.source_panel,
            analysis_output_path=args.analysis_output_path,
            run_specs=run_specs,
            audit=audit,
        )
        logger.info("Wrote %d TPP-40 run specs under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = base.build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=args.analysis_output_path,
            source_panel=str(args.source_panel),
            validation_configs=validation_configs,
            experiment_name=EXPERIMENT_NAME,
            architecture_target_flops=base.TARGET_FLOPS,
            wandb_tags=(
                "delphi-tpp40-augmented-swarm",
                "architecture=3e18-selected",
                "total-tpp=40",
                "fit-panel",
                "two-phase",
            ),
            table9_wandb_group="olmo_base_eval_table9_delphi_tpp40_augmented_swarm",
            provenance_panel="delphi_tpp40_augmented_fit_swarm",
            provenance_scale="fixed_n_total_tpp40",
            steps_per_eval=STEPS_PER_EVAL,
            permanent_checkpoint_interval=PERMANENT_CHECKPOINT_INTERVAL,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built TPP-40 graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: exact 280-row augmented fit panel on the 3e18-selected "
            "architecture at total TPP 40 with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
