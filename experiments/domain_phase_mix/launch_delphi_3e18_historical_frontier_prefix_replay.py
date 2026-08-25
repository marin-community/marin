# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the phase-0 state of the observed Delphi 3e18 Uncheatable frontier.

The July frontier run did not retain its phase-boundary checkpoint. This job
reconstructs its phase-0 mixture and original data/trainer seeds, trains only
through update 2400, and retains the full trainer state for a conditional
phase-1 search. The replay runs on v6e-8 in east5b, so it is numerically close
to, but not bitwise identical to, the original v5p-8 trajectory.
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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_historical_frontier_prefix_replay_20260825"
HISTORICAL_FRONTIER_RUN = "dphase_unch05_eff_e0p005_3e18-2cef98"
HISTORICAL_FRONTIER_BPB = 0.9824552536010742
HISTORICAL_MIXTURE_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_validation_20260712/mixtures/dphase_unch05_eff_e0p005.csv"
)
HISTORICAL_MIXTURE_SHA256 = "57a2aa39a5b0e07d40fc6f55f14aaa86327c332e9ef86738b1cca547924c4a59"
HISTORICAL_RUN_ID = 690_300
HISTORICAL_DATA_SEED = 690_300
HISTORICAL_TRAINER_SEED = 0
TPU_TYPE = "v6e-8"
TPU_REGION = "us-east5"
TPU_ZONE = "us-east5-b"
PARENT_ZONE = "us-east5-a"
LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_3e18_historical_frontier_prefix_replay_20260825"
    / "launch_dry_run"
)


def historical_phase_weights(source_bytes: bytes) -> dict[str, dict[str, float]]:
    """Parse and validate the immutable historical two-phase mixture."""
    if hashlib.sha256(source_bytes).hexdigest() != HISTORICAL_MIXTURE_SHA256:
        raise ValueError("Historical frontier mixture bytes changed")
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    by_domain = {row["domain"]: row for row in rows}
    if set(by_domain) != set(base.DOMAIN_NAMES):
        missing = sorted(set(base.DOMAIN_NAMES) - set(by_domain))
        extra = sorted(set(by_domain) - set(base.DOMAIN_NAMES))
        raise ValueError(f"Historical frontier domains changed: {missing=}, {extra=}")
    phase_weights = {
        "phase_0": {domain: float(by_domain[domain]["phase_0_weight"]) for domain in base.DOMAIN_NAMES},
        "phase_1": {domain: float(by_domain[domain]["phase_1_weight"]) for domain in base.DOMAIN_NAMES},
    }
    for phase, weights in phase_weights.items():
        total = sum(weights.values())
        if any(weight < 0.0 for weight in weights.values()) or abs(total - 1.0) > 1e-8:
            raise ValueError(f"Invalid {phase} historical weights: sum={total}")
    return phase_weights


def historical_frontier_spec(
    canonical: base.DelphiSwarmRunSpec,
    phase_weights: dict[str, dict[str, float]],
) -> base.DelphiSwarmRunSpec:
    """Replace a canonical shape template with the historical frontier identity."""
    max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(phase_weights)
    return replace(
        canonical,
        run_order=0,
        run_id=HISTORICAL_RUN_ID,
        run_name="historical_frontier_prefix_seed690300",
        source_run_name=HISTORICAL_FRONTIER_RUN,
        source_experiment="delphi_decoupled_phase_information_validation_3e18_20260712",
        panel_source="historical_frontier_replay",
        tpu_type=TPU_TYPE,
        tpu_region=TPU_REGION,
        tpu_zone=TPU_ZONE,
        tensor_parallel_size=base._tensor_parallel_size(canonical.model_hidden_dim, TPU_TYPE),
        data_seed=HISTORICAL_DATA_SEED,
        trainer_seed=HISTORICAL_TRAINER_SEED,
        max_simulated_epoch=max_epoch,
        q95_simulated_epoch=q95_epoch,
        mean_phase_tv_to_proportional=phase_tv,
        phase_weights=phase_weights,
    )


def load_spec(analysis_output_path: str) -> tuple[base.DelphiSwarmRunSpec, dict[str, Any]]:
    """Load the canonical model shape and replace it with the frontier policy."""
    canonical, canonical_audit = replay.load_replay_specs(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=analysis_output_path,
        tpu_region=base.DEFAULT_TPU_REGION,
        tpu_zone=base.DEFAULT_TPU_ZONE,
    )
    with fsspec.open(HISTORICAL_MIXTURE_URI, "rb") as handle:
        source_bytes = handle.read()
    spec = historical_frontier_spec(canonical[0], historical_phase_weights(source_bytes))
    audit: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "purpose": "recover the missing phase-0 state of the observed 3e18 Uncheatable frontier",
        "historical_frontier_run": HISTORICAL_FRONTIER_RUN,
        "historical_frontier_bpb": HISTORICAL_FRONTIER_BPB,
        "historical_mixture_uri": HISTORICAL_MIXTURE_URI,
        "historical_mixture_sha256": HISTORICAL_MIXTURE_SHA256,
        "historical_run_id": HISTORICAL_RUN_ID,
        "historical_data_seed": HISTORICAL_DATA_SEED,
        "historical_trainer_seed": HISTORICAL_TRAINER_SEED,
        "prefix_train_steps": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "optimizer_schedule_num_train_steps": replay.EXPECTED_FULL_TRAIN_STEPS,
        "prefix_expected_hf_step": replay.EXPECTED_PREFIX_HF_STEP,
        "full_trainer_state_retained": True,
        "hardware": {"tpu_type": TPU_TYPE, "region": TPU_REGION, "zone": TPU_ZONE},
        "hardware_identity": "numerically migrated; not bitwise identical to the original v5p-8 run",
        "v5p_to_v6e_canary_uncheatable_delta_bpb": 5.2809715270996094e-05,
        "canonical_shape_audit": canonical_audit,
    }
    return spec, audit


def build_steps(
    *,
    spec: base.DelphiSwarmRunSpec,
    analysis_output_path: str,
    replay_code_commit: str,
    audit: dict[str, Any],
) -> list[ExecutorStep]:
    """Build one idempotent prefix training step and its immutable manifest."""
    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    resources = ResourceConfig.with_tpu(TPU_TYPE, regions=[TPU_REGION], zone=TPU_ZONE)
    training = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/{spec.run_name}",
        fn=remote(
            replay.run_phase_0_prefix,
            resources=resources,
            env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
        ),
        resources=resources,
        config=replay.PrefixTrainingConfig(
            analysis_output_path=analysis_output_path,
            output_path=this_output_path(),
            run_spec=spec,
            validation_configs=validation_configs,
            prefix_train_steps=replay.EXPECTED_PREFIX_TRAIN_STEPS,
            optimizer_schedule_num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
            replay_code_commit=replay_code_commit,
        ),
    )
    manifest = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest_001_{HISTORICAL_MIXTURE_SHA256[:12]}",
        fn=replay.save_prefix_manifest,
        config=replay.SavePrefixManifestConfig(
            output_path=this_output_path(),
            source_panel=HISTORICAL_MIXTURE_URI,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec)], sort_keys=True),
            launch_audit_json=json.dumps(audit, sort_keys=True),
            replay_code_commit=replay_code_commit,
        ),
    )
    return [manifest, training]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--replay-code-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    expected_prefix = marin_prefix_for_region(TPU_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    replay_code_commit = replay.validate_replay_code_commit(args.replay_code_commit, get_git_commit())
    spec, audit = load_spec(args.analysis_output_path)
    audit["replay_code_commit"] = replay_code_commit
    audit["parent_zone"] = PARENT_ZONE
    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        replay.save_prefix_manifest(
            replay.SavePrefixManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR),
                source_panel=HISTORICAL_MIXTURE_URI,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec)], sort_keys=True),
                launch_audit_json=json.dumps(audit, sort_keys=True),
                replay_code_commit=replay_code_commit,
            )
        )
        logger.info("Wrote historical-frontier prefix replay dry run to %s", LOCAL_ARTIFACT_DIR)
        return

    with executor_context():
        steps = build_steps(
            spec=spec,
            analysis_output_path=args.analysis_output_path,
            replay_code_commit=replay_code_commit,
            audit=audit,
        )
    executor_main(
        ExecutorMainConfig(max_concurrent=1),
        steps=steps,
        description=(
            "Recover the missing phase-0 checkpoint of the observed Delphi 3e18 Uncheatable frontier "
            "for state-conditioned phase-1 optimization"
        ),
    )


if __name__ == "__main__":
    main()
