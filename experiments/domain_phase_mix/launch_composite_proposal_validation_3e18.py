# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the 6-row 3e18 validation panel for the composite surrogate's own top proposal.

The composite surrogate ranked fixed-aggregate two-phase policies at the 3e18 Uncheatable
frontier aggregate, and its top pick was the ``technical_specialization`` direction in the
``plus`` orientation at phase total variation 0.24, predicted to beat the tied control by
0.0078 BPB. The same model made the same kind of call at the 60M frontier and was right,
beating that tied control by 0.00548, which is 7.2 sigma of the anchor's control standard
deviation. This panel is the 3e18 half of that test, and the prediction was written into the
source manifest before any run existed.

Six rows in two seed blocks: the proposal, its antithetic partner at the same phase total
variation, and a tied control. The antithetic partner is what separates "the model picked the
right orientation" from "any asymmetry along this direction helps", and both orientations are
needed for the odd/even decomposition of the response. The aggregate is identical across all
six rows to machine precision, so nothing varies except phase ordering and seed.

Every training-graph detail -- model shape, token budget, phase schedule, TPU type, validation
sets -- is delegated to ``launch_delphi_augmented_swarm_3e18``, so the run configuration is
identical to the panels this validation will be compared against. Placement is pinned to
us-east5 and the launcher refuses any other region, zone, or bucket.
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
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import numpy as np
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_aggressive_phase_asymmetry_panel_3e18 as aggressive
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.domain_phase_mix.materialize_composite_proposal_validation_3e18 import ALPHA_0, PROPOSAL
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT_DIR = (
    SCRIPT_DIR
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_3e18_composite_proposal_validation_20260726"
)
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_composite_proposal_validation_20260726"
PANEL_ID = "delphi_3e18_composite_proposal_validation_20260726"

DEFAULT_SOURCE_PANEL = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_3e18_composite_proposal_validation_20260726/"
    "source/validation_panel-342a448c42787394.csv"
)
SOURCE_PANEL_SHA256 = "342a448c4278739432fa73e3bb37e7a4864ad398f14d3c8b7a2748909e3cf66d"

EXPECTED_RUNS = 6
EXPECTED_SIGN_COUNTS = {"center": 2, "minus": 2, "plus": 2}
EXPECTED_ANCHOR_ID = "uncheatable_frontier"
EXPECTED_DISTINCT_POLICIES = 3
DEFAULT_MAX_CONCURRENT = 6
AGGREGATE_TOLERANCE = 2e-12
ALPHA_TOLERANCE = 1e-12
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_BUCKET_PREFIX = "gs://marin-us-east5/"


@dataclass(frozen=True)
class ProposalMetadata:
    """Per-row provenance, including the prediction recorded before the run existed."""

    candidate_id: str
    anchor_id: str
    direction_id: str
    contrast_family: str
    sign: str
    phase_tv: float
    seed_block: int
    replicate_index: int
    predicted_gain_vs_tied_bpb: float | None


@dataclass(frozen=True)
class SavePanelManifestConfig:
    output_path: str
    source_panel: str
    source_panel_sha256: str
    analysis_output_path: str
    run_specs_json: str
    metadata_json: str


def save_panel_manifest(config: SavePanelManifestConfig) -> None:
    """Persist source provenance, resolved training configs, and the pre-registered prediction."""
    run_specs = [augmented.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    metadata = [ProposalMetadata(**item) for item in json.loads(config.metadata_json)]
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
        "panel_id": PANEL_ID,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "n_runs": len(run_specs),
        "sign_counts": dict(Counter(item.sign for item in metadata)),
        "phase_tv_counts": dict(Counter(item.phase_tv for item in metadata)),
        "target_flops": augmented.TARGET_FLOPS,
        "native_table9_scheduled": True,
        "prediction": PROPOSAL,
        "realized_phase_fractions": {
            "phase_0": run_specs[0].phase_0_fraction,
            "phase_1": run_specs[0].phase_1_fraction,
        },
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _run_only_patterns(value: str) -> list[str]:
    try:
        patterns = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("--run-only must be a JSON list of regex strings") from error
    if not isinstance(patterns, list) or not patterns or not all(isinstance(pattern, str) for pattern in patterns):
        raise argparse.ArgumentTypeError("--run-only must be a non-empty JSON list of regex strings")
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as error:
            raise argparse.ArgumentTypeError(f"invalid --run-only regex {pattern!r}: {error}") from error
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--source-panel-sha256", default=SOURCE_PANEL_SHA256)
    parser.add_argument("--analysis-output-path", default=augmented.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=augmented.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=augmented.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument(
        "--run-only",
        "--run_only",
        dest="run_only",
        type=_run_only_patterns,
        help="JSON list of step-name regexes to run with their dependencies.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_source_panel(source_panel: str, source_panel_sha256: str) -> list[dict[str, str]]:
    """Read the digest-pinned manifest and check its composition is the panel this launcher trains."""
    with fsspec.open(source_panel, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source_panel_sha256:
        raise ValueError(f"Source panel digest {digest} does not match expected {source_panel_sha256}")
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} rows, found {len(rows)}")
    sign_counts = dict(Counter(row["sign"] for row in rows))
    if sign_counts != EXPECTED_SIGN_COUNTS:
        raise ValueError(f"Sign composition changed: {sign_counts}")
    anchors = {row["anchor_id"] for row in rows}
    if anchors != {EXPECTED_ANCHOR_ID}:
        raise ValueError(f"Anchor set changed: {sorted(anchors)}")
    policies = {row["policy_sha256"] for row in rows}
    if len(policies) != EXPECTED_DISTINCT_POLICIES:
        raise ValueError(f"Expected {EXPECTED_DISTINCT_POLICIES} distinct policies, found {len(policies)}")
    treated = {float(row["phase_tv"]) for row in rows if row["sign"] != "center"}
    if treated != {PROPOSAL["phase_tv"]}:
        raise ValueError(f"Treated rows are not all at the proposed phase TV: {sorted(treated)}")
    predicted = {row["candidate_id"] for row in rows if row["model_predicted_gain_vs_tied_bpb"]}
    if len(predicted) != EXPECTED_SIGN_COUNTS[PROPOSAL["winning_sign"]]:
        raise ValueError(f"Prediction is not recorded on exactly the proposal rows: {sorted(predicted)}")
    return rows


def validate_fixed_aggregate(
    rows: list[dict[str, str]],
    phase_weights: list[dict[str, dict[str, float]]],
    alpha0: float,
    alpha1: float,
) -> None:
    """Confirm one shared aggregate under the realized phase split, not the materializer's assumed one."""
    aggregates = []
    for row, weights in zip(rows, phase_weights, strict=True):
        array = aggressive._weights_array(weights)
        aggregates.append(alpha0 * array[0] + alpha1 * array[1])
        recorded = float(row["aggregate_max_abs_error"])
        if recorded > AGGREGATE_TOLERANCE:
            raise ValueError(f"{row['candidate_id']} records aggregate error {recorded:.3e}")
    reference = aggregates[0]
    for row, aggregate in zip(rows, aggregates, strict=True):
        drift = float(np.max(np.abs(aggregate - reference)))
        if drift > AGGREGATE_TOLERANCE:
            raise ValueError(f"{row['candidate_id']} aggregate drifts from the panel anchor by {drift:.3e}")
    centers = [
        aggressive._weights_array(weights)
        for row, weights in zip(rows, phase_weights, strict=True)
        if row["sign"] == "center"
    ]
    anchor = centers[0][0]
    for center in centers:
        if float(np.max(np.abs(center - np.stack([anchor, anchor])))) > AGGREGATE_TOLERANCE:
            raise ValueError("Tied controls are not one shared tied coordinate")
    if float(np.max(np.abs(reference - anchor))) > AGGREGATE_TOLERANCE:
        raise ValueError("Panel aggregate does not equal the tied control coordinate")
    logger.info("Confirmed one shared aggregate across %d rows at alpha0=%.12f", len(rows), alpha0)


def build_run_specs(
    rows: list[dict[str, str]],
    phase_weights: list[dict[str, dict[str, float]]],
    *,
    candidate,
    alpha0: float,
    alpha1: float,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[augmented.DelphiSwarmRunSpec], list[ProposalMetadata]]:
    realized_train_tokens = candidate.train_steps * augmented.TARGET_BATCH_SIZE * augmented.SEQ_LEN_DELPHI
    if realized_train_tokens > augmented.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError("Resolved Delphi token budget exceeds the fixed simulated-epoch budget")
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(augmented.completed_adamh_heuristic.vocab_size))
    tensor_parallel_size = augmented._tensor_parallel_size(candidate.model_config.hidden_dim, augmented.TARGET_TPU_TYPE)

    run_specs: list[augmented.DelphiSwarmRunSpec] = []
    metadata: list[ProposalMetadata] = []
    for run_order, (row, weights) in enumerate(zip(rows, phase_weights, strict=True)):
        max_epoch, q95_epoch, mean_phase_tv = augmented._weight_diagnostics(weights)
        run_specs.append(
            augmented.DelphiSwarmRunSpec(
                run_order=run_order,
                run_id=int(row["run_id"]),
                run_name=f"cmpval_{run_order:02d}_{row['sign']}_s{row['seed_block']}",
                source_run_name=row["candidate_id"],
                source_experiment=EXPERIMENT_NAME,
                panel_source=PANEL_ID,
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
                phase_0_fraction=alpha0,
                phase_1_fraction=alpha1,
                simulated_epoch_target_budget=augmented.SIMULATED_EPOCH_TARGET_BUDGET,
                available_top_level_tokens=augmented.TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=mean_phase_tv,
                phase_weights=weights,
            )
        )
        recorded_prediction = row["model_predicted_gain_vs_tied_bpb"]
        metadata.append(
            ProposalMetadata(
                candidate_id=row["candidate_id"],
                anchor_id=row["anchor_id"],
                direction_id=row["direction_id"],
                contrast_family=row["contrast_family"],
                sign=row["sign"],
                phase_tv=float(row["phase_tv"]),
                seed_block=int(row["seed_block"]),
                replicate_index=int(row["replicate_index"]),
                predicted_gain_vs_tied_bpb=float(recorded_prediction) if recorded_prediction else None,
            )
        )
    return run_specs, metadata


def build_launch_artifacts(
    *,
    run_specs: list[augmented.DelphiSwarmRunSpec],
    metadata: list[ProposalMetadata],
    analysis_output_path: str,
    source_panel: str,
    source_panel_sha256: str,
    validation_configs,
) -> aggressive.LaunchArtifacts:
    """Build the 6-train plus 6-native-Table-9 graph."""
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
                    "delphi-3e18-composite-proposal-validation",
                    "boundary-aligned-wsd-anneal-content",
                    "token-aggregate-matched-phase-contrast",
                    f"anchor={row_metadata.anchor_id}",
                    f"contrast_family={row_metadata.contrast_family}",
                    f"direction={row_metadata.direction_id}",
                    f"sign={row_metadata.sign}",
                    f"phase_tv={row_metadata.phase_tv:g}",
                    f"seed_block={row_metadata.seed_block}",
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
                wandb_group=f"olmo_base_eval_table9_{PANEL_ID}",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": PANEL_ID,
                    "scale": "3e18",
                    **asdict(row_metadata),
                },
            )
        )
    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_panel_manifest,
        config=SavePanelManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            source_panel_sha256=source_panel_sha256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
        ),
    )
    return aggressive.LaunchArtifacts(manifest_step, training_steps, eval_steps)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.tpu_region != augmented.DEFAULT_TPU_REGION or args.tpu_zone != augmented.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {augmented.DEFAULT_TPU_REGION}/{augmented.DEFAULT_TPU_ZONE}")
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    if args.dry_run and args.run_only is not None:
        raise ValueError("--dry-run validates the full panel and cannot be combined with --run-only")
    if SHA256_PATTERN.fullmatch(args.source_panel_sha256) is None:
        raise ValueError("--source-panel-sha256 must be a lowercase SHA-256 digest")
    if not args.dry_run and not args.source_panel.startswith(REQUIRED_BUCKET_PREFIX):
        raise ValueError(f"Production source panel must live under {REQUIRED_BUCKET_PREFIX}")

    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    rows = load_source_panel(args.source_panel, args.source_panel_sha256)
    scaling_fits = augmented._read_scaling_fits(args.analysis_output_path)
    candidate = augmented._candidate_for_budget(scaling_fits=scaling_fits)
    alpha0, alpha1 = aggressive._realized_phase_fractions(candidate)
    if abs(alpha0 - ALPHA_0) > ALPHA_TOLERANCE:
        raise ValueError(
            f"Realized phase-0 share {alpha0!r} differs from the {ALPHA_0!r} the panel geometry was built for; "
            "the aggregate would not be held fixed at train time"
        )

    phase_weights = [augmented._phase_weights_from_row(row, source_run_name=row["candidate_id"]) for row in rows]
    validate_fixed_aggregate(rows, phase_weights, alpha0, alpha1)
    run_specs, metadata = build_run_specs(
        rows,
        phase_weights,
        candidate=candidate,
        alpha0=alpha0,
        alpha1=alpha1,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )

    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        save_panel_manifest(
            SavePanelManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR / "launch_dry_run"),
                source_panel=args.source_panel,
                source_panel_sha256=args.source_panel_sha256,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
            )
        )
        logger.info(
            "Dry run resolved %d runs at %s/%s: train_steps=%d, tokens=%.4g, alpha0=%.12f, params=%d",
            len(run_specs),
            args.tpu_region,
            args.tpu_zone,
            candidate.train_steps,
            run_specs[0].realized_train_tokens,
            alpha0,
            run_specs[0].total_trainable_params,
        )
        for run_spec, item in zip(run_specs, metadata, strict=True):
            logger.info(
                "  %-24s sign=%-6s tv=%.2f seed=%d max_epoch=%.2f predicted_gain=%s",
                run_spec.run_name,
                item.sign,
                item.phase_tv,
                run_spec.data_seed,
                run_spec.max_simulated_epoch,
                "none" if item.predicted_gain_vs_tied_bpb is None else f"{item.predicted_gain_vs_tied_bpb:+.5f}",
            )
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
            source_panel_sha256=args.source_panel_sha256,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info("Built graph with %d training steps; skipping launch under CI", len(artifacts.training_steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent, run_only=args.run_only),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: 6-row fixed-aggregate validation of the composite surrogate's top 3e18 "
            f"proposal ({PROPOSAL['direction_id']} {PROPOSAL['winning_sign']} at TV {PROPOSAL['phase_tv']:g}) "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
