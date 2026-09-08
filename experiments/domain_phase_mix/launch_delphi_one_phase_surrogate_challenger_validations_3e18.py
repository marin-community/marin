# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate eight aggregate-linear-V one-phase mixtures at Delphi 3e18.

The frozen candidate panel contains both aggregate-linear-V and flexible
separate-head optima. This launcher selects only the aggregate candidates for
Uncheatable and Table-9 at whole-run epoch caps 4, 6, 8, and 10. Every mixture
uses the same trainer and data seeds as the completed one-phase DSP cap sweep,
is tied across the nominal 80/20 schedule phases, and receives inline
Uncheatable plus native Table-9 evaluation.
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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import fsspec
from levanter.optim.adamh import AdamHConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_one_phase_surrogate_challenger_validations_3e18_20260831"
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_one_phase_surrogate_challenger_validations_20260831"
)
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_CANDIDATE_DIR / "manifest.json"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "0e98d5d98354308516050dec9bc09766df06f42367fb5014ed31541739311546"
EXPECTED_CANDIDATE_MANIFEST_SHA256 = "a309c509fe3085e14387816b255d10bd8c0772e835badc28496c3ed7dc1e8fed"
MIXTURE_BLOCK_SIZE = 2048
TPU_TYPE = "v6e-8"
TPU_REGION = "us-east5"
TPU_ZONE = "us-east5-b"
RUN_ID_BASE = 7_310_000
COMMON_DATA_SEED = 7_280_000
TRAINER_SEED = 0
EXPECTED_RUN_COUNT = 8
MAX_CONCURRENT = EXPECTED_RUN_COUNT
CAP_TOLERANCE = 1e-10
MATERIALIZATION_ACCOUNTING_TOLERANCE = 1e-5
MODEL_KEY = "aggregate_linear_v"
CAPS = (4, 6, 8, 10)
TARGETS = ("uncheatable_bpb", "table9_macro_bpb")
CANDIDATE_IDS = tuple(f"{MODEL_KEY}_{target.removesuffix('_bpb')}_cap{cap:02d}" for target in TARGETS for cap in CAPS)


@dataclass(frozen=True)
class CandidateMixture:
    """One exact aggregate-linear-V mixture selected for validation."""

    candidate_id: str
    target: str
    target_label: str
    epoch_cap: int
    runtime_counts: dict[str, int]
    weights: dict[str, float]
    max_materialized_epoch: float
    q95_materialized_epoch: float


@dataclass(frozen=True)
class SaveManifestConfig:
    """Frozen provenance for the eight-row validation graph."""

    output_path: str
    candidate_weights_path: str
    candidate_weights_sha256: str
    candidate_manifest_path: str
    candidate_manifest_sha256: str
    source_panel: str
    source_panel_sha256: str
    analysis_output_path: str
    candidates_json: str
    run_specs_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved manifest, training, and native Table-9 graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


class CurrentCompletedAdamHHeuristic(CompletedAdamHHeuristic):
    """Adapt the frozen heuristic to the current non-Nesterov AdamH API."""

    def build_optimizer_config(self, batch_size: int, tokens: float) -> AdamHConfig:
        return AdamHConfig(
            learning_rate=self._compute_learning_rate(batch_size, tokens),
            adam_lr=self._compute_adam_lr(batch_size, tokens),
            min_lr_ratio=self.min_lr_ratio,
            warmup=self.warmup,
            beta1=self.beta1,
            beta2=self._compute_beta2(batch_size),
            epsilon=self._compute_epsilon(batch_size, tokens),
            max_grad_norm=self.max_grad_norm,
            lr_schedule=self.lr_schedule,
            decay=self.decay,
        )


current_completed_adamh_heuristic = CurrentCompletedAdamHHeuristic()


def run_one_phase_training(config: base.DelphiSwarmTrainingConfig) -> None:
    """Install the current AdamH adapter in the child before training."""
    base.completed_adamh_heuristic = current_completed_adamh_heuristic
    base.run_delphi_swarm_training(config)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _q95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[round(0.95 * (len(ordered) - 1))]


def validate_candidate_manifest(path: Path, expected_sha256: str) -> None:
    """Bind the launch to the exact materializer manifest and output hash."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Candidate manifest changed: {actual_sha256} != {expected_sha256}")
    manifest = json.loads(path.read_text())
    expected = {
        "caps": list(CAPS),
        "models": ["aggregate_linear_v", "corrected_separate_heads"],
        "targets": list(TARGETS),
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "candidate_count": 16,
    }
    observed = {key: manifest.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Candidate manifest contract changed: {observed} != {expected}")
    candidate_hash = manifest.get("outputs", {}).get("candidate_weights.csv")
    if candidate_hash != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError(
            "Candidate manifest does not bind the expected weight table: "
            f"{candidate_hash} != {EXPECTED_CANDIDATE_WEIGHTS_SHA256}"
        )


def load_candidate_mixtures(path: Path, expected_sha256: str) -> list[CandidateMixture]:
    """Load and strictly validate the eight aggregate-linear-V candidates."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Candidate weights changed: {actual_sha256} != {expected_sha256}")

    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    required = {
        "candidate_id",
        "model",
        "target",
        "target_label",
        "epoch_cap",
        "domain",
        "runtime_count",
        "weight",
        "materialized_epochs",
    }
    if not rows:
        raise ValueError("Candidate weights table is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Candidate weights are missing columns: {sorted(missing)}")

    selected_rows = [row for row in rows if row["model"] == MODEL_KEY and int(row["epoch_cap"]) in CAPS]
    ordered_ids = tuple(dict.fromkeys(row["candidate_id"] for row in selected_rows))
    if ordered_ids != CANDIDATE_IDS:
        raise ValueError(f"Candidate order or identities changed: {ordered_ids}")

    candidates = []
    coordinates: set[tuple[int, ...]] = set()
    for candidate_id in CANDIDATE_IDS:
        candidate_rows = [row for row in selected_rows if row["candidate_id"] == candidate_id]
        domains = tuple(row["domain"] for row in candidate_rows)
        if len(candidate_rows) != len(DOMAIN_NAMES) or set(domains) != set(DOMAIN_NAMES):
            raise ValueError(f"{candidate_id} does not contain exactly the Delphi runtime buckets")
        if len(set(domains)) != len(domains):
            raise ValueError(f"{candidate_id} contains duplicate buckets")

        targets = {row["target"] for row in candidate_rows}
        target_labels = {row["target_label"] for row in candidate_rows}
        caps = {int(row["epoch_cap"]) for row in candidate_rows}
        if len(targets) != 1 or len(target_labels) != 1 or len(caps) != 1:
            raise ValueError(f"{candidate_id} has inconsistent target or cap metadata")
        target = next(iter(targets))
        cap = next(iter(caps))
        if target not in TARGETS or cap not in CAPS or not candidate_id.endswith(f"cap{cap:02d}"):
            raise ValueError(f"{candidate_id} has unexpected target={target!r} or cap={cap}")

        counts = {row["domain"]: int(row["runtime_count"]) for row in candidate_rows}
        weights = {row["domain"]: float(row["weight"]) for row in candidate_rows}
        if sum(counts.values()) != MIXTURE_BLOCK_SIZE or min(counts.values()) < 0:
            raise ValueError(f"{candidate_id} has invalid runtime counts")
        for domain in DOMAIN_NAMES:
            expected_weight = counts[domain] / MIXTURE_BLOCK_SIZE
            if weights[domain] != expected_weight:
                raise ValueError(f"{candidate_id}/{domain} is not on the exact runtime grid")

        epochs = {row["domain"]: float(row["materialized_epochs"]) for row in candidate_rows}
        max_epoch = max(epochs.values())
        if max_epoch > cap + CAP_TOLERANCE:
            raise ValueError(f"{candidate_id} violates its whole-run epoch cap: {max_epoch} > {cap}")
        coordinate = tuple(counts[domain] for domain in DOMAIN_NAMES)
        if coordinate in coordinates:
            raise ValueError(f"{candidate_id} duplicates an earlier runtime mixture")
        coordinates.add(coordinate)
        candidates.append(
            CandidateMixture(
                candidate_id=candidate_id,
                target=target,
                target_label=next(iter(target_labels)),
                epoch_cap=cap,
                runtime_counts=counts,
                weights=weights,
                max_materialized_epoch=max_epoch,
                q95_materialized_epoch=_q95(list(epochs.values())),
            )
        )
    if len(candidates) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} mixtures, found {len(candidates)}")
    return candidates


def build_run_specs(
    *,
    template: base.DelphiSwarmRunSpec,
    candidates: list[CandidateMixture],
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[base.DelphiSwarmRunSpec]:
    """Bind exact tied mixtures to the canonical full-horizon setup."""
    tensor_parallel_size = base._tensor_parallel_size(template.model_hidden_dim, tpu_type)
    run_specs = []
    for run_order, candidate in enumerate(candidates):
        weights = {domain: candidate.weights[domain] for domain in template.phase_weights["phase_0"]}
        phase_weights = {"phase_0": dict(weights), "phase_1": dict(weights)}
        max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(phase_weights)
        if abs(max_epoch - candidate.max_materialized_epoch) > MATERIALIZATION_ACCOUNTING_TOLERANCE:
            raise ValueError(
                f"{candidate.candidate_id} materialized-epoch accounting changed: "
                f"{max_epoch} != {candidate.max_materialized_epoch}"
            )
        if max_epoch > candidate.epoch_cap + CAP_TOLERANCE:
            raise ValueError(
                f"{candidate.candidate_id} violates its runtime epoch cap: {max_epoch} > {candidate.epoch_cap}"
            )
        run_specs.append(
            replace(
                template,
                run_order=run_order,
                run_id=RUN_ID_BASE + run_order,
                run_name=f"onephase_av_{candidate.target.removesuffix('_bpb')}_cap{candidate.epoch_cap:02d}",
                source_run_name=candidate.candidate_id,
                source_experiment=EXPERIMENT_NAME,
                panel_source="aggregate_linear_v_epoch_cap_optimum",
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                tensor_parallel_size=tensor_parallel_size,
                data_seed=COMMON_DATA_SEED,
                trainer_seed=TRAINER_SEED,
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv,
                phase_weights=phase_weights,
            )
        )
    if len(run_specs) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} run specs, found {len(run_specs)}")
    return run_specs


def save_manifest(config: SaveManifestConfig) -> None:
    """Persist exact launch rows, candidate provenance, and phase weights."""
    candidates = [CandidateMixture(**item) for item in json.loads(config.candidates_json)]
    run_specs = [base.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    if len(candidates) != EXPECTED_RUN_COUNT or len(run_specs) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Candidate/run-spec mismatch: {len(candidates)} != {len(run_specs)}")

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(spec) for spec in run_specs], handle, indent=2, sort_keys=True)

    weights_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        weights_buffer,
        fieldnames=["candidate_id", "phase", "domain", "runtime_count", "weight"],
    )
    writer.writeheader()
    for candidate in candidates:
        for phase in base.PHASE_NAMES:
            for domain in DOMAIN_NAMES:
                writer.writerow(
                    {
                        "candidate_id": candidate.candidate_id,
                        "phase": phase,
                        "domain": domain,
                        "runtime_count": candidate.runtime_counts[domain],
                        "weight": candidate.weights[domain],
                    }
                )
    with fs.open(os.path.join(config.output_path, "phase_weights.csv"), "w") as handle:
        handle.write(weights_buffer.getvalue())

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "candidate_weights_path": config.candidate_weights_path,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "candidate_manifest_path": config.candidate_manifest_path,
        "candidate_manifest_sha256": config.candidate_manifest_sha256,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "runtime_distinct_trainings": EXPECTED_RUN_COUNT,
        "candidate_ids": list(CANDIDATE_IDS),
        "model": MODEL_KEY,
        "targets": list(TARGETS),
        "whole_run_epoch_caps": list(CAPS),
        "policy_class": "single_phase_tied",
        "common_random_numbers": {"data_seed": COMMON_DATA_SEED, "trainer_seed": TRAINER_SEED},
        "inline_uncheatable_scheduled": True,
        "native_table9_scheduled": True,
        "hardware": {"tpu_type": TPU_TYPE, "region": TPU_REGION, "zone": TPU_ZONE},
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[base.DelphiSwarmRunSpec],
    candidates: list[CandidateMixture],
    candidate_weights_path: Path,
    candidate_manifest_path: Path,
    analysis_output_path: str,
    validation_configs,
) -> LaunchArtifacts:
    """Build eight full trainings, eight native Table-9 evals, and a manifest."""
    table9_wandb_group = "olmo_base_eval_table9_delphi_3e18_one_phase_aggregate_v_validation"
    training_wandb_group = "delphi_3e18_one_phase_aggregate_v_validation"
    provenance_panel = "delphi_3e18_one_phase_aggregate_v_validation"
    base_artifacts = base.build_launch_artifacts(
        run_specs=run_specs,
        analysis_output_path=analysis_output_path,
        source_panel=str(candidate_weights_path),
        validation_configs=validation_configs,
        experiment_name=EXPERIMENT_NAME,
        wandb_tags=("delphi-3e18", "one-phase", "aggregate-linear-v", "surrogate-validation"),
        training_wandb_group=training_wandb_group,
        table9_wandb_group=table9_wandb_group,
        provenance_panel=provenance_panel,
        provenance_scale="3e18",
        steps_per_eval=1000,
        permanent_checkpoint_interval=5000,
    )

    training_steps = []
    eval_steps = []
    for original_step, run_spec in zip(base_artifacts.training_steps, run_specs, strict=True):
        if original_step.resources is None:
            raise ValueError(f"Training step {original_step.name} has no resource configuration")
        training_step = replace(
            original_step,
            fn=remote(
                run_one_phase_training,
                resources=original_step.resources,
                env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=base.TABLE9_REQUEST_SET_DIR,
                resource_config=base.TABLE9_EVAL_RESOURCES,
                wandb_group=table9_wandb_group,
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": provenance_panel,
                    "scale": "3e18",
                    "source_run_name": run_spec.source_run_name,
                    "swarm_run_name": run_spec.run_name,
                    "panel_source": run_spec.panel_source,
                },
            )
        )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_manifest,
        config=SaveManifestConfig(
            output_path=this_output_path(),
            candidate_weights_path=str(candidate_weights_path),
            candidate_weights_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
            candidate_manifest_path=str(candidate_manifest_path),
            candidate_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
            source_panel=base.DEFAULT_SOURCE_PANEL,
            source_panel_sha256=base.SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            candidates_json=json.dumps([asdict(candidate) for candidate in candidates], sort_keys=True),
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )
    if len(training_steps) != EXPECTED_RUN_COUNT or len(eval_steps) != EXPECTED_RUN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RUN_COUNT} train/eval pairs, found {len(training_steps)}/{len(eval_steps)}"
        )
    return LaunchArtifacts(manifest_step=manifest_step, training_steps=training_steps, eval_steps=eval_steps)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-type", default=TPU_TYPE)
    parser.add_argument("--tpu-region", default=TPU_REGION)
    parser.add_argument("--tpu-zone", default=TPU_ZONE)
    parser.add_argument("--table9-tpu-zone", default=TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    hardware = (args.tpu_type, args.tpu_region, args.tpu_zone)
    if hardware != (TPU_TYPE, TPU_REGION, TPU_ZONE):
        raise ValueError(f"This launcher is pinned to {(TPU_TYPE, TPU_REGION, TPU_ZONE)}, got {hardware}")
    if args.table9_tpu_zone != TPU_ZONE:
        raise ValueError(f"Table-9 evaluation is pinned to {TPU_ZONE}, got {args.table9_tpu_zone}")
    if not 1 <= args.max_concurrent <= MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    validate_candidate_manifest(args.candidate_manifest, EXPECTED_CANDIDATE_MANIFEST_SHA256)
    candidates = load_candidate_mixtures(args.candidate_weights, EXPECTED_CANDIDATE_WEIGHTS_SHA256)
    base.completed_adamh_heuristic = current_completed_adamh_heuristic
    source_specs = base.load_source_panel(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        tpu_type=args.tpu_type,
    )
    run_specs = build_run_specs(
        template=source_specs[0],
        candidates=candidates,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        save_manifest(
            SaveManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR),
                candidate_weights_path=str(args.candidate_weights),
                candidate_weights_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
                candidate_manifest_path=str(args.candidate_manifest),
                candidate_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
                source_panel=base.DEFAULT_SOURCE_PANEL,
                source_panel_sha256=base.SOURCE_PANEL_SHA256,
                analysis_output_path=args.analysis_output_path,
                candidates_json=json.dumps([asdict(candidate) for candidate in candidates], sort_keys=True),
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            )
        )
        logger.info("Wrote %d validation rows under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            candidates=candidates,
            candidate_weights_path=args.candidate_weights,
            candidate_manifest_path=args.candidate_manifest,
            analysis_output_path=args.analysis_output_path,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built %d full trainings and %d native Table-9 evals; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: eight aggregate-linear-V one-phase optima at whole-run epoch caps 4-10, "
            "with inline Uncheatable and native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
