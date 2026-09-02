# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the runtime-distinct Delphi 3e18 one-phase DSP epoch-cap optima.

The frozen materialization contains 20 nominal target/cap cells but only 11
runtime-distinct 1/2048 mixtures. This launcher trains each distinct mixture
once, ties its weights across both schedule phases, and evaluates every final
checkpoint on Uncheatable and native Table-9.
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

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_one_phase_dsp_epoch_cap_sweep_3e18_20260828"
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_one_phase_dsp_epoch_cap_sweep_20260828"
)
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "63d0b92be2f45f2444d49d95b95a2f98cecb3308cfb40cf8ed5f439cabbef07d"
MIXTURE_BLOCK_SIZE = 2048
TPU_TYPE = "v6e-8"
TPU_REGION = "us-east5"
TPU_ZONE = "us-east5-b"
RUN_ID_BASE = 7_280_000
COMMON_DATA_SEED = 7_280_000
TRAINER_SEED = 0
EXPECTED_RUN_COUNT = 11
MAX_CONCURRENT = EXPECTED_RUN_COUNT
CAP_TOLERANCE = 1e-10
MATERIALIZATION_ACCOUNTING_TOLERANCE = 1e-5

NOMINAL_CANDIDATE_IDS = (
    "uncheatable_cap02",
    "uncheatable_cap04",
    "uncheatable_cap06",
    "uncheatable_cap08",
    "uncheatable_cap10",
    "uncheatable_cap12",
    "uncheatable_cap14",
    "uncheatable_cap16",
    "uncheatable_cap18",
    "uncheatable_cap20",
    "table9_macro_cap02",
    "table9_macro_cap04",
    "table9_macro_cap06",
    "table9_macro_cap08",
    "table9_macro_cap10",
    "table9_macro_cap12",
    "table9_macro_cap14",
    "table9_macro_cap16",
    "table9_macro_cap18",
    "table9_macro_cap20",
)
EXPECTED_ALIAS_MAP = {
    "uncheatable_cap02": "uncheatable_cap02",
    "uncheatable_cap04": "uncheatable_cap04",
    "uncheatable_cap06": "uncheatable_cap06",
    "uncheatable_cap08": "uncheatable_cap08",
    "uncheatable_cap10": "uncheatable_cap10",
    "uncheatable_cap12": "uncheatable_cap10",
    "uncheatable_cap14": "uncheatable_cap10",
    "uncheatable_cap16": "uncheatable_cap10",
    "uncheatable_cap18": "uncheatable_cap10",
    "uncheatable_cap20": "uncheatable_cap10",
    "table9_macro_cap02": "table9_macro_cap02",
    "table9_macro_cap04": "table9_macro_cap04",
    "table9_macro_cap06": "table9_macro_cap06",
    "table9_macro_cap08": "table9_macro_cap08",
    "table9_macro_cap10": "table9_macro_cap10",
    "table9_macro_cap12": "table9_macro_cap12",
    "table9_macro_cap14": "table9_macro_cap12",
    "table9_macro_cap16": "table9_macro_cap12",
    "table9_macro_cap18": "table9_macro_cap12",
    "table9_macro_cap20": "table9_macro_cap12",
}


@dataclass(frozen=True)
class CandidateMixture:
    """One runtime-distinct mixture and the nominal cells that alias it."""

    candidate_id: str
    target: str
    target_label: str
    epoch_cap: int
    aliases: tuple[str, ...]
    runtime_counts: dict[str, int]
    weights: dict[str, float]
    max_materialized_epoch: float
    q95_materialized_epoch: float


@dataclass(frozen=True)
class SweepDefinition:
    """Immutable identities and provenance for one frozen candidate sweep."""

    experiment_name: str
    nominal_candidate_ids: tuple[str, ...]
    expected_alias_map: dict[str, str]
    expected_run_count: int
    run_id_base: int
    common_data_seed: int
    trainer_seed: int
    run_name_prefix: str
    panel_source: str
    table9_wandb_group: str
    provenance_panel: str
    wandb_tags: tuple[str, ...]


DEFAULT_SWEEP_DEFINITION = SweepDefinition(
    experiment_name=EXPERIMENT_NAME,
    nominal_candidate_ids=NOMINAL_CANDIDATE_IDS,
    expected_alias_map=EXPECTED_ALIAS_MAP,
    expected_run_count=EXPECTED_RUN_COUNT,
    run_id_base=RUN_ID_BASE,
    common_data_seed=COMMON_DATA_SEED,
    trainer_seed=TRAINER_SEED,
    run_name_prefix="onephase_dsp",
    panel_source="dsp_epoch_cap_optimum",
    table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_dsp_epoch_cap_sweep",
    provenance_panel="delphi_3e18_one_phase_dsp_epoch_cap_sweep",
    wandb_tags=("delphi-3e18", "one-phase", "dsp", "whole-run-epoch-cap-sweep"),
)


@dataclass(frozen=True)
class SaveSweepManifestConfig:
    """Inputs needed to materialize immutable launch provenance."""

    output_path: str
    candidate_weights_path: str
    candidate_weights_sha256: str
    source_panel: str
    source_panel_sha256: str
    analysis_output_path: str
    candidates_json: str
    run_specs_json: str
    sweep_definition_json: str


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


def load_candidate_mixtures(
    path: Path,
    expected_sha256: str,
    *,
    definition: SweepDefinition = DEFAULT_SWEEP_DEFINITION,
) -> tuple[list[CandidateMixture], dict[str, str]]:
    """Load, validate, and exactly deduplicate the frozen 20-cell table."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Candidate weights changed: {actual_sha256} != {expected_sha256}")

    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    required = {
        "candidate_id",
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

    ordered_ids = tuple(dict.fromkeys(row["candidate_id"] for row in rows))
    if ordered_ids != definition.nominal_candidate_ids:
        raise ValueError(f"Candidate order or identities changed: {ordered_ids}")

    nominal: dict[str, CandidateMixture] = {}
    coordinates: dict[str, tuple[int, ...]] = {}
    for candidate_id in definition.nominal_candidate_ids:
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        domains = tuple(row["domain"] for row in candidate_rows)
        if len(candidate_rows) != len(DOMAIN_NAMES) or set(domains) != set(DOMAIN_NAMES):
            raise ValueError(f"{candidate_id} does not contain exactly the Delphi runtime buckets")
        if len(set(domains)) != len(domains):
            raise ValueError(f"{candidate_id} contains duplicate buckets")

        target_values = {row["target"] for row in candidate_rows}
        target_labels = {row["target_label"] for row in candidate_rows}
        cap_values = {int(row["epoch_cap"]) for row in candidate_rows}
        if len(target_values) != 1 or len(target_labels) != 1 or len(cap_values) != 1:
            raise ValueError(f"{candidate_id} has inconsistent target or cap metadata")
        cap = next(iter(cap_values))
        if not candidate_id.endswith(f"cap{cap:02d}"):
            raise ValueError(f"{candidate_id} does not agree with epoch_cap={cap}")

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
        coordinates[candidate_id] = coordinate
        nominal[candidate_id] = CandidateMixture(
            candidate_id=candidate_id,
            target=next(iter(target_values)),
            target_label=next(iter(target_labels)),
            epoch_cap=cap,
            aliases=(),
            runtime_counts=counts,
            weights=weights,
            max_materialized_epoch=max_epoch,
            q95_materialized_epoch=_q95(list(epochs.values())),
        )

    first_for_coordinate: dict[tuple[int, ...], str] = {}
    alias_map: dict[str, str] = {}
    for candidate_id in definition.nominal_candidate_ids:
        coordinate = coordinates[candidate_id]
        canonical = first_for_coordinate.setdefault(coordinate, candidate_id)
        alias_map[candidate_id] = canonical
    if alias_map != definition.expected_alias_map:
        raise ValueError(f"Runtime alias structure changed: {alias_map}")

    candidates = []
    for candidate_id in definition.nominal_candidate_ids:
        if alias_map[candidate_id] != candidate_id:
            continue
        aliases = tuple(alias for alias in definition.nominal_candidate_ids if alias_map[alias] == candidate_id)
        candidates.append(replace(nominal[candidate_id], aliases=aliases))
    if len(candidates) != definition.expected_run_count:
        raise ValueError(f"Expected {definition.expected_run_count} runtime-distinct mixtures, found {len(candidates)}")
    return candidates, alias_map


def build_run_specs(
    *,
    template: base.DelphiSwarmRunSpec,
    candidates: list[CandidateMixture],
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    definition: SweepDefinition = DEFAULT_SWEEP_DEFINITION,
) -> list[base.DelphiSwarmRunSpec]:
    """Bind the frozen mixtures to the canonical full Delphi training setup."""
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
                f"{candidate.candidate_id} violates its whole-run epoch cap under runtime accounting: "
                f"{max_epoch} > {candidate.epoch_cap}"
            )
        run_specs.append(
            replace(
                template,
                run_order=run_order,
                run_id=definition.run_id_base + run_order,
                run_name=f"{definition.run_name_prefix}_{candidate.candidate_id}",
                source_run_name=candidate.candidate_id,
                source_experiment=definition.experiment_name,
                panel_source=definition.panel_source,
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                tensor_parallel_size=tensor_parallel_size,
                data_seed=definition.common_data_seed,
                trainer_seed=definition.trainer_seed,
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv,
                phase_weights=phase_weights,
            )
        )
    if len(run_specs) != definition.expected_run_count:
        raise ValueError(f"Expected {definition.expected_run_count} run specs, found {len(run_specs)}")
    return run_specs


def save_sweep_manifest(config: SaveSweepManifestConfig) -> None:
    """Persist the exact launch rows, aliases, runtime counts, and weights."""
    candidates = [CandidateMixture(**item) for item in json.loads(config.candidates_json)]
    run_specs = [base.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    definition_data = json.loads(config.sweep_definition_json)
    definition = SweepDefinition(
        **{
            **definition_data,
            "nominal_candidate_ids": tuple(definition_data["nominal_candidate_ids"]),
            "wandb_tags": tuple(definition_data["wandb_tags"]),
        }
    )
    if len(candidates) != len(run_specs):
        raise ValueError(f"Candidate/run-spec mismatch: {len(candidates)} != {len(run_specs)}")

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(spec) for spec in run_specs], handle, indent=2, sort_keys=True)

    manifest_buffer = io.StringIO(newline="")
    manifest_writer = csv.DictWriter(
        manifest_buffer,
        fieldnames=[
            "run_order",
            "run_name",
            "candidate_id",
            "target",
            "epoch_cap",
            "aliases",
            "run_id",
            "data_seed",
            "trainer_seed",
            "train_steps",
            "expected_checkpoint_step",
            "tpu_type",
            "tpu_region",
            "tpu_zone",
            "fit_panel_max_materialized_epoch",
            "runtime_max_simulated_epoch",
        ],
    )
    manifest_writer.writeheader()
    for candidate, spec in zip(candidates, run_specs, strict=True):
        manifest_writer.writerow(
            {
                "run_order": spec.run_order,
                "run_name": spec.run_name,
                "candidate_id": candidate.candidate_id,
                "target": candidate.target,
                "epoch_cap": candidate.epoch_cap,
                "aliases": ";".join(candidate.aliases),
                "run_id": spec.run_id,
                "data_seed": spec.data_seed,
                "trainer_seed": spec.trainer_seed,
                "train_steps": spec.train_steps,
                "expected_checkpoint_step": spec.expected_checkpoint_step,
                "tpu_type": spec.tpu_type,
                "tpu_region": spec.tpu_region,
                "tpu_zone": spec.tpu_zone,
                "fit_panel_max_materialized_epoch": candidate.max_materialized_epoch,
                "runtime_max_simulated_epoch": spec.max_simulated_epoch,
            }
        )
    with fs.open(os.path.join(config.output_path, "training_manifest.csv"), "w") as handle:
        handle.write(manifest_buffer.getvalue())

    weights_buffer = io.StringIO(newline="")
    weights_writer = csv.DictWriter(
        weights_buffer,
        fieldnames=["candidate_id", "phase", "domain", "runtime_count", "weight"],
    )
    weights_writer.writeheader()
    for candidate in candidates:
        for phase in base.PHASE_NAMES:
            for domain in DOMAIN_NAMES:
                weights_writer.writerow(
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
        "experiment_name": definition.experiment_name,
        "candidate_weights_path": config.candidate_weights_path,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "nominal_cells": len(definition.nominal_candidate_ids),
        "runtime_distinct_trainings": len(run_specs),
        "policy_class": "single_phase_tied",
        "cap_scope": "whole training run",
        "common_random_numbers": {
            "data_seed": definition.common_data_seed,
            "trainer_seed": definition.trainer_seed,
        },
        "native_table9_scheduled": True,
        "inline_uncheatable_scheduled": True,
        "alias_map": definition.expected_alias_map,
        "hardware": {
            "tpu_type": run_specs[0].tpu_type,
            "region": run_specs[0].tpu_region,
            "zone": run_specs[0].tpu_zone,
        },
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[base.DelphiSwarmRunSpec],
    candidates: list[CandidateMixture],
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
    analysis_output_path: str,
    validation_configs,
    definition: SweepDefinition = DEFAULT_SWEEP_DEFINITION,
) -> LaunchArtifacts:
    """Build the frozen full trainings, native Table-9 evals, and manifest."""
    base_artifacts = base.build_launch_artifacts(
        run_specs=run_specs,
        analysis_output_path=analysis_output_path,
        source_panel=str(candidate_weights_path),
        validation_configs=validation_configs,
        experiment_name=definition.experiment_name,
        wandb_tags=definition.wandb_tags,
        table9_wandb_group=definition.table9_wandb_group,
        provenance_panel=definition.provenance_panel,
        provenance_scale="3e18",
        steps_per_eval=1000,
        permanent_checkpoint_interval=5000,
    )
    manifest_step = ExecutorStep(
        name=f"{definition.experiment_name}/manifest",
        fn=save_sweep_manifest,
        config=SaveSweepManifestConfig(
            output_path=this_output_path(),
            candidate_weights_path=str(candidate_weights_path),
            candidate_weights_sha256=candidate_weights_sha256,
            source_panel=base.DEFAULT_SOURCE_PANEL,
            source_panel_sha256=base.SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            candidates_json=json.dumps([asdict(candidate) for candidate in candidates], sort_keys=True),
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            sweep_definition_json=json.dumps(asdict(definition), sort_keys=True),
        ),
    )
    if (
        len(base_artifacts.training_steps) != definition.expected_run_count
        or len(base_artifacts.eval_steps) != definition.expected_run_count
    ):
        raise ValueError(
            f"Expected {definition.expected_run_count} train/eval pairs, found "
            f"{len(base_artifacts.training_steps)}/{len(base_artifacts.eval_steps)}"
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
                wandb_group=definition.table9_wandb_group,
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": definition.provenance_panel,
                    "scale": "3e18",
                    "source_run_name": run_spec.source_run_name,
                    "swarm_run_name": run_spec.run_name,
                    "panel_source": run_spec.panel_source,
                },
            )
        )
    return LaunchArtifacts(
        manifest_step=manifest_step,
        training_steps=training_steps,
        eval_steps=eval_steps,
    )


def parse_sweep_args(
    *,
    default_candidate_weights: Path,
    expected_candidate_sha256: str,
    max_concurrent: int,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=default_candidate_weights)
    parser.add_argument("--expected-candidate-sha256", default=expected_candidate_sha256)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-type", default=TPU_TYPE)
    parser.add_argument("--tpu-region", default=TPU_REGION)
    parser.add_argument("--tpu-zone", default=TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=max_concurrent)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def run_sweep(
    args: argparse.Namespace,
    remaining: list[str],
    *,
    definition: SweepDefinition,
    expected_candidate_sha256: str,
    local_artifact_dir: Path,
) -> None:
    logging.basicConfig(level=logging.INFO)
    sys.argv = [sys.argv[0], *remaining]
    hardware = (args.tpu_type, args.tpu_region, args.tpu_zone)
    if hardware != (TPU_TYPE, TPU_REGION, TPU_ZONE):
        raise ValueError(f"This launcher is pinned to {(TPU_TYPE, TPU_REGION, TPU_ZONE)}, got {hardware}")
    if not 1 <= args.max_concurrent <= definition.expected_run_count:
        raise ValueError(f"--max-concurrent must be in [1, {definition.expected_run_count}]")
    if args.expected_candidate_sha256 != expected_candidate_sha256:
        raise ValueError(
            "The frozen candidate hash cannot be overridden: "
            f"{args.expected_candidate_sha256} != {expected_candidate_sha256}"
        )
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    candidates, _ = load_candidate_mixtures(
        args.candidate_weights,
        expected_candidate_sha256,
        definition=definition,
    )
    base.completed_adamh_heuristic = current_completed_adamh_heuristic
    source_specs = base.load_source_panel(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    run_specs = build_run_specs(
        template=source_specs[0],
        candidates=candidates,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        definition=definition,
    )
    if args.dry_run:
        local_artifact_dir.mkdir(parents=True, exist_ok=True)
        save_sweep_manifest(
            SaveSweepManifestConfig(
                output_path=str(local_artifact_dir),
                candidate_weights_path=str(args.candidate_weights),
                candidate_weights_sha256=args.expected_candidate_sha256,
                source_panel=base.DEFAULT_SOURCE_PANEL,
                source_panel_sha256=base.SOURCE_PANEL_SHA256,
                analysis_output_path=args.analysis_output_path,
                candidates_json=json.dumps([asdict(candidate) for candidate in candidates], sort_keys=True),
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                sweep_definition_json=json.dumps(asdict(definition), sort_keys=True),
            )
        )
        logger.info("Wrote %d runtime-distinct launch rows under %s", len(run_specs), local_artifact_dir)
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
            candidate_weights_sha256=args.expected_candidate_sha256,
            analysis_output_path=args.analysis_output_path,
            validation_configs=validation_configs,
            definition=definition,
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
            f"{definition.experiment_name}: {definition.expected_run_count} runtime-distinct "
            "single-phase DSP optima under "
            "whole-run epoch caps, with Uncheatable and native Table-9 evaluation"
        ),
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    return parse_sweep_args(
        default_candidate_weights=DEFAULT_CANDIDATE_WEIGHTS,
        expected_candidate_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        max_concurrent=MAX_CONCURRENT,
    )


def main() -> None:
    args, remaining = parse_args()
    run_sweep(
        args,
        remaining,
        definition=DEFAULT_SWEEP_DEFINITION,
        expected_candidate_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        local_artifact_dir=LOCAL_ARTIFACT_DIR,
    )


if __name__ == "__main__":
    main()
