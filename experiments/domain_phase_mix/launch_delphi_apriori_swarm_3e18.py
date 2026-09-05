# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Train the new rows of the prospectively frozen 280-row Delphi 3e18 swarm.

Reads the design table written by ``design_delphi_apriori_swarm_280_20260904.py`` and builds one training row per
new mixture (reused rows already have runs and are skipped), with the design's paired seeds: the data seed of a
row is its seed block's seed, the trainer seed is fixed, and the simulated-epoch subset seed equals the data seed,
so half- and quarter-pool rows are nested subsets of the full-support control of the same block. Rows with a pool
fraction below one pass ``simulated_epoch_pool_fractions`` to the mixture config; every other setting (model,
horizon, optimizer, validation sets, Table-9 evaluation) is the augmented-swarm launcher's. ``--wave pilot`` limits
the graph to the pilot rows. ``--dry-run`` writes the resolved manifest locally and launches nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from marin.execution.executor import ExecutorMainConfig, executor_context, executor_main
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as swarm
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_TOTAL_AVAILABLE_TOKENS
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES, PHASE_BOUNDARIES, PHASE_NAMES
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DESIGN_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs" / "delphi_apriori_swarm_280_20260904"
DEFAULT_DESIGN_TABLE = DESIGN_DIR / "swarm_mixtures.csv"
LOCAL_ARTIFACT_DIR = DESIGN_DIR / "launch_dry_run"
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_apriori_swarm_3e18_20260904"
WANDB_TAGS = ("delphi-3e18-apriori-swarm", "fit-panel", "single-phase")
TABLE9_WANDB_GROUP = "olmo_base_eval_table9_delphi_3e18_apriori_swarm"
PROVENANCE_PANEL = "delphi_3e18_apriori_swarm"
RUN_ID_BASE = 7_150_000
# sha256 of the reviewed swarm_mixtures.csv; the launcher runs nothing else.
FROZEN_DESIGN_SHA256 = "c5ec2b0ae1b5c68dc44f6ede1a5caa1bd2918f5bfd9f6d0f23be508637856bd6"
WAVES = ("pilot", "full")
POOL_FRACTION_PREFIX = "pool_fraction_"


def read_design_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def frozen_design_sha256(design_table: Path) -> str:
    """The table's hash, required to equal the pinned ``FROZEN_DESIGN_SHA256`` and the manifest beside the table."""
    actual = hashlib.sha256(design_table.read_bytes()).hexdigest()
    if actual != FROZEN_DESIGN_SHA256:
        raise ValueError(f"{design_table} (sha256 {actual}) is not the reviewed design ({FROZEN_DESIGN_SHA256})")
    manifest_path = design_table.with_name("manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"{design_table} has no manifest.json beside it; the launcher only runs frozen tables")
    expected = json.loads(manifest_path.read_text())["mixtures_sha256"]
    if actual != expected:
        raise ValueError(f"{design_table} (sha256 {actual}) does not match its manifest ({expected})")
    return actual


def pool_fractions_from_row(row: dict[str, str]) -> dict[str, float] | None:
    fractions = {}
    for domain in DOMAIN_NAMES:
        column = f"{POOL_FRACTION_PREFIX}{domain}"
        fraction = float(row[column])
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"{row['run_name']}/{domain}: pool fraction {fraction} outside (0, 1]")
        if fraction < 1.0:
            fractions[domain] = fraction
    return fractions or None


def run_specs_from_rows(
    rows: list[dict[str, str]],
    *,
    wave: str,
    train_steps: int,
    batch_size: int,
    model_hidden_dim: int,
    model_layers: int,
    non_embedding_params: int,
    total_trainable_params: int,
    tensor_parallel_size: int,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[swarm.DelphiSwarmRunSpec]:
    """One run spec per new design row in the wave; seeds and pool fractions come from the table."""
    if wave not in WAVES:
        raise ValueError(f"wave must be one of {WAVES}")
    realized_train_tokens = train_steps * batch_size * swarm.SEQ_LEN_DELPHI
    specs: list[swarm.DelphiSwarmRunSpec] = []
    for table_index, row in enumerate(rows):  # identities come from the unfiltered table, not from the wave
        if row["source"] != "new" or (wave == "pilot" and row["wave"] != "pilot"):
            continue
        phase_weights = swarm._phase_weights_from_row(row, source_run_name=row["run_name"])
        if phase_weights[PHASE_NAMES[0]] != phase_weights[PHASE_NAMES[1]]:
            raise ValueError(f"{row['run_name']}: the design is single-phase; phase weights differ")
        data_seed, trainer_seed, subset_seed = int(row["data_seed"]), int(row["trainer_seed"]), int(row["subset_seed"])
        if data_seed < 0 or subset_seed != data_seed:
            raise ValueError(
                f"{row['run_name']}: paired seeds required (data seed {data_seed}, subset seed {subset_seed})"
            )
        max_epoch, q95_epoch, phase_tv = swarm._weight_diagnostics(phase_weights)
        specs.append(
            swarm.DelphiSwarmRunSpec(
                run_order=table_index,
                run_id=RUN_ID_BASE + table_index,
                run_name=row["run_name"],
                source_run_name=row["run_name"],
                source_experiment=EXPERIMENT_NAME,
                panel_source=row["block"],
                target_flops=swarm.TARGET_FLOPS,
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                batch_size=batch_size,
                train_steps=train_steps,
                realized_train_tokens=realized_train_tokens,
                expected_checkpoint_step=train_steps - 1,
                model_hidden_dim=model_hidden_dim,
                model_layers=model_layers,
                non_embedding_params=non_embedding_params,
                total_trainable_params=total_trainable_params,
                tensor_parallel_size=tensor_parallel_size,
                data_seed=data_seed,
                trainer_seed=trainer_seed,
                phase_boundary=PHASE_BOUNDARIES[0],
                phase_0_fraction=swarm.PHASE_FRACTIONS["phase_0"],
                phase_1_fraction=swarm.PHASE_FRACTIONS["phase_1"],
                simulated_epoch_target_budget=swarm.SIMULATED_EPOCH_TARGET_BUDGET,
                available_top_level_tokens=TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv,
                phase_weights=phase_weights,
                simulated_epoch_subset_seed=subset_seed,
                simulated_epoch_pool_fractions=pool_fractions_from_row(row),
            )
        )
    if not specs:
        raise ValueError(f"no new rows in wave {wave!r}")
    return specs


def load_design(
    design_table: Path, *, wave: str, analysis_output_path: str, tpu_type: str, tpu_region: str, tpu_zone: str
) -> list[swarm.DelphiSwarmRunSpec]:
    frozen_design_sha256(design_table)
    rows = read_design_rows(design_table)
    scaling_fits = swarm._read_scaling_fits(analysis_output_path)
    candidate = swarm._candidate_for_budget(scaling_fits=scaling_fits)
    realized = candidate.train_steps * swarm.TARGET_BATCH_SIZE * swarm.SEQ_LEN_DELPHI
    if realized > swarm.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError(
            f"Delphi train tokens {realized} exceed the simulated-epoch target {swarm.SIMULATED_EPOCH_TARGET_BUDGET}"
        )
    vocab_size = swarm.completed_adamh_heuristic.vocab_size
    return run_specs_from_rows(
        rows,
        wave=wave,
        train_steps=candidate.train_steps,
        batch_size=swarm.TARGET_BATCH_SIZE,
        model_hidden_dim=int(candidate.model_config.hidden_dim),
        model_layers=int(candidate.model_config.num_layers),
        non_embedding_params=int(candidate.model_config.total_trainable_params(0)),
        total_trainable_params=int(candidate.model_config.total_trainable_params(vocab_size)),
        tensor_parallel_size=swarm._tensor_parallel_size(candidate.model_config.hidden_dim, tpu_type),
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )


def write_local_dry_run(
    design_table: Path, run_specs: list[swarm.DelphiSwarmRunSpec], analysis_output_path: str
) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    swarm.save_delphi_swarm_manifest(
        swarm.SaveDelphiSwarmManifestConfig(
            output_path=str(LOCAL_ARTIFACT_DIR),
            experiment_name=EXPERIMENT_NAME,
            source_panel=str(design_table),
            source_panel_sha256=frozen_design_sha256(design_table),
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            architecture_target_flops=swarm.TARGET_FLOPS,
        )
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-table", type=Path, default=DEFAULT_DESIGN_TABLE)
    parser.add_argument("--wave", choices=WAVES, default="pilot")
    parser.add_argument("--analysis-output-path", default=swarm.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-type", default=swarm.TARGET_TPU_TYPE)
    parser.add_argument("--tpu-region", default=swarm.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=swarm.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=swarm.DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != swarm.DEFAULT_TPU_REGION or args.tpu_zone != swarm.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {swarm.DEFAULT_TPU_REGION}/{swarm.DEFAULT_TPU_ZONE}")
    if args.max_concurrent < 1 or args.max_concurrent > swarm.DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {swarm.DEFAULT_MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    run_specs = load_design(
        args.design_table,
        wave=args.wave,
        analysis_output_path=args.analysis_output_path,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        write_local_dry_run(args.design_table, run_specs, args.analysis_output_path)
        logger.info("Wrote %d dry-run run specs for wave %s under %s", len(run_specs), args.wave, LOCAL_ARTIFACT_DIR)
        return
    validation_steps = swarm._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = swarm.build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=args.analysis_output_path,
            source_panel=str(args.design_table),
            validation_configs=validation_configs,
            experiment_name=EXPERIMENT_NAME,
            wandb_tags=WANDB_TAGS,
            training_wandb_group=f"{PROVENANCE_PANEL}_{args.wave}",
            table9_wandb_group=TABLE9_WANDB_GROUP,
            provenance_panel=PROVENANCE_PANEL,
            source_panel_sha256=frozen_design_sha256(args.design_table),
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: {args.wave} wave of the frozen 280-row swarm at Delphi 3e18 "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
