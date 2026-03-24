"""Evaluate synth-data models: seed-sweep runs from `synth_data_var.py` and
the best single-model runs from each copy-scaling series.

Usage:
    uv run python experiments/data_efficiency/synth_data_eval_models.py \
        --prefix gs://marin-us-central2 --dry_run true

By default this script runs one eval step at a time (`--max_concurrent 1`) to avoid
launching many TPU eval jobs simultaneously. Override with `--max_concurrent N`.
"""

import argparse
import sys

from fray.cluster import ResourceConfig

from experiments.data_efficiency.eval_single_model import data_efficiency_eval_latest_single_model
from experiments.data_efficiency.train import DataEfficiencyConfig
from marin.execution.executor import ExecutorStep, executor_main


VAL_NAMES = ["dc_1k_val_normal", "dc_500_val_short", "dc_500_val_long"]

# ---------------------------------------------------------------------------
# Seed-sweep candidates (from synth_data_var.py)
# ---------------------------------------------------------------------------

SYNTH_DATA_VAR_CANDIDATES: list[tuple[str, float, int, int, float, float, str]] = [
    ("f8", 0.75, 777, 8, 3e-3, 0.40, "300m4k"),
    ("b8", 0.75, 777, 16, 3e-3, 0.40, "300m4k"),
    ("s8", 0.75, 777, 8, 3e-3, 0.80, "300m4k"),
    ("z8", 0.75, 777, 8, 3e-3, 0.40, "300m4k"),
]

SYNTH_DATA_VAR_SEEDS = range(3)

# ---------------------------------------------------------------------------
# Best single-model runs from each copy-scaling series
# (stream, mix_ratio, base_train_steps, epochs, lr, wd, model_name)
# ---------------------------------------------------------------------------

BEST_COPY_SCALING_CANDIDATES: list[tuple[str, float, int, int, float, float, str]] = [
    ("w2s", 0.75, 777,  4, 3e-3, 0.80, "300m4k"),
    ("s4",  0.75, 777,  8, 3e-3, 0.80, "300m4k"),
    ("s8",  0.75, 777,  8, 3e-3, 0.80, "300m4k"),
    ("s16", 0.75, 777, 16, 3e-3, 0.40, "300m4k"),
    ("s32", 0.75, 777, 16, 3e-3, 0.40, "300m4k"),
    ("w2",  0.75, 777,  8, 3e-3, 0.80, "300m4k"),
    ("b4",  0.75, 777,  8, 3e-3, 0.40, "300m4k"),
    ("b8",  0.75, 777, 16, 3e-3, 0.40, "300m4k"),
    ("b16", 0.75, 777, 16, 3e-3, 0.40, "300m4k"),
    ("b32", 0.90, 777, 32, 1e-3, 0.40, "300m4k"),
    ("z2",  0.75, 777,  4, 3e-3, 0.80, "300m4k"),
    ("z4",  0.75, 777,  8, 3e-3, 0.80, "300m4k"),
    ("z8",  0.75, 777,  8, 3e-3, 0.40, "300m4k"),
    ("z16", 0.75, 777, 16, 3e-3, 0.40, "300m4k"),
    ("z32", 0.90, 777, 32, 1e-3, 0.40, "300m4k"),
]


def _synth_data_var_config(
    *,
    synthetic_data_name: str,
    synthetic_data_weight: float,
    base_train_steps: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    model_name: str,
    seed: int,
) -> DataEfficiencyConfig:
    return DataEfficiencyConfig(
        train_seed=seed,
        data_seed=seed,
        data_name="dcr",
        val_name=VAL_NAMES,
        teacher_data_name=synthetic_data_name,
        teacher_data_weight=synthetic_data_weight,
        block_cross_document_attention=False,
        epochs=epochs,
        base_train_steps=base_train_steps,
        lr_schedule="cosine",
        lr=lr,
        weight_decay=weight_decay,
        train_batch_size=64,
        train_seq_len=4096,
        steps_per_eval=1,
        model_name=model_name,
        nametag=f"-seed{seed}",
        bs_in_name=False,
    )


def _best_model_config(
    *,
    synthetic_data_name: str,
    synthetic_data_weight: float,
    base_train_steps: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    model_name: str,
) -> DataEfficiencyConfig:
    return DataEfficiencyConfig(
        data_name="dcr",
        val_name=VAL_NAMES,
        teacher_data_name=synthetic_data_name,
        teacher_data_weight=synthetic_data_weight,
        block_cross_document_attention=False,
        epochs=epochs,
        base_train_steps=base_train_steps,
        lr_schedule="cosine",
        lr=lr,
        weight_decay=weight_decay,
        train_batch_size=64,
        train_seq_len=4096,
        steps_per_eval=1,
        model_name=model_name,
    )


def _build_eval_runs() -> list[tuple[str, DataEfficiencyConfig]]:
    runs: list[tuple[str, DataEfficiencyConfig]] = []

    # Seed-sweep runs
    """
    for synthetic_data_name, synthetic_data_weight, base_train_steps, epochs, lr, weight_decay, model_name in (
        SYNTH_DATA_VAR_CANDIDATES
    ):
        for seed in SYNTH_DATA_VAR_SEEDS:
            cfg = _synth_data_var_config(
                synthetic_data_name=synthetic_data_name,
                synthetic_data_weight=synthetic_data_weight,
                base_train_steps=base_train_steps,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                model_name=model_name,
                seed=seed,
            )
            runs.append((cfg.build_name(), cfg))
    """

    # Best copy-scaling runs
    for synthetic_data_name, synthetic_data_weight, base_train_steps, epochs, lr, weight_decay, model_name in (
        BEST_COPY_SCALING_CANDIDATES
    ):
        cfg = _best_model_config(
            synthetic_data_name=synthetic_data_name,
            synthetic_data_weight=synthetic_data_weight,
            base_train_steps=base_train_steps,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            model_name=model_name,
        )
        runs.append((cfg.build_name(), cfg))

    return runs


def _parse_args() -> tuple[list[str] | None, bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run_name", action="append", default=None, help="Exact run name to evaluate. Repeatable.")
    parser.add_argument("--list_runs", action="store_true", help="Print all run names and exit.")
    args, remaining = parser.parse_known_args()
    return args.run_name, args.list_runs, remaining


def _inject_default_max_concurrent(argv: list[str], default: int) -> list[str]:
    has_max_concurrent = any(arg == "--max_concurrent" or arg.startswith("--max_concurrent=") for arg in argv)
    if has_max_concurrent:
        return argv
    return [*argv, "--max_concurrent", str(default)]


if __name__ == "__main__":
    run_names, list_runs, remaining = _parse_args()

    runs = _build_eval_runs()
    if list_runs:
        for name, _ in runs:
            print(name)
        raise SystemExit(0)

    if run_names:
        lookup = {name: cfg for name, cfg in runs}
        missing = [name for name in run_names if name not in lookup]
        if missing:
            raise ValueError(f"Unknown run_name(s): {', '.join(missing)}")
        runs = [(name, lookup[name]) for name in run_names]

    # Running these TPU evals concurrently can destabilize JAX distributed workers.
    # Keep this script conservative by default; callers can still override via CLI.
    sys.argv = [sys.argv[0], *_inject_default_max_concurrent(remaining, default=1)]

    eval_steps: list[ExecutorStep] = []
    for run_name, cfg in runs:
        train_cfg = cfg.build_train_lm_config()
        eval_steps.append(
            data_efficiency_eval_latest_single_model(
                run_name=run_name,
                eval_data=train_cfg.data,
                model=train_cfg.model,
                resource_config=ResourceConfig.with_tpu("v4-8"),
                # eval_label="synth-data-var-models",
                eval_label="synth-data-best-models",
                wandb_tags=[
                    "data-efficiency",
                    "single",
                    "latest",
                    # "synth-data-var",
                    "synth-data-best",
                    *VAL_NAMES,
                ],
                # Route single-model eval through the ensemble evaluator with one
                # checkpoint so it follows the same TPU job path as eval_ensembles.
                use_ensemble_path=True,
                # ensemble_run_prefix="synth-data-var-models",
                ensemble_run_prefix="synth-data-best-models",
                ensemble_key="single",
                log_entropy=False,
            )
        )

    executor_main(
        steps=eval_steps,
        description="Eval all synth-data-best single-model checkpoints",
    )
