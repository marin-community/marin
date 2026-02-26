"""Evaluate the best single-model checkpoint for each copy-scaling data stream.

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


def _dcr_plus_teacher(
    *,
    teacher_data_name: str,
    teacher_data_weight: float,
    epochs: int,
    weight_decay: float,
) -> DataEfficiencyConfig:
    return DataEfficiencyConfig(
        data_name="dcr",
        teacher_data_name=teacher_data_name,
        teacher_data_weight=teacher_data_weight,
        val_name=["dc_1k_val_normal"],
        epochs=epochs,
        base_train_steps=777,
        train_batch_size=64,
        lr_schedule="cosine",
        lr=3e-3,
        weight_decay=weight_decay,
        model_name="300m4k",
        block_cross_document_attention=False,
        nametag="",
        bs_in_name=True,
    )


EVAL_RUNS: list[tuple[str, DataEfficiencyConfig]] = [
    # Shuffled copy-scaling best runs
    (
        "300m4kcda-203Mx4-dcr+w2s^0.75-cos-lr0.0030-wd0.80-bs64",
        _dcr_plus_teacher(teacher_data_name="w2s", teacher_data_weight=0.75, epochs=4, weight_decay=0.80),
    ),
    (
        "300m4kcda-203Mx8-dcr+s4^0.75-cos-lr0.0030-wd0.80-bs64",
        _dcr_plus_teacher(teacher_data_name="s4", teacher_data_weight=0.75, epochs=8, weight_decay=0.80),
    ),
    (
        "300m4kcda-203Mx8-dcr+s8^0.75-cos-lr0.0030-wd0.80-bs64",
        _dcr_plus_teacher(teacher_data_name="s8", teacher_data_weight=0.75, epochs=8, weight_decay=0.80),
    ),
    (
        "300m4kcda-203Mx16-dcr+s16^0.75-cos-lr0.0030-wd0.40-bs64",
        _dcr_plus_teacher(teacher_data_name="s16", teacher_data_weight=0.75, epochs=16, weight_decay=0.40),
    ),
    (
        "300m4kcda-203Mx16-dcr+s32^0.75-cos-lr0.0030-wd0.40-bs64",
        _dcr_plus_teacher(teacher_data_name="s32", teacher_data_weight=0.75, epochs=16, weight_decay=0.40),
    ),
    # Sorted copy-scaling best runs
    (
        "300m4kcda-203Mx8-dcr+w2^0.75-cos-lr0.0030-wd0.80-bs64",
        _dcr_plus_teacher(teacher_data_name="w2", teacher_data_weight=0.75, epochs=8, weight_decay=0.80),
    ),
    (
        "300m4kcda-203Mx8-dcr+b4^0.75-cos-lr0.0030-wd0.40-bs64",
        _dcr_plus_teacher(teacher_data_name="b4", teacher_data_weight=0.75, epochs=8, weight_decay=0.40),
    ),
    (
        "300m4kcda-203Mx16-dcr+b8^0.75-cos-lr0.0030-wd0.40-bs64",
        _dcr_plus_teacher(teacher_data_name="b8", teacher_data_weight=0.75, epochs=16, weight_decay=0.40),
    ),
    (
        "300m4kcda-203Mx16-dcr+b16^0.75-cos-lr0.0030-wd0.40-bs64",
        _dcr_plus_teacher(teacher_data_name="b16", teacher_data_weight=0.75, epochs=16, weight_decay=0.40),
    ),
    (
        "300m4kcda-203Mx32-dcr+b32^0.9-cos-lr0.0030-wd0.40-bs64",
        _dcr_plus_teacher(teacher_data_name="b32", teacher_data_weight=0.9, epochs=32, weight_decay=0.40),
    ),
]


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

    runs = EVAL_RUNS
    if list_runs:
        for name, _ in runs:
            print(name)
        raise SystemExit(0)

    if run_names:
        lookup = {name: cfg for name, cfg in runs}
        missing = [n for n in run_names if n not in lookup]
        if missing:
            raise ValueError(f"Unknown run_name(s): {', '.join(missing)}")
        runs = [(n, lookup[n]) for n in run_names]

    # Running these TPU evals concurrently can destabilize JAX distributed workers.
    # Keep this script conservative by default; callers can still override via CLI.
    sys.argv = [sys.argv[0], *_inject_default_max_concurrent(remaining, default=1)]

    eval_steps: list[ExecutorStep] = []
    for expected_run_name, cfg in runs:
        built = cfg.build_name()
        assert built == expected_run_name, f"Config name mismatch: expected {expected_run_name}, got {built}"
        train_cfg = cfg.build_train_lm_config()
        eval_steps.append(
            data_efficiency_eval_latest_single_model(
                run_name=expected_run_name,
                eval_data=train_cfg.data,
                model=train_cfg.model,
                resource_config=ResourceConfig.with_tpu("v4-8"),
                eval_label="synth-data-best-models",
                wandb_tags=[
                    "data-efficiency",
                    "single",
                    "latest",
                    "synth-data-best",
                    "dc_1k_val_normal",
                ],
                # Route single-model eval through the ensemble evaluator with one
                # checkpoint so it follows the same TPU job path as eval_ensembles.
                use_ensemble_path=True,
                ensemble_run_prefix="synth-data-best-models",
                ensemble_key="single",
                log_entropy=False,
            )
        )

    executor_main(
        steps=eval_steps,
        description="Eval best single-model checkpoints for copy-scaling data streams",
    )
