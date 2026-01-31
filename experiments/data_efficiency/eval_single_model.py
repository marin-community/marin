# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a single data-efficiency model checkpoint on one or more val sets.

This mirrors `eval_test.py` (ensemble evaluation) but for a single model, and it
selects the most recently-written HuggingFace checkpoint under `hf/step-*`.

Intended usage:

```bash
uv run python experiments/data_efficiency/eval_single_model.py \
  --prefix gs://marin-us-central2 \
  --dry_run true
```
"""

import argparse
import dataclasses
import sys
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig

from experiments.data_efficiency.train import DataEfficiencyConfig
from marin.evaluation.log_probs import EvalLmConfig, evaluate_lm_log_probs
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path


@dataclass
class EvalLatestHfLogProbsConfig:
    """Eval config that discovers the latest HF checkpoint under a run directory."""

    checkpoint_root: str | InputName
    model: LmConfig
    datasets: LMMixtureDatasetConfig
    resource_config: ResourceConfig
    per_device_batch_size: int = 4
    max_samples_per_dataset: int | None = None
    checkpoint_is_hf: bool = True
    name: str | None = None
    wandb_tags: list[str] | None = None
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore


def evaluate_latest_hf_log_probs(config: EvalLatestHfLogProbsConfig) -> None:
    """Resolve latest `hf/step-*` checkpoint and run the standard LM eval."""
    checkpoints = discover_hf_checkpoints(str(config.checkpoint_root))
    if not checkpoints:
        raise FileNotFoundError(f"No HF checkpoints found under {config.checkpoint_root}")

    latest_checkpoint = checkpoints[-1]
    evaluate_lm_log_probs(
        EvalLmConfig(
            name=config.name,
            checkpoint_path=latest_checkpoint,
            model=config.model,
            datasets=config.datasets,
            resource_config=config.resource_config,
            per_device_batch_size=config.per_device_batch_size,
            max_samples_per_dataset=config.max_samples_per_dataset,
            checkpoint_is_hf=config.checkpoint_is_hf,
            wandb_tags=config.wandb_tags,
            output_path=config.output_path,
        )
    )


def _data_efficiency_config_for_run() -> DataEfficiencyConfig:
    # This run name is expected to match `DataEfficiencyConfig.build_name()`.
    #
    # - 196M: 4096 * 750 * 64 = 196,608,000 tokens/epoch -> int(M) formatting yields "196M"
    # - cda: cross-document attention is *enabled* (i.e., not blocked)
    return DataEfficiencyConfig(
        data_name="dc_shuffled",
        val_name=["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled", "tav"],
        epochs=16,
        base_train_steps=750,
        train_batch_size=64,
        lr_schedule="cosine",
        lr=3e-3,
        weight_decay=1.60,
        model_name="300m4k",
        block_cross_document_attention=False,
        # keep name stable: "-bs64" included, no seed suffix
        nametag="",
        bs_in_name=True,
    )


def _data_efficiency_config_for_run_dc_shuffled_plus_hq4() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-196Mx16-dc_shuffled+hq4^0.5-cos-lr0.0030-wd1.60-bs64
    return DataEfficiencyConfig(
        data_name="dc_shuffled",
        teacher_data_name="hq4",
        teacher_data_weight=0.5,
        val_name=["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled", "tav"],
        epochs=16,
        base_train_steps=750,
        train_batch_size=64,
        lr_schedule="cosine",
        lr=3e-3,
        weight_decay=1.60,
        model_name="300m4k",
        block_cross_document_attention=False,
        nametag="",
        bs_in_name=True,
    )


def _data_efficiency_config_for_run_dc_shuffled_plus_wic1() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-196Mx16-dc_shuffled+wic1^0.5-cos-lr0.0030-wd1.60-bs64
    return DataEfficiencyConfig(
        data_name="dc_shuffled",
        teacher_data_name="wic1",
        teacher_data_weight=0.5,
        val_name=["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled", "tav"],
        epochs=16,
        base_train_steps=750,
        train_batch_size=64,
        lr_schedule="cosine",
        lr=3e-3,
        weight_decay=1.60,
        model_name="300m4k",
        block_cross_document_attention=False,
        # keep name stable: "-bs64" included, no seed suffix
        nametag="",
        bs_in_name=True,
    )


def _data_efficiency_config_for_run_wrap_ic1() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-196Mx16-wrap_ic1-cos-lr0.0030-wd1.60-bs64
    return DataEfficiencyConfig(
        data_name="wrap_ic1",
        val_name=["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled", "tav"],
        epochs=16,
        base_train_steps=750,
        train_batch_size=64,
        lr_schedule="cosine",
        lr=3e-3,
        weight_decay=1.60,
        model_name="300m4k",
        block_cross_document_attention=False,
        nametag="",
        bs_in_name=True,
    )


def _data_efficiency_config_for_run_dcr_plus_teacher(
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
        val_name=["dc_1k_val_normal", "dc_1k_val_normal_doc"],
        epochs=epochs,
        base_train_steps=775,
        train_batch_size=64,
        lr_schedule="cosine",
        lr=3e-3,
        weight_decay=weight_decay,
        model_name="300m4k",
        block_cross_document_attention=False,
        nametag="",
        bs_in_name=True,
    )


def _data_efficiency_config_for_run_dcr_plus_hqr_075_8x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx8-dcr+hqr^0.75-cos-lr0.0030-wd0.40-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="hqr",
        teacher_data_weight=0.75,
        epochs=8,
        weight_decay=0.40,
    )


def _data_efficiency_config_for_run_dcr_plus_hqs_05_16x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx16-dcr+hqs^0.5-cos-lr0.0030-wd0.80-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="hqs",
        teacher_data_weight=0.5,
        epochs=16,
        weight_decay=0.80,
    )


def _data_efficiency_config_for_run_dcr_plus_hqr_05_16x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx16-dcr+hqr^0.5-cos-lr0.0030-wd0.80-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="hqr",
        teacher_data_weight=0.5,
        epochs=16,
        weight_decay=0.80,
    )


def _data_efficiency_config_for_run_dcr_plus_b8_075_16x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx16-dcr+b8^0.75-cos-lr0.0030-wd0.40-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="b8",
        teacher_data_weight=0.75,
        epochs=16,
        weight_decay=0.40,
    )


def _data_efficiency_config_for_run_dcr_plus_b16_075_16x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx16-dcr+b16^0.75-cos-lr0.0030-wd0.40-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="b16",
        teacher_data_weight=0.75,
        epochs=16,
        weight_decay=0.40,
    )


def _data_efficiency_config_for_run_dcr_plus_s16_075_16x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx16-dcr+s16^0.75-cos-lr0.0030-wd0.40-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="s16",
        teacher_data_weight=0.75,
        epochs=16,
        weight_decay=0.40,
    )


def _data_efficiency_config_for_run_dcr_plus_w2_075_8x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx8-dcr+w2^0.75-cos-lr0.0030-wd0.80-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="w2",
        teacher_data_weight=0.75,
        epochs=8,
        weight_decay=0.80,
    )


def _data_efficiency_config_for_run_dcr_plus_s8_075_8x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx8-dcr+s8^0.75-cos-lr0.0030-wd0.80-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="s8",
        teacher_data_weight=0.75,
        epochs=8,
        weight_decay=0.80,
    )


def _data_efficiency_config_for_run_dcr_plus_w2s_075_4x() -> DataEfficiencyConfig:
    # Expected name:
    # 300m4kcda-203Mx4-dcr+w2s^0.75-cos-lr0.0030-wd0.80-bs64
    return _data_efficiency_config_for_run_dcr_plus_teacher(
        teacher_data_name="w2s",
        teacher_data_weight=0.75,
        epochs=4,
        weight_decay=0.80,
    )



def data_efficiency_eval_latest_single_model(
    *,
    run_name: str,
    eval_data: LMMixtureDatasetConfig,
    model: LmConfig,
    resource_config: ResourceConfig,
    eval_label: str = "tav",
    wandb_tags: list[str] | None = None,
) -> ExecutorStep:
    """Create an Executor step that evals the latest HF checkpoint for a run."""
    checkpoint_root = InputName.hardcoded(f"checkpoints/data_efficiency/{run_name}/hf")
    if wandb_tags is None:
        wandb_tags = [
            "data-efficiency",
            "single",
            "latest",
            "dc_1k_val_normal",
            "dc_t10x_val",
            "dc_t10x_val_shuffled",
            "tav",
        ]
    return ExecutorStep(
        name=f"analysis/log_probs/data-efficiency/{eval_label}-eval-latest/{run_name}",
        fn=evaluate_latest_hf_log_probs,
        config=EvalLatestHfLogProbsConfig(
            checkpoint_root=checkpoint_root,  # type: ignore[arg-type]
            model=model,
            datasets=eval_data,
            resource_config=resource_config,
            checkpoint_is_hf=True,
            name=f"{eval_label}-eval-{run_name}",
            wandb_tags=wandb_tags,
        ),
    )


def _get_eval_runs() -> list[tuple[str, DataEfficiencyConfig]]:
    return [
        (
            "300m4kcda-203Mx16-dcr+b8^0.75-cos-lr0.0030-wd0.40-bs64",
            _data_efficiency_config_for_run_dcr_plus_b8_075_16x(),
        ),
        (
            "300m4kcda-203Mx16-dcr+b16^0.75-cos-lr0.0030-wd0.40-bs64",
            _data_efficiency_config_for_run_dcr_plus_b16_075_16x(),
        ),
        (
            "300m4kcda-203Mx16-dcr+hqr^0.5-cos-lr0.0030-wd0.80-bs64",
            _data_efficiency_config_for_run_dcr_plus_hqr_05_16x(),
        ),
        (
            "300m4kcda-203Mx8-dcr+hqr^0.75-cos-lr0.0030-wd0.40-bs64",
            _data_efficiency_config_for_run_dcr_plus_hqr_075_8x(),
        ),
        (
            "300m4kcda-203Mx8-dcr+w2^0.75-cos-lr0.0030-wd0.80-bs64",
            _data_efficiency_config_for_run_dcr_plus_w2_075_8x(),
        ),
        (
            "300m4kcda-203Mx8-dcr+s8^0.75-cos-lr0.0030-wd0.80-bs64",
            _data_efficiency_config_for_run_dcr_plus_s8_075_8x(),
        ),
        (
            "300m4kcda-203Mx16-dcr+s16^0.75-cos-lr0.0030-wd0.40-bs64",
            _data_efficiency_config_for_run_dcr_plus_s16_075_16x(),
        ),
        (
            "300m4kcda-203Mx16-dcr+hqs^0.5-cos-lr0.0030-wd0.80-bs64",
            _data_efficiency_config_for_run_dcr_plus_hqs_05_16x(),
        ),
        (
            "300m4kcda-203Mx4-dcr+w2s^0.75-cos-lr0.0030-wd0.80-bs64",
            _data_efficiency_config_for_run_dcr_plus_w2s_075_4x(),
        ),
    ]


def _select_runs(
    runs: list[tuple[str, DataEfficiencyConfig]],
    run_names: list[str] | None,
) -> list[tuple[str, DataEfficiencyConfig]]:
    if not run_names:
        return runs

    run_lookup = {name: cfg for name, cfg in runs}
    missing = [name for name in run_names if name not in run_lookup]
    if missing:
        raise ValueError(f"Unknown run_name(s): {', '.join(missing)}")

    return [(name, run_lookup[name]) for name in run_names]


def _parse_run_args() -> tuple[list[str] | None, bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--run_name",
        action="append",
        default=None,
        help="Exact run name to evaluate. Repeatable.",
    )
    parser.add_argument(
        "--list_runs",
        action="store_true",
        help="Print all run names and exit.",
    )
    args, remaining = parser.parse_known_args()
    return args.run_name, args.list_runs, remaining


if __name__ == "__main__":
    run_names, list_runs, remaining = _parse_run_args()
    runs = _get_eval_runs()
    if list_runs:
        for run_name, _ in runs:
            print(run_name)
        raise SystemExit(0)

    runs = _select_runs(runs, run_names)
    sys.argv = [sys.argv[0], *remaining]

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
                eval_label="01-30-26-v4-evals",
                wandb_tags=[
                    "data-efficiency",
                    "single",
                    "latest",
                    "dc_1k_val_normal",
                    "dc_1k_val_normal_doc",
                ],
            )
        )

    executor_main(
        steps=eval_steps,
        description="Eval latest checkpoint(s) on dc_1k_val_normal + dc_1k_val_normal_doc",
    )

