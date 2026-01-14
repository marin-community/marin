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

"""Iso-token sweep: vary hidden_size/width across fixed token fractions (1/16, 1/8, 1/4 of a dataset)."""

import logging
import math
from dataclasses import dataclass

from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig

from experiments.defaults import default_train, _prepare_data_config
from experiments.llama import compute_num_parameters
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main
from fray.cluster import ResourceConfig

from experiments.plantcad.exp2101_plantcad_isoflop_sweep import (
    prepare_plantcad_dataset,
    tokenize_plantcad_dataset,
    IsoFlopTokenizeConfig,
    IsoFlopDataConfig,
    pick_v5p_type,
    format_num,
)

logger = logging.getLogger("ray")

# Sweep parameters
HIDDEN_SIZES = [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
TOKEN_FRACTIONS = [1.0 / 2]
TRAIN_STEPS = 8192
MLP_RATIO = 4
HIDDEN_HEAD_RATIO = 128
LR_CONSTANT = 0.33
LR_MAX = 0.02


@dataclass(frozen=True)
class WidthSweepConfig:
    tokenized_dataset: ExecutorStep
    vocab_size: int
    seq_len: int
    dataset_tokens: int
    experiment_name: str = "plantcad_width_sweep_v1.0"


def round_to_power_of_two(x: float) -> int:
    return max(1, 2 ** round(math.log2(x)))


def generate_width_sweep_steps(cfg: WidthSweepConfig) -> list[ExecutorStep]:
    """Generate training steps for width sweep."""
    steps: list[ExecutorStep] = []
    rows: list[dict] = []

    for hidden_size in HIDDEN_SIZES:
        for token_fraction in TOKEN_FRACTIONS:
            target_tokens = int(cfg.dataset_tokens * token_fraction)
            batch_size = round_to_power_of_two(target_tokens / (TRAIN_STEPS * cfg.seq_len))

            # LR scales with sqrt(batch) / hidden
            lr = min(LR_MAX, (LR_CONSTANT * math.sqrt(batch_size)) / hidden_size)
            beta2 = 0.98 ** (batch_size / 128)

            # Derive architecture from hidden size
            intermediate_dim = hidden_size * MLP_RATIO
            n_heads = max(1, hidden_size // HIDDEN_HEAD_RATIO)
            hs_pow = math.log2(hidden_size)
            num_layers = max(1, round(hidden_size / (64 + (hs_pow * 4) - 8)))

            model_cfg = Qwen3Config(
                max_seq_len=cfg.seq_len,
                hidden_dim=hidden_size,
                intermediate_dim=intermediate_dim,
                num_heads=n_heads,
                num_kv_heads=n_heads,
                num_layers=num_layers,
            )

            tpu_type = pick_v5p_type(model_cfg, hidden_size, num_layers, batch_size, cfg.seq_len, cfg.vocab_size)
            num_params = compute_num_parameters(model_cfg, cfg.vocab_size)
            actual_tokens = TRAIN_STEPS * batch_size * cfg.seq_len

            optimizer_cfg = CautiousConfig(
                learning_rate=lr,
                weight_decay=0.1,
                min_lr_ratio=0.0,
                warmup=0.1,
                beta1=0.95,
                beta2=beta2,
                epsilon=1e-15,
                max_grad_norm=1,
                adamc_weight_decay=True,
                lr_schedule="linear",
                decay=0.2,
            )

            train_cfg = SimpleTrainConfig(
                resources=ResourceConfig.with_tpu(tpu_type),
                train_batch_size=batch_size,
                num_train_steps=TRAIN_STEPS,
                learning_rate=lr,
                weight_decay=0.1,
                min_lr_ratio=0.0,
                lr_schedule="linear",
                decay=0.2,
                steps_per_eval=TRAIN_STEPS // 2,
                per_device_eval_parallelism=512,
                max_eval_batches=64,
                optimizer_config=optimizer_cfg,
            )

            pretraining_data = _prepare_data_config(cfg.tokenized_dataset, use_default_validation=False)

            frac_label = f"{int(1/token_fraction)}x"
            step = default_train(
                name=f"{cfg.experiment_name}-H{hidden_size}-F{frac_label}-P{format_num(num_params)}",
                tokenized=pretraining_data,
                model_config=model_cfg,
                train_config=train_cfg,
                tags=(
                    f"hidden_size={hidden_size}",
                    f"token_fraction={token_fraction}",
                    f"params={num_params}",
                    f"tokens={actual_tokens}",
                    f"batch_size={batch_size}",
                    f"tpu={tpu_type}",
                ),
                use_default_validation=False,
                eval_harness_tasks=[],
            )
            steps.append(step)

            rows.append(
                {
                    "hidden": hidden_size,
                    "layers": num_layers,
                    "heads": n_heads,
                    "frac": f"{token_fraction:.2f}",
                    "batch": batch_size,
                    "lr": f"{lr:.2e}",
                    "beta2": f"{beta2:.3f}",
                    "params": format_num(num_params),
                    "target_tok": format_num(target_tokens),
                    "actual_tok": format_num(actual_tokens),
                    "tpu": tpu_type,
                }
            )

    # Print config table
    if rows:
        headers = list(rows[0].keys())
        col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
        header_line = " | ".join(h.rjust(col_widths[h]) for h in headers)
        sep_line = "-+-".join("-" * col_widths[h] for h in headers)
        print(f"\n{header_line}\n{sep_line}")
        for r in rows:
            print(" | ".join(str(r[h]).rjust(col_widths[h]) for h in headers))
        print()

    logger.info(f"Generated {len(steps)} width sweep configurations")
    return steps


def main():
    plantcad_prepared = prepare_plantcad_dataset()
    plantcad_tokenized = tokenize_plantcad_dataset(prepared=plantcad_prepared)

    # Effective dataset tokens after cropping
    data_cfg: IsoFlopDataConfig = plantcad_prepared.config
    tok_cfg: IsoFlopTokenizeConfig = plantcad_tokenized.config
    dataset_tokens = int(data_cfg.total_token_count * (data_cfg.output_seq_len / data_cfg.input_seq_len))

    sweep_cfg = WidthSweepConfig(
        tokenized_dataset=plantcad_tokenized,
        vocab_size=tok_cfg.vocab_size,
        seq_len=data_cfg.output_seq_len,
        dataset_tokens=dataset_tokens,
    )

    sweep_steps = generate_width_sweep_steps(sweep_cfg)
    executor_main(steps=[plantcad_prepared, plantcad_tokenized, *sweep_steps])


if __name__ == "__main__":
    main()
