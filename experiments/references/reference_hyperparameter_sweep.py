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

"""LR and batch size sweep for a ~130M Qwen3 model with MuonH on Nemotron mix."""

import itertools
import json
import os
from dataclasses import dataclass

import fsspec
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import MuonHConfig

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path


# ~130M Qwen3
qwen3_130m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=4,
    num_layers=12,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
    cross_entropy_block_size=32000,  # blockwise CE to reduce memory spike
)

LEARNING_RATES = [0.005, 0.01, 0.02]
BATCH_SIZES = [64, 128, 256]
TARGET_TOKENS = 2_600_000_000
SEQ_LEN = 4096


def best_run(runs, mode="min"):
    """Return the run with the best metric."""
    key = lambda r: r["metric"]
    return min(runs, key=key) if mode == "min" else max(runs, key=key)


@dataclass(frozen=True)
class SweepAnalysisConfig:
    run_paths: list
    hparams_list: list
    metric_file: str
    metric_key: str
    mode: str
    output_path: str


def run_sweep_analysis(config):
    """Analyze sweep runs and find the best one."""
    results = []

    for run_path, hparams in zip(config.run_paths, config.hparams_list):
        metric_path = os.path.join(run_path, config.metric_file)
        try:
            fs, _, _ = fsspec.get_fs_token_paths(metric_path)
            with fs.open(metric_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    continue
                data = json.loads(lines[-1])

            value = data["summary"][config.metric_key]

            results.append({"metric": float(value), "hparams": hparams, "run_path": run_path})
            print(f"Run {hparams}: {config.metric_key} = {value}")
        except Exception as e:
            print(f"Failed to read {run_path}: {e}")

    if not results:
        raise RuntimeError("No valid results found")

    best = best_run(results, config.mode)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    output = {
        "best_hparams": best["hparams"],
        "best_metric": best["metric"],
        "best_run_path": best["run_path"],
        "all_results": sorted(results, key=lambda r: r["metric"], reverse=(config.mode == "max")),
    }

    with fs.open(os.path.join(config.output_path, "sweep_result.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest: {best['hparams']} -> {config.metric_key} = {best['metric']}")


# Generate training runs
training_steps = []
hparams_list = []

for lr, batch_size in itertools.product(LEARNING_RATES, BATCH_SIZES):
    num_steps = TARGET_TOKENS // (batch_size * SEQ_LEN)

    muonh_config = MuonHConfig(
        learning_rate=lr,
        adam_lr=lr * 0.15,
        min_lr_ratio=0.0,
        momentum=0.98,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=2.0,
        warmup=1000,
    )

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=batch_size,
        num_train_steps=num_steps,
        learning_rate=lr,
        train_seq_len=SEQ_LEN,
        z_loss_weight=5e-6,
        optimizer_config=muonh_config,
        steps_per_eval=500,
    )

    step = default_train(
        name=f"ref-sweep-qwen3-130m-lr{lr}-bs{batch_size}",
        tokenized=nemotron_mix,
        model_config=qwen3_130m,
        train_config=train_config,
        tags=["sweep", "qwen3", "130m", "muonh", f"lr={lr}", f"bs={batch_size}"],
        eval_harness_tasks=[],
    )

    training_steps.append(step)
    hparams_list.append({"lr": lr, "batch_size": batch_size})


analysis_step = ExecutorStep(
    name="ref-sweep-qwen3-130m-analysis",
    fn=run_sweep_analysis,
    config=SweepAnalysisConfig(
        run_paths=[s.as_input_name() for s in training_steps],
        hparams_list=hparams_list,
        metric_file="tracker_metrics.jsonl",
        metric_key="eval/paloma/c4_en/bpb",
        mode="min",
        output_path=this_output_path(),
    ),
)

all_steps = [*training_steps, analysis_step]

if __name__ == "__main__":
    executor_main(steps=all_steps)
