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

"""Convenience driver for sweeping OLMoE 1B/7B hyperparameters."""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
from collections.abc import Iterable

from experiments.speedrun.olmoe_1b7b_nemotron_40b import (
    DATASET_OPTIONS,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    DEFAULT_PROFILER_NUM_STEPS,
    DEFAULT_PROFILER_START_STEP,
    MODEL_OPTIONS,
    SEQ_LEN,
    TOKEN_TARGET,
    TPU_TYPE,
    nemotron_only_speedrun,
    make_speedrun_config,
)
from marin.execution.executor import executor_main

DEFAULT_GLOBAL_BATCH_SIZE = 128
DEFAULT_LEARNING_RATES = (1e-4, 2e-4, 3e-4)
DEFAULT_BETA2_VALUES = (0.95, 0.999)
DEFAULT_TPU_TYPE = TPU_TYPE


def _float_list(values: Iterable[str]) -> list[float]:
    return [float(v) for v in values]


def build_variant_config(
    *,
    learning_rate: float,
    beta2: float,
    seq_len: int,
    global_batch_size: int,
    num_train_steps: int,
    dataset_name: str,
    tpu_type: str,
    model: str,
    profile: bool,
    profiler_start_step: int,
    profiler_num_steps: int,
):
    cfg = make_speedrun_config(
        model=model,
        global_batch_size=global_batch_size,
        num_train_steps=num_train_steps,
        profiler=profile,
        profiler_start_step=profiler_start_step,
        profiler_num_steps=profiler_num_steps,
        dataset_name=dataset_name,
        seq_len=seq_len,
        tpu_type=tpu_type,
    )
    train_cfg = dataclasses.replace(
        cfg.train_config,
        train_batch_size=global_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        beta2=beta2,
        train_seq_len=seq_len,
    )
    return dataclasses.replace(cfg, train_config=train_cfg)


def main():
    parser = argparse.ArgumentParser(description="Sweep OLMoE 1B/7B settings on Nemotron-40B.")
    parser.add_argument("--model", choices=MODEL_OPTIONS, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", choices=DATASET_OPTIONS.keys(), default=DEFAULT_DATASET)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--learning-rates", nargs="*", default=[str(v) for v in DEFAULT_LEARNING_RATES])
    parser.add_argument("--beta2-values", nargs="*", default=[str(v) for v in DEFAULT_BETA2_VALUES])
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--tpu-type", type=str, default=DEFAULT_TPU_TYPE)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=DEFAULT_PROFILER_START_STEP)
    parser.add_argument("--profile-num-steps", type=int, default=DEFAULT_PROFILER_NUM_STEPS)
    parser.add_argument("--append-ici-ag-pipelining-flags", action="store_true")
    parser.add_argument("--append-async-collective-permute-flag", action="store_true")
    args = parser.parse_args()

    learning_rates = _float_list(args.learning_rates)
    beta2_values = _float_list(args.beta2_values)

    seq_len = args.seq_len
    batch = args.global_batch_size
    if args.num_train_steps is not None:
        num_steps = args.num_train_steps
    else:
        num_steps = math.ceil(TOKEN_TARGET / (batch * seq_len))

    for lr, beta2 in itertools.product(learning_rates, beta2_values):
        config = build_variant_config(
            learning_rate=lr,
            beta2=beta2,
            seq_len=seq_len,
            global_batch_size=batch,
            num_train_steps=num_steps,
            dataset_name=args.dataset,
            tpu_type=args.tpu_type,
            model=args.model,
            profile=args.profile,
            profiler_start_step=args.profile_start_step,
            profiler_num_steps=args.profile_num_steps,
        )
        suffix = f"{args.model}_nemotron40b_{args.tpu_type}_bs{batch}_seq{seq_len}_lr{lr:.0e}_beta2-{beta2}"
        executor_main(
            steps=nemotron_only_speedrun(
                suffix,
                config,
                append_ici_ag_pipelining_flags=args.append_ici_ag_pipelining_flags,
                append_async_collective_permute_flag=args.append_async_collective_permute_flag,
            )
        )


if __name__ == "__main__":
    main()
