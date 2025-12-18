# nodryrun
"""Minimal dense profiling smoke test."""

import argparse
import logging
import math
import sys

from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3
from experiments.llama import LlamaConfig, llama3_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

SEQ_LEN = 512
GLOBAL_BATCH = 16
TOKEN_TARGET = 65_536  # ~0.07M tokens
STEPS = max(20, math.ceil(TOKEN_TARGET / (GLOBAL_BATCH * SEQ_LEN)))
LR = 3e-4
WD = 0.0
TPU_TYPE = "v5p-8"
PROFILE_START = 5
PROFILE_STEPS = 10

model_cfg = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=256,
    intermediate_dim=1024,
    num_layers=4,
    num_heads=4,
    num_kv_heads=4,
)

nemotron_cc_steps = tokenize_nemotron(tokenizer=llama3_tokenizer)
nemotron_cc_mixture = lm_mixture_data_config(
    components=nemotron_cc_steps,
    weights=NEMOTRON_WEIGHTS,
    permutation_type="linear",
)

DATASET_OPTIONS = {
    "nemotron_cc": nemotron_cc_mixture,
    "dclm": dclm_mixture_config_llama3,
}


def make_config(dataset: str) -> SpeedrunConfig:
    tokenized_dataset = DATASET_OPTIONS[dataset]
    return SpeedrunConfig(
        author=Author(name="Marin Team", affiliation="Marin Project", url=None),
        description="Profiling smoke test: tiny dense Llama on v5p-8",
        model_config=model_cfg,
        train_config=SimpleTrainConfig(
            resources=ResourceConfig.with_tpu(tpu_type=TPU_TYPE),
            train_batch_size=GLOBAL_BATCH,
            num_train_steps=STEPS,
            learning_rate=LR,
            weight_decay=WD,
            steps_per_eval=STEPS,
            steps_per_export=STEPS,
            profiler=True,
            profiler_start_step=PROFILE_START,
            profiler_num_steps=PROFILE_STEPS,
        ),
        tokenized_dataset=tokenized_dataset,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Profiling smoke test (dense)")
    parser.add_argument("--dataset", choices=DATASET_OPTIONS.keys(), default="dclm")
    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional run suffix to disambiguate multiple profiling iterations (e.g. timestamp).",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0]] + remaining
    cfg = make_config(args.dataset)
    base_suffix = f"profile_smoke_dense_{TPU_TYPE}_bs{GLOBAL_BATCH}_seq{SEQ_LEN}_{args.dataset}"
    run_suffix = f"{base_suffix}_{args.suffix}" if args.suffix else base_suffix
    logger.info(
        "Launching profiling smoke: dataset=%s, batch=%d, seq=%d, steps=%d",
        args.dataset,
        GLOBAL_BATCH,
        SEQ_LEN,
        STEPS,
    )
    executor_main(steps=default_speedrun(run_suffix, cfg))
