# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI entry to run the Table 9 BPB evaluator directly on a TPU pod.

Used for one-off / canary runs via ``iris job run --tpu ... -- python -m
marin.evaluation.olmo_base_eval.cli ...``. For executor-managed runs use
``experiments/evals/olmo_base_eval_table9.py`` instead.
"""

from __future__ import annotations

import argparse

import jmp
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.evaluation.olmo_base_eval.run import DEFAULT_WANDB_PROJECT, OlmoBaseEvalConfig, olmo_base_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--request-set-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="olmo_base_eval_table9")
    parser.add_argument("--max-eval-length", type=int, default=8192)
    # fp32 reproduces the SC oracle exactly (the SC HF provider scored in fp32).
    parser.add_argument("--dtype", default="f32")
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    args = parser.parse_args()

    config = OlmoBaseEvalConfig(
        name=args.name,
        checkpoint_path=args.checkpoint,
        checkpoint_is_hf=True,
        request_set_dir=args.request_set_dir,
        output_path=args.output_path,
        tokenizer=args.tokenizer,
        max_eval_length=args.max_eval_length,
        provenance={"checkpoint": args.checkpoint, "request_set_dir": args.request_set_dir, "dtype": args.dtype},
        trainer=TrainerConfig(
            tracker=(
                WandbConfig(
                    project=args.wandb_project,
                    name=args.name,
                    tags=["olmo_base_eval_table9"],
                    group=args.wandb_group,
                ),
                JsonFileTrackerConfig(output_path=args.output_path),
            ),
            per_device_eval_parallelism=args.per_device_batch_size,
            mp=jmp.get_policy(args.dtype),
        ),
    )
    olmo_base_eval(config)


if __name__ == "__main__":
    main()
