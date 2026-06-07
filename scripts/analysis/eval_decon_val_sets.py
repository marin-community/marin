# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval a Delphi midtrain checkpoint on the paranoid decon val sets.

Runs ON a TPU iris job. Loads the run's final HF export (native levanter
checkpoints are cleaned from the midtrain run dirs; arch is Qwen3) and
evaluates four datasets as SEPARATE tagged validation sets in one job:

- decon_j050 / decon_j075 / decon_j090 — the paranoid short-doc caches at
  gs://marin-us-east5/tokenized/nemotron_math_val_decon/.
- nemotron_cc_math_v1/4plus — the original 12,500-window val carve-out,
  copied verbatim from the p33m67 data section (same cache + feistel split),
  as an in-harness anchor. Per-scale contamination effect = anchor loss
  minus decon loss with zero harness confounds.

Losses land as eval/{tag}/loss in W&B and in the JSON tracker file under
gs://…/midtrain_dedup/decon_val_sets/evals/{run}/step-{N}/.

max_eval_length is 4096 explicitly — levanter's 2048 default silently halves
the eval windows and breaks comparability with the training-time val loss.

Launch (one job per run; us-east5 keeps caches + checkpoints in-region):

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --tpu v6e-4 --enable-extra-resources --priority interactive \
        --preemptible --region us-east5 \
        --job-name decon-eval-<run> \
        -e WANDB_API_KEY <KEY> -e HF_TOKEN <TOKEN> \
        -- python scripts/analysis/eval_decon_val_sets.py --run <run-name>
"""

import argparse
import json
import logging
import re
from pathlib import Path

import draccus
import fsspec
import jmp
from levanter.compat.hf_checkpoints import RepoRef
from levanter.data.text import LmDataConfig
from levanter.main.eval_lm import EvalLmConfig
from levanter.main.eval_lm import main as eval_lm_main
from levanter.models.qwen import Qwen3Config
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.utils import fsspec_exists, fsspec_glob
from transformers import Qwen3Config as HfQwen3Config

logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = "gs://marin-us-east5/checkpoints"
DECON_CACHE_ROOT = "gs://marin-us-east5/tokenized/nemotron_math_val_decon"
EVAL_OUT_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals"
DATA_SECTION = Path(__file__).parent / "p33m67_data_section.json"
ANCHOR_COMPONENT = "nemotron_cc_math_v1/4plus"
SEQ_LEN = 4096

# Canonical p33m67 lr0.33 K=0.20 ladder, verified complete on 2026-06-07:
# every run's final hf/step-N matches round(0.2 x base num_train_steps) from
# the delphi registry; attempt suffixes are fresh-restart counters and the
# listed attempt is the one with a finished export.
RUNS = [
    "delphi-3e18-p33m67-k0p20-lr33-a003",
    "delphi-9e18-p33m67-k0p20-lr33-a002",
    "delphi-2e19-p33m67-k0p20-lr33-a002",
    "delphi-3e19-p33m67-k0p20-lr33-a002",
    "delphi-9e19-p33m67-k0p20-lr33-a002",
    "delphi-2e20-p33m67-k0p20-lr33-a001",
    "delphi-3e20-p33m67-k0p20-lr33-a001",
    "delphi-1e21-p33m67-9p25b-lr0.33-58ebcb",
    "delphi-1e22-p33m67-32p07b-lr0.33-e9132105",
]

# LmDataConfig has no use for these midtrain-spec bookkeeping keys.
DATA_SECTION_EXTRA_KEYS = ("experiment_budget", "target_budget")


def build_data_config() -> LmDataConfig:
    """Anchor component verbatim from the p33m67 data section + 3 decon caches."""
    section = json.loads(DATA_SECTION.read_text())
    for key in DATA_SECTION_EXTRA_KEYS:
        section.pop(key, None)

    section["components"] = {ANCHOR_COMPONENT: section["components"][ANCHOR_COMPONENT]}
    section["train_weights"] = {ANCHOR_COMPONENT: 1.0}
    section["num_validation_sequences"] = {ANCHOR_COMPONENT: section["num_validation_sequences"][ANCHOR_COMPONENT]}
    section["max_train_batches"] = None

    for tag in ("j050", "j075", "j090"):
        name = f"decon_{tag}"
        section["components"][name] = {
            "cache_dir": f"{DECON_CACHE_ROOT}/{tag}",
            "format": {"text_key": "text"},
            "pack": None,
            "source": None,
            "split": "validation",
            "tags": [],
        }
        section["train_weights"][name] = 0.0

    return draccus.decode(LmDataConfig, section)


def final_hf_step(run: str) -> int:
    steps = [
        int(m.group(1)) for p in fsspec_glob(f"{CHECKPOINT_ROOT}/{run}/hf/step-*") if (m := re.search(r"step-(\d+)$", p))
    ]
    if not steps:
        raise FileNotFoundError(f"no hf exports under {CHECKPOINT_ROOT}/{run}/hf/")
    return max(steps)


def load_model_config(run: str, step: int) -> Qwen3Config:
    with fsspec.open(f"{CHECKPOINT_ROOT}/{run}/hf/step-{step}/config.json") as f:
        hf_config_dict = json.load(f)
    hf_config = HfQwen3Config(**hf_config_dict)
    return Qwen3Config.from_hf_config(hf_config)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--step", type=int, default=None, help="HF export step; default = latest.")
    parser.add_argument("--per-device-parallelism", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-run even if the output metrics file exists.")
    args = parser.parse_args()

    step = args.step if args.step is not None else final_hf_step(args.run)
    out_path = f"{EVAL_OUT_ROOT}/{args.run}/step-{step}/metrics.jsonl"
    if fsspec_exists(out_path) and not args.force:
        raise RuntimeError(f"{out_path} already exists; pass --force to re-run")

    model = load_model_config(args.run, step)
    assert model.max_seq_len >= SEQ_LEN, f"model max_seq_len {model.max_seq_len} < {SEQ_LEN}"

    config = EvalLmConfig(
        hf_checkpoint=RepoRef.from_string(f"{CHECKPOINT_ROOT}/{args.run}/hf/step-{step}"),
        model=model,
        data=build_data_config(),
        max_eval_length=SEQ_LEN,
        trainer=TrainerConfig(
            tracker=(
                WandbConfig(project="marin", tags=["decon_val_eval", args.run], name=f"decon-eval-{args.run}"),
                JsonFileTrackerConfig(output_path=out_path),
            ),
            # Match training-time eval: params f32, compute bf16.
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=args.per_device_parallelism,
        ),
    )
    logger.info("evaluating %s step %d -> %s", args.run, step, out_path)
    eval_lm_main(config)


if __name__ == "__main__":
    main()
