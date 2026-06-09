# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval a Delphi midtrain checkpoint on the paranoid decon val sets.

Runs ON a TPU iris job. Loads the run's final HF export (native levanter
checkpoints are cleaned from the midtrain run dirs; arch is Qwen3) and
evaluates ten datasets as SEPARATE tagged validation sets in one job:

- decon_j050 … decon_j090 — the nine paranoid short-doc caches on the
  0.05-spaced Jaccard grid at
  gs://marin-us-east5/tokenized/nemotron_math_val_decon/. Sweeping the cutoff
  traces val loss vs decontamination aggressiveness per checkpoint.
- nemotron_cc_math_v1/4plus — the original 12,500-window val carve-out,
  copied verbatim from the p33m67 data section (same cache + feistel split),
  as an in-harness anchor ("no filter"). Per-scale contamination effect =
  anchor loss minus decon loss with zero harness confounds.

Losses land as eval/{tag}/loss in W&B and in the JSON tracker file under
gs://…/midtrain_dedup/decon_val_sets/evals/{run}/step-{N}/.

max_eval_length is 4096 explicitly — levanter's 2048 default silently halves
the eval windows and breaks comparability with the training-time val loss.

Launch (one job per run; us-east5 keeps caches + checkpoints in-region):

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --tpu v6e-4 --enable-extra-resources --priority interactive \
        --extra tpu --preemptible --region us-east5 \
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

# Full 0.05-spaced decontamination grid (j050..j090). Each cutoff has its own
# validation-only cache under DECON_CACHE_ROOT/<tag>; one eval_lm job evaluates
# all of them plus the original-val anchor as separate tagged datasets, so a
# checkpoint's val loss can be plotted against the cutoff. j050/j075/j090 are
# the canonical paranoid caches; the rest are filled in by
# scripts/analysis/build_decon_val_sweep.py.
DECON_CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DECON_TAGS = [f"j{round(cutoff * 100):03d}" for cutoff in DECON_CUTOFFS]
DATA_SECTION = Path(__file__).parent / "p33m67_data_section.json"
ANCHOR_COMPONENT = "nemotron_cc_math_v1/4plus"
SEQ_LEN = 4096
JSON_TRACKER_DIR = "metrics.jsonl"
JSON_TRACKER_RESULTS = "eval_results.json"

# Canonical p33m67 K=0.20 ladder, all lr factors, verified complete on
# 2026-06-07: every run's final hf/step-N matches
# round(0.2 x base num_train_steps) from the delphi registry; attempt
# suffixes are fresh-restart counters and the listed attempt is the one with
# a finished export (e.g. 1e22 lr0.33 abdeba / lr0.5 91bcb9 / lr0.67 089468
# have no exports and are excluded).
RUNS = [
    # lr0.33
    "delphi-3e18-p33m67-k0p20-lr33-a003",
    "delphi-9e18-p33m67-k0p20-lr33-a002",
    "delphi-2e19-p33m67-k0p20-lr33-a002",
    "delphi-3e19-p33m67-k0p20-lr33-a002",
    "delphi-9e19-p33m67-k0p20-lr33-a002",
    "delphi-2e20-p33m67-k0p20-lr33-a001",
    "delphi-3e20-p33m67-k0p20-lr33-a001",
    "delphi-1e21-p33m67-9p25b-lr0.33-58ebcb",
    "delphi-1e22-p33m67-32p07b-lr0.33-e9132105",
    # lr0.5
    "delphi-3e18-p33m67-k0p20-lr50-a003",
    "delphi-9e18-p33m67-k0p20-lr50-a002",
    "delphi-2e19-p33m67-k0p20-lr50-a002",
    "delphi-3e19-p33m67-k0p20-lr50-a002",
    "delphi-9e19-p33m67-k0p20-lr50-a002",
    "delphi-2e20-p33m67-k0p20-lr50-a001",
    "delphi-3e20-p33m67-k0p20-lr50-a001",
    "delphi-1e21-p33m67-9p25b-lr0.5-efbc63",
    "delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d",
    # lr0.67
    "delphi-3e18-p33m67-k0p20-lr67-a003",
    "delphi-9e18-p33m67-k0p20-lr67-a002",
    "delphi-2e19-p33m67-k0p20-lr67-a002",
    "delphi-3e19-p33m67-k0p20-lr67-a002",
    "delphi-9e19-p33m67-k0p20-lr67-a002",
    "delphi-2e20-p33m67-k0p20-lr67-a001",
    "delphi-3e20-p33m67-k0p20-lr67-a001",
    "delphi-1e21-p33m67-9p25b-lr0.67-9cf8da",
    "delphi-1e22-p33m67-32p07b-lr0.67-54770ae7",
    # lr0.83
    "delphi-3e18-p33m67-k0p20-lr83-a003",
    "delphi-9e18-p33m67-k0p20-lr83-a002",
    "delphi-2e19-p33m67-k0p20-lr83-a002",
    "delphi-3e19-p33m67-k0p20-lr83-a002",
    "delphi-9e19-p33m67-k0p20-lr83-a002",
    "delphi-2e20-p33m67-k0p20-lr83-a001",
    "delphi-3e20-p33m67-k0p20-lr83-a001",
    "delphi-1e21-p33m67-9p25b-lr0.83-0cb048",
    "delphi-1e22-p33m67-32p07b-lr0.83-78fd44",
    # p50m50 (35/36; 1e22 lr0.67 e78260 incomplete, excluded)
    "delphi-3e18-p50m50-k0p20-lr33-a003",
    "delphi-3e18-p50m50-k0p20-lr50-a003",
    "delphi-3e18-p50m50-k0p20-lr67-a003",
    "delphi-3e18-p50m50-k0p20-lr83-a003",
    "delphi-9e18-p50m50-k0p20-lr33-a002",
    "delphi-9e18-p50m50-k0p20-lr50-a002",
    "delphi-9e18-p50m50-k0p20-lr67-a002",
    "delphi-9e18-p50m50-k0p20-lr83-a002",
    "delphi-2e19-p50m50-k0p20-lr33-a002",
    "delphi-2e19-p50m50-k0p20-lr50-a002",
    "delphi-2e19-p50m50-k0p20-lr67-a002",
    "delphi-2e19-p50m50-k0p20-lr83-a002",
    "delphi-3e19-p50m50-k0p20-lr33-a002",
    "delphi-3e19-p50m50-k0p20-lr50-a002",
    "delphi-3e19-p50m50-k0p20-lr67-a002",
    "delphi-3e19-p50m50-k0p20-lr83-a002",
    "delphi-9e19-p50m50-k0p20-lr33-a002",
    "delphi-9e19-p50m50-k0p20-lr50-a002",
    "delphi-9e19-p50m50-k0p20-lr67-a002",
    "delphi-9e19-p50m50-k0p20-lr83-a002",
    "delphi-2e20-p50m50-k0p20-lr33-a001",
    "delphi-2e20-p50m50-k0p20-lr50-a001",
    "delphi-2e20-p50m50-k0p20-lr67-a001",
    "delphi-2e20-p50m50-k0p20-lr83-a001",
    "delphi-3e20-p50m50-k0p20-lr33-a001",
    "delphi-3e20-p50m50-k0p20-lr50-a001",
    "delphi-3e20-p50m50-k0p20-lr67-a001",
    "delphi-3e20-p50m50-k0p20-lr83-a001",
    "delphi-1e21-p50m50-9p25b-lr0.33-bccff4",
    "delphi-1e21-p50m50-9p25b-lr0.5-973c46",
    "delphi-1e21-p50m50-9p25b-lr0.67-7e82b3",
    "delphi-1e21-p50m50-9p25b-lr0.83-f9edd2",
    "delphi-1e22-p50m50-32p07b-lr0.33-c43ada",
    "delphi-1e22-p50m50-32p07b-lr0.5-ecfa99",
    "delphi-1e22-p50m50-32p07b-lr0.83-3c9f70",
    # p67m33 (36/36)
    "delphi-3e18-p67m33-k0p20-lr33-a003",
    "delphi-3e18-p67m33-k0p20-lr50-a003",
    "delphi-3e18-p67m33-k0p20-lr67-a003",
    "delphi-3e18-p67m33-k0p20-lr83-a003",
    "delphi-9e18-p67m33-k0p20-lr33-a002",
    "delphi-9e18-p67m33-k0p20-lr50-a002",
    "delphi-9e18-p67m33-k0p20-lr67-a002",
    "delphi-9e18-p67m33-k0p20-lr83-a002",
    "delphi-2e19-p67m33-k0p20-lr33-a002",
    "delphi-2e19-p67m33-k0p20-lr50-a002",
    "delphi-2e19-p67m33-k0p20-lr67-a002",
    "delphi-2e19-p67m33-k0p20-lr83-a002",
    "delphi-3e19-p67m33-k0p20-lr33-a002",
    "delphi-3e19-p67m33-k0p20-lr50-a002",
    "delphi-3e19-p67m33-k0p20-lr67-a002",
    "delphi-3e19-p67m33-k0p20-lr83-a002",
    "delphi-9e19-p67m33-k0p20-lr33-a002",
    "delphi-9e19-p67m33-k0p20-lr50-a002",
    "delphi-9e19-p67m33-k0p20-lr67-a002",
    "delphi-9e19-p67m33-k0p20-lr83-a002",
    "delphi-2e20-p67m33-k0p20-lr33-a001",
    "delphi-2e20-p67m33-k0p20-lr50-a001",
    "delphi-2e20-p67m33-k0p20-lr67-a001",
    "delphi-2e20-p67m33-k0p20-lr83-a001",
    "delphi-3e20-p67m33-k0p20-lr33-a001",
    "delphi-3e20-p67m33-k0p20-lr50-a001",
    "delphi-3e20-p67m33-k0p20-lr67-a001",
    "delphi-3e20-p67m33-k0p20-lr83-a001",
    "delphi-1e21-p67m33-9p25b-lr0.33-ab4e64",
    "delphi-1e21-p67m33-9p25b-lr0.5-114e49",
    "delphi-1e21-p67m33-9p25b-lr0.67-ecbd27",
    "delphi-1e21-p67m33-9p25b-lr0.83-a1a261",
    "delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7",
    "delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a",
    "delphi-1e22-p67m33-32p07b-lr0.67-3c17740e",
    "delphi-1e22-p67m33-32p07b-lr0.83-d35daa",
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

    for tag in DECON_TAGS:
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


def eval_output_dir(out_root: str, run: str, step: int) -> str:
    return f"{out_root}/{run}/step-{step}/{JSON_TRACKER_DIR}"


def eval_results_path(output_dir: str) -> str:
    return f"{output_dir}/{JSON_TRACKER_RESULTS}"


def existing_eval_output_paths(output_dir: str) -> list[str]:
    """Return existing tracker artifacts under output_dir.

    JsonFileTracker writes eval_results.json inside output_dir. On object stores,
    the directory marker itself may not exist, so checking only output_dir can
    miss a completed eval.
    """
    existing = []
    if fsspec_exists(output_dir):
        existing.append(output_dir)

    result_path = eval_results_path(output_dir)
    if fsspec_exists(result_path):
        existing.append(result_path)

    existing.extend(fsspec_glob(f"{output_dir}/*"))
    return sorted(set(existing))


def require_unused_output(output_dir: str, *, force: bool) -> None:
    existing = existing_eval_output_paths(output_dir)
    if not existing:
        return

    if force:
        logger.warning("re-running with existing eval output under %s: %s", output_dir, existing)
        return

    preview = "\n".join(f"  - {path}" for path in existing[:10])
    suffix = "" if len(existing) <= 10 else f"\n  ... and {len(existing) - 10} more"
    raise RuntimeError(f"eval output already exists under {output_dir}; pass --force to re-run:\n{preview}{suffix}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--step", type=int, default=None, help="HF export step; default = latest.")
    parser.add_argument("--per-device-parallelism", type=int, default=4)
    parser.add_argument(
        "--out-root",
        default=EVAL_OUT_ROOT,
        help="Root for the per-run JSON tracker output; the 9-cutoff sweep writes to a distinct root.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if the output metrics file exists.")
    args = parser.parse_args()

    step = args.step if args.step is not None else final_hf_step(args.run)
    out_path = eval_output_dir(args.out_root, args.run, step)
    require_unused_output(out_path, force=args.force)

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
