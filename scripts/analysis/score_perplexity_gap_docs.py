# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the curated perplexity-gap docs per-token with one Delphi checkpoint.

Calls levanter's ``score_main`` (the model-perplexity scorer behind the
perplexity-gap report) on the curated document set
(``perplexity_gap/curated_docs.jsonl``) for a single Delphi p33m67 lr0.33
checkpoint, writing ``scored_documents.parquet`` (per-byte loss + token byte
spans + text per doc) so per-document and per-token loss can be read back and
plotted across the scaling ladder.

Model config is built from the run's ``hf/step-N/config.json`` (Qwen3), exactly
like ``eval_decon_val_sets.py``; tokenizer is llama3 (the training tokenizer).
``max_eval_length=4096`` explicit. One v6e-4 TPU job per scale.

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --tpu v6e-4 --extra tpu --enable-extra-resources --priority interactive \
        --preemptible --region us-east5 --cpu 8 --memory 64GB --disk 50GB \
        --job-name score-ppl-gap-<scale> \
        -- python scripts/analysis/score_perplexity_gap_docs.py --run <run>
"""

import argparse
import json
import logging
import re

import fsspec
import jmp
from levanter.data.text import DatasetComponent, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.main.perplexity_gap import GapFinderModelConfig, ModelPerplexityConfig, score_main
from levanter.models.qwen import Qwen3Config
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.utils import fsspec_exists, fsspec_glob
from transformers import Qwen3Config as HfQwen3Config

logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = "gs://marin-us-east5/checkpoints"
PPL_GAP_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/perplexity_gap"
CURATED_JSONL = f"{PPL_GAP_ROOT}/curated_docs.jsonl"
SCORES_ROOT = f"{PPL_GAP_ROOT}/scores"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
SEQ_LEN = 4096
SCORED_DOCS_FILE = "scored_documents.parquet"


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
    return Qwen3Config.from_hf_config(HfQwen3Config(**hf_config_dict))


def curated_dataset() -> DatasetComponent:
    fmt = TextLmDatasetFormat(text_key="text")
    source = UrlDatasetSourceConfig(train_urls=[], validation_urls=[CURATED_JSONL], format=fmt)
    return DatasetComponent(source=source, format=fmt, tags=["curated"], split="validation")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, default=None, help="HF export step; default = latest.")
    parser.add_argument("--per-device-parallelism", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-score even if scored_documents.parquet exists.")
    args = parser.parse_args()

    step = args.step if args.step is not None else final_hf_step(args.run)
    out_path = f"{SCORES_ROOT}/{args.run}/step-{step}"
    if fsspec_exists(f"{out_path}/{SCORED_DOCS_FILE}") and not args.force:
        raise FileExistsError(f"{out_path}/{SCORED_DOCS_FILE} exists; pass --force to overwrite")

    model = load_model_config(args.run, step)
    assert model.max_seq_len >= SEQ_LEN, f"model max_seq_len {model.max_seq_len} < {SEQ_LEN}"

    config = ModelPerplexityConfig(
        model=GapFinderModelConfig(
            checkpoint_path=f"{CHECKPOINT_ROOT}/{args.run}/hf/step-{step}",
            model=model,
            checkpoint_is_hf=True,
            tokenizer=TOKENIZER,
        ),
        datasets={"curated": curated_dataset()},
        trainer=TrainerConfig(
            tracker=WandbConfig(project="marin", tags=["ppl_gap_score", args.run], name=f"ppl-gap-score-{args.run}"),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=args.per_device_parallelism,
        ),
        output_path=out_path,
        max_eval_length=SEQ_LEN,
        max_docs_per_dataset=None,
        max_doc_bytes=65_536,
    )
    logger.info("scoring %s step %d -> %s", args.run, step, out_path)
    score_main(config)


if __name__ == "__main__":
    main()
