# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate validation loss for every checkpoint in a folder.

Self-contained eval script with no Marin pipeline dependencies.
Discovers all Levanter (or HuggingFace) checkpoints under a GCS (or local)
directory, evaluates each on the unified validation datasets, and logs
per-step results to a new wandb run.

Usage:
    # Full eval on all Levanter checkpoints (US)
    uv run experiments/unified/eval_all_checkpoints.py \
        --checkpoint_folder gs://marin-us-central2/checkpoints/unified-qwen3-1.7b-...-cb843a \
        --model_size 1.7b \
        --wandb_project marin \
        --wandb_name "eval-all-ckpts"

    # Eval HuggingFace checkpoints
    uv run experiments/unified/eval_all_checkpoints.py \
        --checkpoint_folder gs://marin-us-central2/checkpoints/unified-qwen3-1.7b-...-cb843a/hf \
        --model_size 1.7b \
        --checkpoint_is_hf

    # Use EU data caches (gs://marin-vlm-eu)
    uv run experiments/unified/eval_all_checkpoints.py \
        --checkpoint_folder gs://marin-eu-west4/checkpoints/unified-qwen3-1.7b-...-cb843a/hf \
        --model_size 1.7b \
        --checkpoint_is_hf \
        --region eu

    # Quick smoke test (2 batches per dataset)
    uv run experiments/unified/eval_all_checkpoints.py \
        --checkpoint_folder gs://marin-us-central2/checkpoints/unified-qwen3-1.7b-...-cb843a \
        --model_size 1.7b \
        --max_eval_batches 2

    # Skip text-only eval benchmarks to speed up
    uv run experiments/unified/eval_all_checkpoints.py \
        --checkpoint_folder gs://marin-us-central2/checkpoints/unified-qwen3-1.7b-...-cb843a \
        --model_size 1.7b \
        --no_text_eval_benchmarks

    # Resume a previous run (skips already-evaluated steps)
    uv run experiments/unified/eval_all_checkpoints.py \
        --checkpoint_folder gs://marin-us-central2/checkpoints/unified-qwen3-1.7b-...-cb843a/hf \
        --model_size 1.7b \
        --checkpoint_is_hf \
        --resume_wandb_run <wandb_run_id>
"""

import argparse
import json
import logging
import os
import re
from typing import Callable

import equinox as eqx
import fsspec
import jax
import jmp
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.eval import TaggedEvaluator, eval_model
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.jax_utils import use_cpu_device

from experiments.qwen3 import qwen3_0_6b, qwen3_1_7b, qwen3_4b
from marin.evaluation.utils import discover_hf_checkpoints, discover_levanter_checkpoints

logger = logging.getLogger(__name__)

# --- US Cache Paths (no Marin pipeline dependencies) ---
US_UNIFIED_TOKENIZER_PATH = "gs://marin-us-central2/tokenizers/llama3-unified-144k"
US_UNIFIED_CACHE_PATH = "gs://marin-vlm/hf_85m_levanter_cache_v2"
US_VISUAL_ONLY_CACHE_PATH = "gs://marin-vlm/hf_85m_levanter_cache_v2/visual_only"
US_UNIFIED_EVAL_CACHE_PATH = "gs://marin-vlm/unified_eval_cache"
US_TEXT_EVAL_CACHE_PATH = "gs://marin-vlm/text_eval_cache"

# --- EU Cache Paths ---
EU_UNIFIED_TOKENIZER_PATH = "gs://marin-eu-west4/tokenizers/llama3-unified-144k"
EU_UNIFIED_CACHE_PATH = "gs://marin-vlm-eu/hf_85m_levanter_cache_v2"
EU_VISUAL_ONLY_CACHE_PATH = "gs://marin-vlm-eu/hf_85m_levanter_cache_v2/visual_only"
EU_UNIFIED_EVAL_CACHE_PATH = "gs://marin-vlm-eu/unified_eval_cache"
EU_TEXT_EVAL_CACHE_PATH = "gs://marin-vlm-eu/text_eval_cache"

LLAMA3_VOCAB_SIZE = 128_256
VISION_START_ID = 128_004
VISION_END_ID = 128_005
ENDOFTEXT_ID = 128_001
VISUAL_TOKEN_OFFSET = 128_256

DEFAULT_EVAL_BENCHMARKS = [
    "textvqa",
    "chartqa",
    "ai2d",
    "mmmu",
    "cifar10_small",
]

DEFAULT_TEXT_EVAL_BENCHMARKS = [
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]

UNDERSTANDING_BENCHMARKS = {"textvqa", "chartqa", "ai2d", "mmmu"}
GENERATION_BENCHMARKS = {"cifar10_small"}

MODEL_CONFIGS = {
    "0.6b": qwen3_0_6b,
    "1.7b": qwen3_1_7b,
    "4b": qwen3_4b,
}

REGION_CACHE_PATHS = {
    "us": {
        "tokenizer_path": US_UNIFIED_TOKENIZER_PATH,
        "multimodal_cache_path": US_UNIFIED_CACHE_PATH,
        "eval_cache_path": US_UNIFIED_EVAL_CACHE_PATH,
        "text_eval_cache_path": US_TEXT_EVAL_CACHE_PATH,
    },
    "eu": {
        "tokenizer_path": EU_UNIFIED_TOKENIZER_PATH,
        "multimodal_cache_path": EU_UNIFIED_CACHE_PATH,
        "eval_cache_path": EU_UNIFIED_EVAL_CACHE_PATH,
        "text_eval_cache_path": EU_TEXT_EVAL_CACHE_PATH,
    },
}


# --- Loss weight transforms ---


def _zero_fractional_transform(weights: np.ndarray) -> np.ndarray:
    mask = (weights > 0) & (weights < 1.0)
    if not mask.any():
        return weights
    result = weights.copy()
    result[mask] = 0.0
    return result


def _only_fractional_transform(weights: np.ndarray) -> np.ndarray:
    result = np.zeros_like(weights)
    mask = (weights > 0) & (weights < 1.0)
    result[mask] = 1.0
    return result


def _swap_primary_secondary_transform(weights: np.ndarray) -> np.ndarray:
    result = weights.copy()
    fractional_mask = (weights > 0) & (weights < 1.0)
    primary_mask = weights == 1.0
    if fractional_mask.any():
        frac_val = weights[fractional_mask][0]
        result[fractional_mask] = 1.0
        result[primary_mask] = frac_val
    return result


def _make_visual_weight_transform(w_visual: float) -> Callable[[np.ndarray], np.ndarray]:
    def transform(weights: np.ndarray) -> np.ndarray:
        result = weights.copy()
        mask = (weights > 0) & (weights < 1.0)
        result[mask] = w_visual
        return result

    return transform


def _compose_transforms(
    *transforms: Callable[[np.ndarray], np.ndarray] | None,
) -> Callable[[np.ndarray], np.ndarray] | None:
    active = [t for t in transforms if t is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def composed(weights: np.ndarray) -> np.ndarray:
        for t in active:
            weights = t(weights)
        return weights

    return composed


def _replace_visual_with_eos(input_ids: np.ndarray) -> np.ndarray:
    result = input_ids.copy()
    visual_mask = (result >= VISUAL_TOKEN_OFFSET) | (result == VISION_START_ID) | (result == VISION_END_ID)
    result[visual_mask] = ENDOFTEXT_ID
    return result


def _replace_text_with_eos(input_ids: np.ndarray) -> np.ndarray:
    result = input_ids.copy()
    text_mask = (
        (result > 0)
        & (result < VISUAL_TOKEN_OFFSET)
        & (result != VISION_START_ID)
        & (result != VISION_END_ID)
        & (result != ENDOFTEXT_ID)
    )
    result[text_mask] = ENDOFTEXT_ID
    return result


def build_eval_data_config(
    tokenizer_path: str = US_UNIFIED_TOKENIZER_PATH,
    multimodal_cache_path: str = US_UNIFIED_CACHE_PATH,
    eval_cache_path: str = US_UNIFIED_EVAL_CACHE_PATH,
    text_eval_cache_path: str = US_TEXT_EVAL_CACHE_PATH,
    w_visual: float | None = 1.0,
    eval_benchmarks: list[str] | None = None,
    text_eval_benchmarks: list[str] | None = None,
) -> LmDataConfig:
    """Build eval-only data config with no Marin pipeline dependencies."""
    w_visual_transform = _make_visual_weight_transform(w_visual) if w_visual is not None else None

    # Validation formats
    und_val_format = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=w_visual_transform,
    )
    gen_val_format = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, w_visual_transform),
    )

    und_val_cache = f"{multimodal_cache_path}/understanding/val_understanding"
    gen_val_cache = f"{multimodal_cache_path}/generation/val_generation"

    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}

    # --- Multimodal validation ---
    components["val_understanding"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=und_val_format,
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding"] = 0.0

    components["val_generation"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=gen_val_format,
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation"] = 0.0

    # --- Per-modality breakdown ---
    components["val_understanding_text_only"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_zero_fractional_transform,
        ),
        pack=True,
        tags=["multimodal", "understanding"],
    )
    components["val_understanding_visual_only"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_only_fractional_transform,
        ),
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding_text_only"] = 0.0
    weights["val_understanding_visual_only"] = 0.0

    components["val_generation_text_only"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, _zero_fractional_transform),
        ),
        pack=True,
        tags=["multimodal", "generation"],
    )
    components["val_generation_visual_only"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, _only_fractional_transform),
        ),
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation_text_only"] = 0.0
    weights["val_generation_visual_only"] = 0.0

    # --- Ablation val: wo_visual / wo_language ---
    # wo_visual: mask visual → compute TEXT loss (unconditional on visual context)
    components["val_understanding_wo_visual"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_zero_fractional_transform,
            input_ids_transform=_replace_visual_with_eos,
        ),
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding_wo_visual"] = 0.0

    # wo_language: mask text → compute VISUAL loss (unconditional on text context)
    components["val_generation_wo_language"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, _zero_fractional_transform),
            input_ids_transform=_replace_text_with_eos,
        ),
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation_wo_language"] = 0.0

    # --- VLM eval benchmarks ---
    if eval_benchmarks is not None:
        for bench in eval_benchmarks:
            tags = ["eval"]
            if bench in UNDERSTANDING_BENCHMARKS:
                tags.append("understanding")
                eval_fmt = und_val_format
            elif bench in GENERATION_BENCHMARKS:
                tags.append("generation")
                eval_fmt = gen_val_format
            else:
                eval_fmt = und_val_format

            components[f"eval_{bench}"] = DatasetComponent(
                cache_dir=f"{eval_cache_path}/{bench}",
                format=eval_fmt,
                pack=True,
                tags=tags,
            )
            weights[f"eval_{bench}"] = 0.0

    # --- Text eval benchmarks ---
    if text_eval_benchmarks is not None:
        text_prebuilt_format = PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
        )
        for bench in text_eval_benchmarks:
            components[f"text_eval_{bench}"] = DatasetComponent(
                cache_dir=f"{text_eval_cache_path}/{bench}",
                format=text_prebuilt_format,
                pack=True,
                tags=["eval", "text"],
            )
            weights[f"text_eval_{bench}"] = 0.0

    return LmDataConfig(
        tokenizer=tokenizer_path,
        components=components,
        train_weights=weights,
        shuffle=False,
        block_cross_document_attention=True,
    )


def _get_evaluated_steps(wandb_project: str, run_id: str) -> set[int]:
    """Fetch steps that already have eval/loss logged in a wandb run."""
    import wandb

    api = wandb.Api()
    run = api.run(f"{wandb_project}/{run_id}")
    evaluated = set()
    for row in run.scan_history(keys=["eval/loss", "_step"], min_step=0):
        if "_step" in row:
            evaluated.add(row["_step"])
    return evaluated


def _load_levanter_checkpoint_step(ckpt_path: str) -> int:
    """Read the training step from a Levanter checkpoint's metadata.json."""
    fs, _, _ = fsspec.get_fs_token_paths(ckpt_path)
    with fs.open(os.path.join(ckpt_path, "metadata.json")) as f:
        metadata = json.load(f)
    return metadata["step"]


def _parse_step_from_path(ckpt_path: str) -> int:
    """Parse the training step from a checkpoint path like '.../step-6000/'."""
    parts = ckpt_path.rstrip("/").split("/")
    for part in reversed(parts):
        m = re.match(r"step[-_](\d+)$", part)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot parse step number from checkpoint path: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate validation loss for every checkpoint in a folder.")
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        required=True,
        help="Path to the checkpoint folder (local or gs://)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="1.7b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size to evaluate (default: 1.7b)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="marin",
        help="Wandb project name (default: marin)",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Wandb run name. Defaults to 'eval-<folder_basename>'",
    )
    parser.add_argument(
        "--per_device_eval_parallelism",
        type=int,
        default=4,
        help="Number of examples per device for eval (default: 4)",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Max batches per eval dataset. None = all (default: None)",
    )
    parser.add_argument(
        "--no_eval_benchmarks",
        action="store_true",
        help="Skip VLM eval benchmarks (textvqa, ai2d, etc.)",
    )
    parser.add_argument(
        "--no_text_eval_benchmarks",
        action="store_true",
        help="Skip text eval benchmarks (hellaswag, mmlu, etc.)",
    )
    parser.add_argument(
        "--checkpoint_is_hf",
        action="store_true",
        help="Treat checkpoints as HuggingFace format (config.json) instead of Levanter format (metadata.json)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us",
        choices=list(REGION_CACHE_PATHS.keys()),
        help="Region for eval data caches: 'us' (gs://marin-vlm) or 'eu' (gs://marin-vlm-eu) (default: us)",
    )
    parser.add_argument(
        "--resume_wandb_run",
        type=str,
        default=None,
        help="Resume a previous wandb run by ID, skipping already-evaluated steps",
    )
    parser.add_argument(
        "--w_visual",
        type=float,
        default=1.0,
        help="Visual token loss weight (default: 1.0). Must match the value used during training.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging (dry run)",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model_size]

    # --- Set up TrainerConfig with wandb ---
    folder_parts = args.checkpoint_folder.rstrip("/").split("/")
    folder_basename = next(
        (p for p in reversed(folder_parts) if p not in ("hf", "checkpoints", "")),
        folder_parts[-1],
    )
    wandb_name = args.wandb_name or f"eval-{folder_basename}"

    if args.no_wandb:
        tracker_config: WandbConfig | tuple = ()
    elif args.resume_wandb_run:
        tracker_config = WandbConfig(
            project=args.wandb_project,
            name=wandb_name,
            id=args.resume_wandb_run,
            resume="must",
        )
    else:
        tracker_config = WandbConfig(
            project=args.wandb_project,
            name=wandb_name,
            resume="never",
        )

    trainer_config = TrainerConfig(
        tracker=tracker_config,
        per_device_eval_parallelism=args.per_device_eval_parallelism,
        per_device_parallelism=args.per_device_eval_parallelism,
        mp=jmp.get_policy("f32"),
        max_eval_batches=args.max_eval_batches,
        ray=RayConfig(auto_start_cluster=False),
    )
    levanter.initialize(trainer_config)

    # --- Discover checkpoints ---
    logger.info("Discovering checkpoints in %s ...", args.checkpoint_folder)
    if args.checkpoint_is_hf:
        ckpt_paths = discover_hf_checkpoints(args.checkpoint_folder)
        ckpt_format = "HuggingFace"
    else:
        ckpt_paths = discover_levanter_checkpoints(args.checkpoint_folder)
        ckpt_format = "Levanter"

    if not ckpt_paths:
        logger.error("No %s checkpoints found in %s", ckpt_format, args.checkpoint_folder)
        return

    if args.checkpoint_is_hf:
        ckpt_steps = [(path, _parse_step_from_path(path)) for path in ckpt_paths]
    else:
        ckpt_steps = [(path, _load_levanter_checkpoint_step(path)) for path in ckpt_paths]
    ckpt_steps.sort(key=lambda x: x[1])
    logger.info(
        "Found %d %s checkpoints: steps %s",
        len(ckpt_steps),
        ckpt_format,
        [s for _, s in ckpt_steps],
    )

    # --- Resume: skip already-evaluated steps ---
    if args.resume_wandb_run and not args.no_wandb:
        logger.info("Fetching already-evaluated steps from wandb run %s ...", args.resume_wandb_run)
        evaluated_steps = _get_evaluated_steps(args.wandb_project, args.resume_wandb_run)
        before = len(ckpt_steps)
        ckpt_steps = [(p, s) for p, s in ckpt_steps if s not in evaluated_steps]
        skipped = before - len(ckpt_steps)
        logger.info(
            "Skipping %d already-evaluated steps, %d remaining: %s",
            skipped,
            len(ckpt_steps),
            [s for _, s in ckpt_steps],
        )
        if not ckpt_steps:
            logger.info("All checkpoints already evaluated. Nothing to do.")
            return

    # --- Build data config (eval-only, no pipeline deps) ---
    eval_benchmarks = None if args.no_eval_benchmarks else DEFAULT_EVAL_BENCHMARKS
    text_eval_benchmarks = None if args.no_text_eval_benchmarks else DEFAULT_TEXT_EVAL_BENCHMARKS

    cache_paths = REGION_CACHE_PATHS[args.region]
    data_config = build_eval_data_config(
        w_visual=args.w_visual,
        eval_benchmarks=eval_benchmarks,
        text_eval_benchmarks=text_eval_benchmarks,
        **cache_paths,
    )

    Pos = model_config.max_Pos
    tokenizer = data_config.the_tokenizer

    Batch = trainer_config.EvalBatch
    datasets = data_config.tagged_eval_sets(Pos)

    if not datasets:
        logger.error("No eval datasets found in data config!")
        return

    max_examples = None
    if args.max_eval_batches is not None:
        max_examples = args.max_eval_batches * trainer_config.eval_batch_size

    compute_axis_mapping = trainer_config.compute_axis_mapping
    parameter_axis_mapping = trainer_config.parameter_axis_mapping

    logger.info(
        "Eval setup: %d datasets, batch_size=%d, max_eval_batches=%s",
        len(datasets),
        trainer_config.eval_batch_size,
        args.max_eval_batches,
    )

    # --- Evaluate all checkpoints ---
    with trainer_config.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        evaluator = TaggedEvaluator(
            Batch,
            datasets,
            tokenizer,
            max_examples_per_dataset=max_examples,
            axis_mapping=compute_axis_mapping,
        )

        key = jax.random.PRNGKey(0)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info("Rounded vocab size from %d to %d for partitioning", vocab_size, Vocab.size)

        if not args.checkpoint_is_hf:
            with use_cpu_device():
                model_shape = eqx.filter_eval_shape(model_config.build, Vocab, key=key)

        for ckpt_path, step in ckpt_steps:
            logger.info("=== Evaluating step %d  (%s) ===", step, ckpt_path)

            if args.checkpoint_is_hf:
                converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
                converter = converter.replaced(reference_checkpoint=ckpt_path, tokenizer=tokenizer)
                model = converter.load_pretrained(
                    model_config.model_type,
                    ref=ckpt_path,
                    dtype=trainer_config.mp.compute_dtype,
                    axis_mapping=parameter_axis_mapping,
                )
            else:
                with use_cpu_device():
                    model = load_checkpoint(model_shape, ckpt_path, subpath="model")
                model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

            log_dict = eval_model(evaluator, model, prefix="eval")

            # Log to wandb (and any other active trackers)
            tracker = levanter.tracker.current_tracker()
            logger.info("Logging %d metrics to tracker %s at step %d", len(log_dict), type(tracker).__name__, step)
            levanter.tracker.log(log_dict, step=step)

            loss = log_dict.get("eval/loss", float("nan"))
            logger.info("Step %d: eval/loss = %.4f", step, loss)

            for key_name in sorted(log_dict.keys()):
                if key_name.endswith("/loss") and key_name != "eval/loss":
                    logger.info("  %s = %.4f", key_name, log_dict[key_name])

    levanter.tracker.current_tracker().finish()
    logger.info("Done. Evaluated %d checkpoints.", len(ckpt_steps))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
