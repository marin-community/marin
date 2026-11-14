# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run RandRAND on existing FineWeb-Edu / Paloma caches with current Levanter.

Without --run, print the resolved config without loading data or launching work.
Cache roots must contain train/ and validation/, respectively. Use the same
tokenized caches for the candidate and the baseline; this entrypoint deliberately
does not download, rebuild, or substitute evaluation data.
"""

import argparse
import json
from enum import StrEnum

import draccus
import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.main.train_lm import main as train_main
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import OptimizerConfig
from levanter.optim.randrand import RandRANDConfig, RandRANDVariant
from levanter.optim.randrand_muon_hybrid import RandRANDMuonHybridConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig


class Recipe(StrEnum):
    PR6301 = "pr6301"
    HYBRID = "hybrid"
    RANDRAND_MUON_HYBRID = "randrand_muon_hybrid"
    SPECTRAL_RANDRAND_MUON_HYBRID = "spectral_randrand_muon_hybrid"


def optimizer_config(recipe: Recipe) -> RandRANDConfig:
    """Pair each recipe with the best recorded local RandRAND settings."""
    if recipe == Recipe.PR6301:
        return RandRANDConfig(learning_rate=0.006, embed_lr=0.6, vector_lr=0.006)
    if recipe == Recipe.RANDRAND_MUON_HYBRID:
        return RandRANDMuonHybridConfig()
    if recipe == Recipe.SPECTRAL_RANDRAND_MUON_HYBRID:
        return RandRANDMuonHybridConfig(
            variant=RandRANDVariant.SPECTRAL,
            learning_rate=0.0045,
            matrix_betas=(0.95, 0.99),
            matrix_eps=1e-8,
            weight_decay=0.0303,
            embed_lr=0.2,
            vector_lr=0.03,
            head_lr=0.006,
            rank=64,
            rank_sketch=256,
            randrand_warmup=200,
            basis_every=200,
            basis_ema_beta=0.98,
        )
    return RandRANDConfig()


def model_config(recipe: Recipe) -> Qwen3Config:
    """Construct the historical size with an explicit normalization choice."""
    return Qwen3Config(
        max_seq_len=4096,
        hidden_dim=512,
        intermediate_dim=1792,
        num_layers=6,
        num_heads=8,
        num_kv_heads=8,
        hybrid_norm=recipe != Recipe.PR6301,
        use_qk_norm=False,
        input_embedding_norm=False,
        tie_word_embeddings=False,
        use_bias=False,
        use_layer_norm_weight=True,
        layer_norm_epsilon=1e-5,
        scan_layers=True,
        gradient_checkpointing=True,
    )


def build_config(
    *,
    recipe: Recipe,
    train_cache: str,
    c4_cache: str,
    output_path: str,
    run_id: str,
    optimizer: OptimizerConfig | None = None,
    per_device_batch: int = -1,
    steps_per_eval: int = 1000,
) -> TrainLmConfig:
    """Bind supplied benchmark caches to the current training/evaluation API."""
    data = LmDataConfig(
        tokenizer="marin-community/marin-tokenizer",
        cache_dir=None,
        auto_build_caches=False,
        shuffle=True,
        permutation_type="linear",
        block_cross_document_attention=True,
        enforce_eos=True,
        components={
            "fineweb_edu": DatasetComponent(cache_dir=train_cache, source=UrlDatasetSourceConfig()),
            "paloma/c4_en": DatasetComponent(cache_dir=c4_cache, source=UrlDatasetSourceConfig()),
        },
        train_weights={"fineweb_edu": 1.0, "paloma/c4_en": 0.0},
    )
    return TrainLmConfig(
        model=model_config(recipe),
        train_seq_len=4096,
        data=data,
        data_seed=42,
        optimizer=optimizer_config(recipe) if optimizer is None else optimizer,
        z_loss_weight=0.0,
        hf_save_path=None,
        trainer=TrainerConfig(
            id=run_id,
            seed=42 if recipe in (Recipe.RANDRAND_MUON_HYBRID, Recipe.SPECTRAL_RANDRAND_MUON_HYBRID) else 0,
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=128,
            num_train_steps=4959,
            per_device_parallelism=per_device_batch,
            steps_per_eval=steps_per_eval,
            max_eval_batches=None,
            checkpointer=CheckpointerConfig(base_path=output_path),
            tracker=WandbConfig(project="speedrun", group="randrand-qwen3-130m", name=run_id),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        dest="recipe",
        type=Recipe,
        choices=list(Recipe),
        required=True,
        help="Recipe and its corresponding best-run optimizer settings",
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--c4-cache", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--optimizer-config", help="Optional OptimizerConfig YAML, e.g. a matched Muon baseline")
    parser.add_argument("--per-device-batch", type=int, default=-1)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    optimizer = (
        draccus.parse(OptimizerConfig, config_path=args.optimizer_config, args=[])
        if args.optimizer_config is not None
        else None
    )
    config = build_config(
        recipe=args.recipe,
        train_cache=args.train_cache,
        c4_cache=args.c4_cache,
        output_path=args.output_path,
        run_id=args.run_id,
        optimizer=optimizer,
        per_device_batch=args.per_device_batch,
        steps_per_eval=args.eval_every,
    )
    if not args.run:
        print(json.dumps(draccus.encode(config), indent=2, default=str))
        return
    # Empty-source configs may omit absent splits. Never allow this benchmark
    # to silently train without the requested official C4 validation cache.
    if "paloma/c4_en" not in config.data.build_caches("validation"):
        raise ValueError("The supplied C4 cache must contain a completed validation/ split")
    train_main(config)


if __name__ == "__main__":
    main()
