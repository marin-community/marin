"""Sweep configs for ragged-dot Mixtral MoE runs based on the original 300M setup."""

import logging
from collections.abc import Sequence

from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="Pranshu Chaturvedi", affiliation="Stanford University", url="https://stanford.edu/~pranshu")
LOGGER = logging.getLogger("ray")

VOCAB_SIZE = 32_000
SEQ_LEN = 1024

MODEL_ORDER = (
    "mixtral_300m",
    "mixtral_1b",
    "mixtral_1_5b",
)

BASE_MODEL_ARGS = dict(
    seq_len=SEQ_LEN,
    hidden_dim=768,
    intermediate_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    lbl_coef=None,
    rzl_coef=None,
)

BASE_PARAM_COUNT = int(MixtralConfig(**BASE_MODEL_ARGS, use_gmm=False).total_trainable_params(VOCAB_SIZE))

MODEL_VARIANTS = {
    "mixtral_300m": dict(
        run_name="pranshu_mixtral_ragged_300m_default",
        model_overrides={
            # keeps the original 12-layer, 12-head geometry
        },
        train=dict(
            train_batch_size=1536,
            num_train_steps=4000,
            learning_rate=5e-4,
            weight_decay=0.1,
            steps_per_eval=4000,
        ),
        description="Baseline ragged-dot Mixtral MoE replicating the original 300M run.",
    ),
    "mixtral_1b": dict(
        run_name="pranshu_mixtral_ragged_1b_expanded",
        model_overrides=dict(
            hidden_dim=768,
            intermediate_dim=3072,
            num_layers=16,
            num_heads=16,
            num_kv_heads=16,
        ),
        train=dict(
            train_batch_size=768,
            num_train_steps=32000,
            learning_rate=3e-4,
            weight_decay=0.1,
            steps_per_eval=4800,
        ),
        description="Intermediate ragged-dot Mixtral MoE targeting ~1B parameters with 16-layer, 16-head geometry.",
    ),
    "mixtral_1_5b": dict(
        run_name="pranshu_mixtral_ragged_1_5b_expanded",
        model_overrides=dict(
            hidden_dim=896,
            intermediate_dim=3584,
            num_layers=16,
            num_heads=16,
            num_kv_heads=16,
        ),
        train=dict(
            train_batch_size=640,
            num_train_steps=57600,
            learning_rate=2.5e-4,
            weight_decay=0.1,
            steps_per_eval=6000,
        ),
        description="Expanded ragged-dot Mixtral MoE targeting ~1.3-1.4B parameters with 16-layer, 16-head config.",
    ),
}

RESOURCE_CONFIG = TpuPodConfig(tpu_type="v5p-32", slice_count=1)


def build_config(name: str) -> tuple[str, SpeedrunConfig, int]:
    if name not in MODEL_VARIANTS:
        raise ValueError(f"unknown model variant: {name}")

    variant = MODEL_VARIANTS[name]
    model_kwargs = {**BASE_MODEL_ARGS, **variant.get("model_overrides", {}), "use_gmm": False}
    model_config = MixtralConfig(**model_kwargs, cross_entropy_block_size=32000)
    param_count = int(model_config.total_trainable_params(VOCAB_SIZE))

    train_kwargs = variant["train"]
    batch_size = train_kwargs["train_batch_size"]
    num_train_steps = train_kwargs["num_train_steps"]
    total_tokens = batch_size * SEQ_LEN * num_train_steps
    ratio = param_count / BASE_PARAM_COUNT

    train_config = SimpleTrainConfig(
        RESOURCE_CONFIG,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=train_kwargs["learning_rate"],
        weight_decay=train_kwargs["weight_decay"],
        steps_per_eval=train_kwargs["steps_per_eval"],
    )

    description = (
        f"{variant['description']} Approx. {param_count / 1e6:.1f}M parameters. "
        f"Global batch {batch_size:,} × {num_train_steps:,} steps (~{total_tokens / 1e9:.2f}B tokens, ratio {ratio:.2f}x base)."
    )
    speedrun_cfg = SpeedrunConfig(
        author=AUTHOR,
        description=description,
        model_config=model_config,
        train_config=train_config,
    )

    LOGGER.info(
        "Variant %s: params=%.1fM ratio=%.2f batch=%d total_tokens=%.2fB",
        name,
        param_count / 1e6,
        ratio,
        batch_size,
        total_tokens / 1e9,
    )

    return variant["run_name"], speedrun_cfg, param_count


if __name__ == "__main__":
    runs = [build_config(name) for name in MODEL_ORDER]

    steps = []
    for name, cfg, params in runs:
        LOGGER.info("Prepared %s with %.1fM parameters", name, params / 1e6)
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Ragged-dot Mixtral sweep (original config extensions)")
