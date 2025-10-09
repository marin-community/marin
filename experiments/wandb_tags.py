from __future__ import annotations

from collections.abc import Iterable

from experiments.anneal_config import AnnealConfig
from levanter.data.text import LMMixtureDatasetConfig


def _format_weight_tags(weights: dict[str, float]) -> list[str]:
    total = sum(weights.values()) or 1.0
    # normalize and sort by name for tag stability
    norm = {k: v / total for k, v in weights.items() if v > 0}
    parts = []
    for name in sorted(norm):
        pct = norm[name] * 100
        # compact percent without trailing zeros, cap small to 2 decimals
        if pct.is_integer():
            pct_str = f"{int(pct)}"
        else:
            pct_str = f"{pct:.2f}".rstrip("0").rstrip(".")
        short = name.split("/")[-1]
        parts.append(f"{short}-{pct_str}")
    return parts


def _first_stage_weights(weights: LMMixtureDatasetConfig) -> dict[str, float]:
    if isinstance(weights, dict):
        return weights
    # list[(step, dict)] → pick first stage
    return weights[0][1] if weights else {}


def _maybe_model_tag(model_config) -> list[str]:
    # Try to derive a simple model size tag like "llama-1b" or "d=4096-L=32"
    tags: list[str] = []
    try:
        hidden = getattr(model_config, "hidden_dim", None)
        layers = getattr(model_config, "num_layers", None)
        if hidden and layers:
            tags.append(f"d={hidden}")
            tags.append(f"L={layers}")
    except Exception:
        pass
    return tags


def _maybe_tokens_tag(num_tokens: int | float | None) -> list[str]:
    if not num_tokens:
        return []
    v = float(num_tokens)
    if v >= 1e12:
        return [f"{v/1e12:.1f}t".rstrip("0").rstrip(".")]
    if v >= 1e9:
        return [f"{v/1e9:.1f}b".rstrip("0").rstrip(".")]
    if v >= 1e6:
        return [f"{v/1e6:.1f}m".rstrip("0").rstrip(".")]
    return [str(int(v))]


def _maybe(tag: str, value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float):
        # concise float
        if value.is_integer():
            v = str(int(value))
        else:
            v = f"{value:.6g}"
    else:
        v = str(value)
    return [f"{tag}={v}"]


def _flatten(items: Iterable[Iterable[str]]) -> list[str]:
    out: list[str] = []
    for it in items:
        out.extend(it)
    return out


def wandb_tags_from_anneal(cfg: AnnealConfig) -> list[str]:
    """
    Build a consistent set of WandB tags from an AnnealConfig.

    Includes:
    - dataset mix percentages (short names)
    - token budget shorthand
    - key hyperparams (lr, wd, warmup, bsz, schedule)
    - model hints (hidden dim, layers)
    - any user-supplied tags (cfg.wandb_tags)
    """

    data: LMMixtureDatasetConfig = cfg.dataset_config

    weights = _first_stage_weights(data.train_weights)
    mix_tags = _format_weight_tags(weights)

    token_tags = _maybe_tokens_tag(cfg.num_anneal_training_tokens)

    hyper_tags = _flatten(
        [
            _maybe("lr", cfg.learning_rate),
            _maybe("wd", cfg.weight_decay),
            _maybe("warmup", cfg.warmup),
            _maybe("bsz", cfg.train_batch_size),
            _maybe("sched", cfg.lr_schedule),
            _maybe("min_lr_ratio", cfg.min_lr_ratio),
        ]
    )

    model_tags = _maybe_model_tag(cfg.model_config)

    # user-provided tags win by being appended at the end
    return [*mix_tags, *token_tags, *hyper_tags, *model_tags, *cfg.wandb_tags]


__all__ = ["wandb_tags_from_anneal"]
