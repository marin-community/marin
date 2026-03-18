# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect data transforms and attention masks for each ablation mode.

Loads raw records from the understanding and generation caches, applies the
transforms for each ablation mode, and prints a breakdown of loss weights
and segment IDs.

Usage:
    # Inspect all 6 ablation modes + baseline (uses default cache paths)
    uv run experiments/unified/inspect_ablation_modes.py

    # Inspect a specific mode
    uv run experiments/unified/inspect_ablation_modes.py --mode mask_und_visual

    # Custom cache paths
    uv run experiments/unified/inspect_ablation_modes.py \
        --und_cache gs://marin-vlm-eu/my_cache/understanding/train \
        --gen_cache gs://marin-vlm-eu/my_cache/generation/train

    # Show more rows
    uv run experiments/unified/inspect_ablation_modes.py --num_rows 5
"""

import argparse
from collections.abc import Callable

import numpy as np

from experiments.unified.vlm_tokenize_captions import (
    ENDOFTEXT_ID,
    VISION_END_ID,
    VISION_START_ID,
    VISUAL_TOKEN_OFFSET,
)

# Constants (duplicated from the experiment script to avoid import issues with
# the space-in-filename copy; these are stable across all unified_pretrain variants).
LLAMA3_VOCAB_SIZE = 128_256
UNIFIED_TOKENIZER_PATH = "gs://marin-vlm/tokenizers/llama3-unified-144k"
UNIFIED_CACHE_PATH = "gs://marin-vlm/hf_85m_levanter_cache_v2"


# --- Transforms (same as in the experiment script) ---


def _make_visual_weight_transform(new_w_visual: float) -> Callable[[np.ndarray], np.ndarray]:
    def transform(weights: np.ndarray) -> np.ndarray:
        mask = (weights > 0) & (weights < 1.0)
        if not mask.any():
            return weights
        result = weights.copy()
        result[mask] = new_w_visual
        return result

    return transform


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


def _make_modality_segment_ids(input_ids: np.ndarray) -> np.ndarray:
    is_visual = (input_ids >= LLAMA3_VOCAB_SIZE).astype(np.int32)
    is_sentinel = ((input_ids == VISION_START_ID) | (input_ids == VISION_END_ID)).astype(np.int32)
    return np.clip(is_visual + is_sentinel, 0, 1)


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


ALL_MODES = [
    "",
    "mask_und_visual",
    "mask_und_text",
    "mask_gen_visual",
    "mask_gen_text",
    "isolate_und_attn",
    "isolate_gen_attn",
    "und_text_only",
    "gen_visual_only",
]

MODE_DESCRIPTIONS = {
    "": "baseline (no ablation)",
    "mask_und_visual": "understanding: mask visual loss → text only",
    "mask_und_text": "understanding: mask text loss → visual only",
    "mask_gen_visual": "generation: mask visual loss → text only",
    "mask_gen_text": "generation: mask text loss → visual only",
    "isolate_und_attn": "understanding: isolate modality attention",
    "isolate_gen_attn": "generation: isolate modality attention",
    "und_text_only": "understanding: replace visual with EOS, text loss only",
    "gen_visual_only": "generation: replace text with EOS, visual loss only",
}


def _get_transforms(mode: str, w_visual: float):
    """Return (und_loss_transform, gen_loss_transform, und_seg_transform, gen_seg_transform, und_ids_transform, gen_ids_transform)."""
    w_visual_transform = _make_visual_weight_transform(w_visual)

    und_ablation = None
    gen_ablation = None
    und_seg = None
    gen_seg = None
    und_ids = None
    gen_ids = None

    if mode == "mask_und_visual":
        und_ablation = _zero_fractional_transform
    elif mode == "mask_und_text":
        und_ablation = _only_fractional_transform
    elif mode == "mask_gen_visual":
        gen_ablation = _zero_fractional_transform
    elif mode == "mask_gen_text":
        gen_ablation = _only_fractional_transform
    elif mode == "isolate_und_attn":
        und_seg = _make_modality_segment_ids
    elif mode == "isolate_gen_attn":
        gen_seg = _make_modality_segment_ids
    elif mode == "und_text_only":
        und_ablation = _zero_fractional_transform
        und_ids = _replace_visual_with_eos
    elif mode == "gen_visual_only":
        gen_ablation = _only_fractional_transform
        gen_ids = _replace_text_with_eos

    und_transform = _compose_transforms(und_ablation, w_visual_transform)
    gen_transform = _compose_transforms(_swap_primary_secondary_transform, gen_ablation, w_visual_transform)

    return und_transform, gen_transform, und_seg, gen_seg, und_ids, gen_ids


def _classify_tokens(ids: np.ndarray):
    """Return boolean masks for visual, text, and special tokens."""
    is_visual = ids >= VISUAL_TOKEN_OFFSET
    is_special = np.isin(ids, [VISION_START_ID, VISION_END_ID, ENDOFTEXT_ID])
    is_text = ~is_visual & ~is_special
    return is_visual, is_text, is_special


def _detect_ordering(ids: np.ndarray) -> str:
    if len(ids) == 0:
        return "empty"
    if ids[0] == VISION_START_ID:
        return "understanding (image-first)"
    return "generation (text-first)"


def print_record_detail(
    label: str,
    ids: np.ndarray,
    raw_weights: np.ndarray,
    transformed_weights: np.ndarray,
    segment_ids: np.ndarray | None,
    tok,
):
    """Print detailed breakdown of a single record after transform."""
    is_visual, is_text, is_special = _classify_tokens(ids)
    ordering = _detect_ordering(ids)

    n_total = len(ids)
    n_visual = int(is_visual.sum())
    n_text = int(is_text.sum())

    # Decode text portion
    non_visual_ids = [int(t) for t in ids if t < VISUAL_TOKEN_OFFSET]
    decoded = tok.decode(non_visual_ids) if non_visual_ids else ""
    if len(decoded) > 120:
        decoded = decoded[:120] + "..."

    print(f"\n  [{label}] {ordering}, {n_total} tokens (visual={n_visual}, text={n_text})")
    print(f"    Decoded: {decoded}")

    # Raw weights breakdown
    raw_vis_w = raw_weights[is_visual]
    raw_txt_w = raw_weights[is_text]
    print(f"    Raw weights:         visual={_weight_summary(raw_vis_w)}, text={_weight_summary(raw_txt_w)}")

    # Transformed weights breakdown
    tfm_vis_w = transformed_weights[is_visual]
    tfm_txt_w = transformed_weights[is_text]
    print(f"    Transformed weights: visual={_weight_summary(tfm_vis_w)}, text={_weight_summary(tfm_txt_w)}")

    # Loss token counts
    vis_loss = int((tfm_vis_w > 0).sum())
    txt_loss = int((tfm_txt_w > 0).sum())
    total_loss = int((transformed_weights > 0).sum())
    print(
        f"    Loss-active tokens:  visual={vis_loss}/{n_visual}, text={txt_loss}/{n_text}, total={total_loss}/{n_total}"
    )

    # Segment IDs for isolation modes
    if segment_ids is not None:
        vis_segs = np.unique(segment_ids[is_visual])
        txt_segs = np.unique(segment_ids[is_text])
        print(f"    Segment IDs:         visual={vis_segs.tolist()}, text={txt_segs.tolist()}")
        overlap = set(vis_segs.tolist()) & set(txt_segs.tolist())
        if overlap:
            print(f"    ⚠ WARNING: visual and text share segment IDs {overlap} — they CAN attend to each other!")
        else:
            print("    ✓ Visual and text in different segments — cross-modality attention blocked")

        # Print a compact attention matrix snippet
        _print_attention_snippet(ids, segment_ids)


def _weight_summary(w: np.ndarray) -> str:
    if len(w) == 0:
        return "n/a"
    nonzero = w[w > 0]
    if len(nonzero) == 0:
        return "all=0.0"
    unique = np.unique(nonzero)
    if len(unique) == 1:
        return f"all={unique[0]:.3f}"
    return f"min={nonzero.min():.3f}, max={nonzero.max():.3f}, mean={nonzero.mean():.3f}"


def _print_attention_snippet(ids: np.ndarray, segment_ids: np.ndarray, window: int = 40):
    """Print a compact view of which tokens can attend to which around the modality boundary."""
    # Find the first vision_start or vision_end token as the boundary
    boundary_positions = np.where((ids == VISION_START_ID) | (ids == VISION_END_ID))[0]
    if len(boundary_positions) == 0:
        return

    # Take a window around the first boundary
    boundary = boundary_positions[0]
    start = max(0, boundary - window // 4)
    end = min(len(ids), boundary + 3 * window // 4)
    snippet_ids = ids[start:end]
    snippet_segs = segment_ids[start:end]
    n = len(snippet_ids)

    # Build token type labels
    labels = []
    for i, tid in enumerate(snippet_ids):
        if tid == VISION_START_ID:
            labels.append("VS")
        elif tid == VISION_END_ID:
            labels.append("VE")
        elif tid == ENDOFTEXT_ID:
            labels.append("EO")
        elif tid >= VISUAL_TOKEN_OFFSET:
            labels.append("v")
        else:
            labels.append("t")

    # Print compact header
    print(f"    Attention snippet (positions {start}..{end-1}, seg_ids shown):")
    seg_line = "    segs: " + " ".join(f"{s:1d}" for s in snippet_segs[:n])
    type_line = "    type: " + " ".join(f"{l:>1s}" for l in labels[:n])
    print(type_line)
    print(seg_line)

    # Show causal attention matrix (small, every 4th position)
    step = max(1, n // 20)
    sampled = list(range(0, n, step))
    if sampled[-1] != n - 1:
        sampled.append(n - 1)

    print(f"    Causal attention (sampled every {step} tokens, . = can attend, x = blocked):")
    header = "          " + "".join(f"{labels[j]:>2s}" for j in sampled)
    print(header)
    for i in sampled:
        row_label = f"    {labels[i]:>2s} {start + i:3d}|"
        chars = []
        for j in sampled:
            if j > i:
                chars.append("  ")  # future token (causal mask)
            elif snippet_segs[i] == snippet_segs[j]:
                chars.append(" .")
            else:
                chars.append(" x")
        print(row_label + "".join(chars))


def inspect_mode(mode: str, und_records: list, gen_records: list, tok, w_visual: float):
    """Inspect a single ablation mode on the given records."""
    und_transform, gen_transform, und_seg, gen_seg, und_ids, gen_ids = _get_transforms(mode, w_visual)

    print(f"\n{'=' * 70}")
    print(f"Mode: {mode or '(baseline)'} — {MODE_DESCRIPTIONS[mode]}")
    print(f"{'=' * 70}")

    # Understanding records
    print(f"\n  --- Understanding cache ({len(und_records)} records) ---")
    for i, record in enumerate(und_records):
        ids = record["input_ids"]
        if und_ids is not None:
            ids = und_ids(ids)
        raw_w = record["loss_weights"]
        tfm_w = und_transform(raw_w.copy()) if und_transform else raw_w.copy()
        seg = und_seg(ids) if und_seg else None
        print_record_detail(f"und[{i}]", ids, raw_w, tfm_w, seg, tok)

    # Generation records
    print(f"\n  --- Generation cache ({len(gen_records)} records) ---")
    for i, record in enumerate(gen_records):
        ids = record["input_ids"]
        if gen_ids is not None:
            ids = gen_ids(ids)
        raw_w = record["loss_weights"]
        tfm_w = gen_transform(raw_w.copy()) if gen_transform else raw_w.copy()
        seg = gen_seg(ids) if gen_seg else None
        print_record_detail(f"gen[{i}]", ids, raw_w, tfm_w, seg, tok)


def main():
    parser = argparse.ArgumentParser(description="Inspect ablation mode transforms on cached data.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Specific ablation mode to inspect (default: all modes)",
    )
    parser.add_argument(
        "--und_cache",
        default=f"{UNIFIED_CACHE_PATH}/understanding/train",
        help="Understanding cache path",
    )
    parser.add_argument(
        "--gen_cache",
        default=f"{UNIFIED_CACHE_PATH}/generation/train",
        help="Generation cache path",
    )
    parser.add_argument("--tokenizer", default=UNIFIED_TOKENIZER_PATH)
    parser.add_argument("--num_rows", type=int, default=2, help="Number of rows to inspect per cache")
    parser.add_argument("--w_visual", type=float, default=1.0, help="w_visual value for transform")
    parser.add_argument("--start_row", type=int, default=0, help="Starting row index")
    args = parser.parse_args()

    from levanter.compat.hf_checkpoints import load_tokenizer
    from levanter.store.cache import TreeCache

    # Load caches
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "loss_weights": np.zeros((0,), dtype=np.float32)}

    print(f"Loading understanding cache: {args.und_cache}")
    und_cache = TreeCache.load(args.und_cache, exemplar=exemplar)
    print(f"  → {len(und_cache)} records")

    print(f"Loading generation cache: {args.gen_cache}")
    gen_cache = TreeCache.load(args.gen_cache, exemplar=exemplar)
    print(f"  → {len(gen_cache)} records")

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = load_tokenizer(args.tokenizer)

    # Load sample records
    und_records = [und_cache[args.start_row + i] for i in range(args.num_rows)]
    gen_records = [gen_cache[args.start_row + i] for i in range(args.num_rows)]

    # Inspect modes
    modes = [args.mode] if args.mode is not None else ALL_MODES
    for mode in modes:
        inspect_mode(mode, und_records, gen_records, tok, args.w_visual)

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
