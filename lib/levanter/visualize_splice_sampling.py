#!/usr/bin/env python3
"""
Visualize sampling coverage of SpliceDocumentDataset on a local .txt file.

Features
- Builds SpliceDocumentDataset using settings from a provided YAML (defaults to the 20k config).
- Wraps the dataset logic with an epoch-permutation mapping (same mapping as EpochPermutationDataset).
- Draws a running histogram of where selected examples land in the original text (by character position).
- Saves frames over time and encodes a short MP4 clip (or PNGs) showing how coverage evolves.

Usage
  uv run scripts/visualize_splice_sampling.py \
      --text my_doc.txt \
      --yaml submodules/levanter/config/memorize/splice_comma_600m_single_20k_central2.yaml \
      --out out/coverage.mp4 \
      --steps 5000 --bins 800 --seed 0

Notes
- This script decodes each token piece once to build a token->character span map. That keeps per-frame cost low.
- We mimic EpochPermutationDataset's index mapping to avoid relying on private internals while still matching its behavior.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import yaml
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

import jax
import jax.random as jrandom

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "submodules", "levanter", "src"))

from levanter.compat.hf_checkpoints import load_tokenizer  # type: ignore
from levanter.data._prp import Permutation
from levanter.shapes import Axis
import haliax as hax

from levanter.data.splice_dataset import SpliceDocumentDataset


@dataclass
class SpliceSettings:
    seq_len: int
    tokenizer_name: str
    content_length: Optional[int]
    content_stride: int
    offset_stride: int
    content_start_mode: str
    min_copy_len: int
    alpha: float


def parse_yaml_settings(path: str) -> SpliceSettings:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    return SpliceSettings(
        seq_len=int(model_cfg.get("seq_len", 4096)),
        tokenizer_name=str(data_cfg.get("tokenizer", "meta-llama/Meta-Llama-3.1-8B")),
        content_length=(None if data_cfg.get("content_length", None) is None else int(data_cfg["content_length"])),
        content_stride=int(data_cfg.get("content_stride", 1)),
        offset_stride=int(data_cfg.get("offset_stride", 1)),
        content_start_mode=str(data_cfg.get("content_start_mode", "coverage_balanced")),
        min_copy_len=int(data_cfg.get("min_copy_len", 2)),
        alpha=float(data_cfg.get("alpha", 0.0)),
    )


def token_char_spans(tokenizer, token_ids: List[int]):
    pieces = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
    starts = np.zeros(len(pieces), dtype=np.int64)
    ends = np.zeros(len(pieces), dtype=np.int64)
    pos = 0
    for i, s in enumerate(pieces):
        starts[i] = pos
        pos += len(s)
        ends[i] = pos
    return pieces, starts, ends, int(pos)


def build_splice_for_text(text_path: str, settings: SpliceSettings):
    tok = load_tokenizer(settings.tokenizer_name)
    enc = tok(open(text_path, "r", encoding="utf-8").read(), add_special_tokens=False)
    doc_ids = np.asarray(enc["input_ids"], dtype=np.int32)

    Pos = Axis("token", settings.seq_len)

    pad_id = tok.pad_token_id or tok.eos_token_id
    eos_id = tok.eos_token_id

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc_ids,
        pad_token_id=int(pad_id),
        eos_token_id=int(eos_id) if eos_id is not None else None,
        content_length=settings.content_length,
        content_stride=settings.content_stride,
        offset_stride=settings.offset_stride,
        content_start_mode=settings.content_start_mode,
        min_copy_len=settings.min_copy_len,
        alpha=settings.alpha,
    )

    return tok, ds


def epoch_permutation_index_map(length: int, i: int, key: jax.Array) -> int:
    epoch, pos = divmod(i, length)
    ek = jax.random.fold_in(key, int(epoch))
    perm = Permutation.make("feistel", length, ek)
    return int(perm(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Path to .txt file")
    ap.add_argument("--yaml", default="submodules/levanter/config/memorize/splice_comma_600m_single_20k_balanced_central2.yaml")
    ap.add_argument("--out", default="out/coverage.mp4")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bins", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10, help="Video frames per second")
    ap.add_argument("--frame_every", type=int, default=10, help="Render a frame every N steps")
    ap.add_argument("--max_frames", type=int, default=400, help="Maximum number of frames to render")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    settings = parse_yaml_settings(args.yaml)
    tokenizer, ds = build_splice_for_text(args.text, settings)

    # Precompute token->char spans for the source document (used to place hits)
    _, t_starts, t_ends, text_char_len = token_char_spans(tokenizer, list(map(int, ds.doc)))

    # Dataset length (#pairs in one epoch)
    ds_len = len(ds.as_sync_dataset())
    key = jrandom.PRNGKey(args.seed)

    # Histogram state (character domain, binned)
    bins = args.bins
    bin_edges = np.linspace(0, text_char_len, bins + 1, dtype=np.int64)
    hist_counts = np.zeros(bins, dtype=np.int64)

    # Video writer
    writer = imageio.get_writer(args.out, fps=args.fps)

    def render_frame(step: int, save_png: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist_counts, color="#1f77b4", linewidth=1.0)
        ax.set_xlim(0, text_char_len)
        ax.set_xlabel("Book position (character)")
        ax.set_ylabel("# samples")
        title = (
            f"Splice coverage — step {step}\n"
            f"S={settings.seq_len} ct_stride={settings.content_stride} off_stride={settings.offset_stride} "
            f"mode={settings.content_start_mode} min_copy={settings.min_copy_len} alpha={settings.alpha}"
        )
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        if save_png:
            # Save directly as PNG
            fig.savefig(save_png, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            # Render to video frame
            fig.canvas.draw()
            # Convert to image (cross-platform: use buffer_rgba)
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Extract RGB channels (drop alpha)
            frame = frame[:, :, :3]
            writer.append_data(frame)
            plt.close(fig)

    frames_emitted = 0
    for i in tqdm(range(args.steps), desc="Sampling steps", unit="step"):
        mapped_idx = epoch_permutation_index_map(ds_len, i, key)
        # Look up (t, s) directly from the dataset's internal pairs
        t, s = ds._pairs[mapped_idx]
        remaining_doc = ds.L - t
        K = ds.K if ds.K is not None else remaining_doc
        K = max(0, min(K, remaining_doc))
        copy_len = min(K, remaining_doc, ds.S - s)

        if copy_len >= ds.min_copy_len:
            start_char = int(t_starts[t])
            end_char_excl = int(t_ends[t + copy_len - 1])
            # Accumulate into bins efficiently
            # Determine bins overlapped by [start_char, end_char_excl)
            # Convert char-range to bin indices
            l = np.searchsorted(bin_edges, start_char, side="right") - 1
            r = np.searchsorted(bin_edges, end_char_excl - 1, side="right") - 1
            l = max(0, min(l, bins - 1))
            r = max(0, min(r, bins - 1))
            if r >= l:
                hist_counts[l : r + 1] += 1

        if (i % args.frame_every == 0) and (frames_emitted < args.max_frames):
            render_frame(i)
            frames_emitted += 1

    # Final frame
    if frames_emitted == 0:
        render_frame(args.steps)
        frames_emitted = 1
    writer.close()
    duration = frames_emitted / args.fps
    print(f"Wrote {args.out} ({frames_emitted} frames, {duration:.1f}s @ {args.fps} fps)")

    # Save final frame as PNG
    png_path = os.path.splitext(args.out)[0] + ".png"
    render_frame(args.steps, save_png=png_path)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
