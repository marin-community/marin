#!/usr/bin/env python3
"""
Visualize per-document position coverage over time for MultiSpliceDocumentDataset.

- Builds a MultiSpliceDocumentDataset from a list of local text files (provided via --texts or YAML).
- Simulates training order using the same epoch permutation mapping as EpochPermutationDataset.
- For each document, accumulates a histogram over character positions of the copied spans and renders 5 panels
  (one per doc), where the x-axis is that document's character space.

Usage
  uv run scripts/visualize_multi_splice_sampling.py \
      --texts samp_doc/wiki.txt samp_doc/wiki.txt samp_doc/wiki.txt samp_doc/wiki.txt samp_doc/wiki.txt \
      --yaml config/memorize/splice_multi_5docs_local.yaml \
      --out balance_coverage.mp4 --steps 10000 --bins 800 --seed 0

Notes
- This script focuses on document-level balance (which doc gets sampled) rather than token-position coverage.
  For within-document position coverage, see scripts/visualize_splice_sampling.py.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

from levanter.data.splice_dataset import MultiSpliceDocumentDataset


@dataclass
class MultiSpliceSettings:
    seq_len: int
    tokenizer_name: str
    content_length: Optional[int]
    content_stride: int
    content_start_mode: str
    min_copy_len: int
    alpha: float
    adaptive_k: bool
    offset_jitter: int
    balance_mode: str
    balance_tau: float
    epoch_length: Optional[int]
    bins: int = 800


def parse_yaml_settings(path: Optional[str]) -> Optional[MultiSpliceSettings]:
    if path is None:
        return None
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # Accept both keys used in docs (balance_mode by_temperature) and defaults
    return MultiSpliceSettings(
        seq_len=int(model_cfg.get("seq_len", 4096)),
        tokenizer_name=str(data_cfg.get("tokenizer", "meta-llama/Meta-Llama-3.1-8B")),
        content_length=(None if data_cfg.get("content_length", None) is None else int(data_cfg["content_length"])),
        content_stride=int(data_cfg.get("content_stride", 1)),
        content_start_mode=str(data_cfg.get("content_start_mode", "coverage_balanced")),
        min_copy_len=int(data_cfg.get("min_copy_len", 2)),
        alpha=float(data_cfg.get("alpha", 0.0)),
        adaptive_k=bool(data_cfg.get("adaptive_k", True)),
        offset_jitter=int(data_cfg.get("offset_jitter", 0)),
        balance_mode=str(data_cfg.get("balance_mode", "by_temperature")),
        balance_tau=float(data_cfg.get("balance_tau", 1.0)),
        epoch_length=(None if data_cfg.get("epoch_length", None) is None else int(data_cfg["epoch_length"])),
        bins=int(data_cfg.get("bins", 800)),
    )


def epoch_permutation_index_map(length: int, i: int, key: jax.Array) -> int:
    epoch, pos = divmod(i, length)
    ek = jax.random.fold_in(key, int(epoch))
    perm = Permutation.make("feistel", length, ek)
    return int(perm(pos))


def tokenize_texts(tokenizer, paths: List[str]) -> List[np.ndarray]:
    docs: List[np.ndarray] = []
    for p in paths:
        text = open(p, "r", encoding="utf-8").read()
        enc = tokenizer(text, add_special_tokens=False)
        ids = np.asarray(enc["input_ids"], dtype=np.int32)
        docs.append(ids)
    return docs


def token_char_spans(tokenizer, token_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    pieces = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
    starts = np.zeros(len(pieces), dtype=np.int64)
    ends = np.zeros(len(pieces), dtype=np.int64)
    pos = 0
    for i, s in enumerate(pieces):
        starts[i] = pos
        pos += len(s)
        ends[i] = pos
    return starts, ends, int(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", nargs="*", default=[], help="List of .txt paths for 5 documents")
    ap.add_argument("--yaml", default=None, help="Optional YAML for splice settings (not reading caches)")
    ap.add_argument("--out", default="out/balance_coverage.mp4")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=None, help="Histogram bins per document (overrides YAML)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--frame_every", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=400)
    args = ap.parse_args()

    if len(args.texts) == 0:
        raise SystemExit("Please provide --texts with 5 local .txt files (can reuse the same file for a quick demo).")
    if len(args.texts) != 5:
        print(f"[warn] You provided {len(args.texts)} texts; this script will plot 5 lines. Using the first 5.")
        args.texts = args.texts[:5]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    settings = parse_yaml_settings(args.yaml)
    if settings is None:
        # Reasonable defaults if YAML not provided
        settings = MultiSpliceSettings(
            seq_len=4096,
            tokenizer_name="meta-llama/Meta-Llama-3.1-8B",
            content_length=4096,
            content_stride=1,
            content_start_mode="coverage_balanced",
            min_copy_len=2,
            alpha=0.0,
            adaptive_k=True,
            offset_jitter=0,
            balance_mode="by_temperature",
            balance_tau=0.7,
            epoch_length=None,
        )

    tokenizer = load_tokenizer(settings.tokenizer_name)
    docs = tokenize_texts(tokenizer, args.texts)

    Pos = Axis("token", settings.seq_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    ds = MultiSpliceDocumentDataset(
        Pos=Pos,
        doc_tokens_list=docs,
        pad_token_id=int(pad_id),
        eos_token_id=int(eos_id) if eos_id is not None else None,
        content_length=settings.content_length,
        content_stride=settings.content_stride,
        content_start_mode=settings.content_start_mode,
        min_copy_len=settings.min_copy_len,
        alpha=settings.alpha,
        adaptive_k=settings.adaptive_k,
        offset_jitter=settings.offset_jitter,
        balance_mode=settings.balance_mode,
        balance_tau=settings.balance_tau,
        epoch_length=settings.epoch_length,
    )

    ds_len = len(ds.as_sync_dataset())
    key = jrandom.PRNGKey(args.seed)

    # Per-document token->char spans and histograms
    num_docs = 5
    bins = args.bins if args.bins is not None else settings.bins
    starts_list: List[np.ndarray] = []
    ends_list: List[np.ndarray] = []
    char_lens: List[int] = []
    bin_edges_list: List[np.ndarray] = []
    hist_counts = np.zeros((num_docs, bins), dtype=np.int64)

    for d in range(num_docs):
        starts, ends, total_chars = token_char_spans(tokenizer, list(map(int, docs[d])))
        starts_list.append(starts)
        ends_list.append(ends)
        char_lens.append(total_chars)
        bin_edges_list.append(np.linspace(0, total_chars, bins + 1, dtype=np.int64))

    # Video writer
    writer = imageio.get_writer(args.out, fps=args.fps)

    def render_frame(step: int, save_png: Optional[str] = None):
        fig, axes = plt.subplots(num_docs, 1, figsize=(12, 1.8 * num_docs), sharex=False)
        if num_docs == 1:
            axes = [axes]
        for d in range(num_docs):
            ax = axes[d]
            edges = bin_edges_list[d]
            ax.plot(0.5 * (edges[:-1] + edges[1:]), hist_counts[d], linewidth=1.0)
            ax.set_xlim(0, char_lens[d])
            ax.set_ylabel(f"doc{d}")
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Character position in document")
        fig.suptitle(
            f"Multi-splice coverage over position (tau={settings.balance_tau}, mode={settings.balance_mode}, jitter={settings.offset_jitter})\n"
            f"S={settings.seq_len} ct_stride={settings.content_stride} mode={settings.content_start_mode}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        if save_png:
            fig.savefig(save_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]
            writer.append_data(frame)
            plt.close(fig)

    frames_emitted = 0
    for i in tqdm(range(args.steps), desc="Sampling steps", unit="step"):
        mapped_idx = epoch_permutation_index_map(ds_len, i, key)
        doc_idx, t = ds._pairs[mapped_idx]
        if 0 <= doc_idx < num_docs:
            # Determine copy length as in dataset
            K_i = ds._Ks[doc_idx]
            L_i = int(ds.docs[doc_idx].shape[0])
            s_base = max(0, ds.S - int(K_i))
            if ds.offset_jitter > 0 and s_base > 0:
                j = mapped_idx % (ds.offset_jitter + 1)
                j = min(j, s_base)
                s = s_base - j
            else:
                s = s_base
            remaining = L_i - int(t)
            copy_len = min(int(K_i), remaining, ds.S - int(s))
            if copy_len >= ds.min_copy_len:
                starts = starts_list[doc_idx]
                ends = ends_list[doc_idx]
                edges = bin_edges_list[doc_idx]
                l = np.searchsorted(edges, int(starts[t]), side="right") - 1
                r = np.searchsorted(edges, int(ends[t + copy_len - 1]) - 1, side="right") - 1
                l = max(0, min(l, bins - 1))
                r = max(0, min(r, bins - 1))
                if r >= l:
                    hist_counts[doc_idx, l : r + 1] += 1

        if (i % args.frame_every == 0) and (frames_emitted < args.max_frames):
            render_frame(i)
            frames_emitted += 1

    if frames_emitted == 0:
        render_frame(args.steps - 1)
        frames_emitted = 1
    writer.close()

    png_path = os.path.splitext(args.out)[0] + ".png"
    render_frame(args.steps - 1, save_png=png_path)
    print(f"Wrote {args.out} and {png_path}")


if __name__ == "__main__":
    main()
