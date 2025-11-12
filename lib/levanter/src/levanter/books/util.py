# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
levanter.books.util
===================
Utility helpers originally sketched in ``careless.py`` but generalized so
other Levanter scripts (e.g. ``eval_careless_lm.py``) can import them
without pulling in PyTorch-specific baggage.

Two groups of helpers live here:

1.  **Extraction-rate metrics** – `(n,p)` discoverability calculations that
    operate purely in NumPy.
2.  **Sliding-window text processing** – functions that replicate the
    logic of `chunk_text_to_sliding_window_token_chunks` from
    ``careless.py`` but yield structures convenient for Levanter/Haliax
    evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import haliax as hax

from levanter.models.lm_model import LmExample


__all__ = [
    "compute_extraction_rates",
    "compute_max_extraction_rates",
    "compute_n_for_pz_and_p",
    "compute_n_values_for_pz_and_p_list",
    "compute_greedy_extraction_rate",
    "chunk_text_to_sliding_window_token_chunks",
    "sliding_lm_examples",
    "chunk_token_ids_to_sliding_windows",
    "token_sliding_lm_examples",
    "create_pz_histogram_linear",
    "create_pz_histogram",
    "create_bar_plot",
]


# -----------------------------------------------------------------------------
# Core metric helpers ---------------------------------------------------------
# -----------------------------------------------------------------------------
def create_pz_histogram_linear(
    pz_list: List[float], threshold: float, save_path: str, book_title: str = "Book", model_name: str = "Unknown Model"
):
    """
    Create a histogram of p_z values matching the style from the reference codebase.
    Uses dynamic bin calculation and styling similar to make_pz_dist_plot.

    Args:
        pz_list: List of suffix probabilities
        threshold: Minimum p_z value to include (e.g., 0.0001 for 0.01%)
        save_path: Path to save the histogram
        book_title: Title for the plot
        model_name: Name of the model for the plot title
    """
    # Filter p_z values above threshold
    filtered_pz = [pz for pz in pz_list if pz >= threshold]

    print(f"Total sequences: {len(pz_list)}")
    print(f"Sequences above {threshold*100:.3f}% threshold: {len(filtered_pz)}")

    if len(filtered_pz) == 0:
        print(f"No sequences above threshold {threshold}. Skipping histogram.")
        return

    # Convert to percentages for plotting
    filtered_pz_percent = [pz * 100 for pz in filtered_pz]

    # Dynamic bin calculation like in the reference codebase
    n = len(filtered_pz_percent)
    num_bins = min(50, max(5, int(np.sqrt(n))))

    # Create histogram with styling matching the reference
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use matplotlib hist with dynamic bins (linear spacing)
    bins = np.linspace(threshold * 100, 100, num_bins)
    counts, bin_edges, patches = ax.hist(
        filtered_pz_percent, bins=bins, color="steelblue", edgecolor="black", linewidth=0.3, alpha=0.8
    )

    # Set log scale for y-axis (no fixed limits - let it auto-scale like reference)
    ax.set_yscale("log")
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlabel("Probability of extraction ($p_z$)", fontsize=12)  # Use LaTeX formatting

    # Set x-axis limits and ticks for better spacing
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # Set title to match the reference style
    threshold_percent = threshold * 100
    ax.set_title(
        f"{book_title}:\nDistribution of $p_z$ (≥ {threshold_percent:.2f}%) for {model_name}",
        fontsize=14,
        fontweight="bold",
    )

    # Remove grid for cleaner look like reference
    ax.grid(False)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to: {save_path}")

    # Return same statistics as original function
    stats = {
        "total_sequences": len(pz_list),
        "above_threshold": len(filtered_pz),
        "mean_pz": np.mean(filtered_pz),
        "median_pz": np.median(filtered_pz),
        "max_pz": max(filtered_pz),
        "threshold_used": threshold,
    }

    return stats


# Add this function after the main function or before it
def create_pz_histogram(pz_list: List[float], threshold: float, save_path: str, book_title: str = "Book"):
    """
    Create a histogram of p_z values with a threshold filter.

    Args:
        pz_list: List of suffix probabilities
        threshold: Minimum p_z value to include (e.g., 0.0001 for 0.01%)
        save_path: Path to save the histogram
        book_title: Title for the plot
    """
    # Filter p_z values above threshold
    filtered_pz = [pz for pz in pz_list if pz >= threshold]

    print(f"Total sequences: {len(pz_list)}")
    print(f"Sequences above {threshold*100:.3f}% threshold: {len(filtered_pz)}")

    if len(filtered_pz) == 0:
        print(f"No sequences above threshold {threshold}. Skipping histogram.")
        return

    # Convert to percentages for plotting
    filtered_pz_percent = [pz * 100 for pz in filtered_pz]

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use log scale for y-axis to match the reference plot
    bins = np.logspace(np.log10(threshold * 100), np.log10(max(filtered_pz_percent)), 50)
    counts, bin_edges, patches = ax.hist(
        filtered_pz_percent, bins=bins, alpha=0.8, color="steelblue", edgecolor="black", linewidth=0.5
    )

    # Set log scale for y-axis
    ax.set_yscale("log")
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlabel("Probability of extraction (p_z)", fontsize=12)

    # Format x-axis as percentages
    ax.set_xticklabels([f"{x:.1f}%" for x in ax.get_xticks()])

    # Set title
    threshold_percent = threshold * 100
    ax.set_title(f"{book_title}: Distribution of p_z (≥ {threshold_percent:.3f}%)", fontsize=14, fontweight="bold")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to: {save_path}")

    # Return some statistics
    stats = {
        "total_sequences": len(pz_list),
        "above_threshold": len(filtered_pz),
        "mean_pz": np.mean(filtered_pz),
        "median_pz": np.median(filtered_pz),
        "max_pz": max(filtered_pz),
        "threshold_used": threshold,
    }

    return stats


def create_bar_plot(
    values: Sequence[float] | np.ndarray,
    save_path: str,
    *,
    title: str = "",
    xlabel: str = "Book position (character)",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """Render a 1xN "bar" plot (as an image strip) of values in [0,1].

    This mirrors the visualization used in eval_sliding_total where we display a
    thin imshow of per-position max probabilities, with a colorbar indicating
    magnitude. Saves a PNG to ``save_path``.

    Args:
        values: 1-D array-like of numeric values (e.g., per-char max P(z)).
        save_path: Where to write the PNG.
        title: Plot title.
        xlabel: X-axis label.
        vmin: Minimum value for colormap normalization.
        vmax: Maximum value for colormap normalization.
    """
    arr = np.asarray(values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(14, 2))
    im = ax.imshow(
        arr[np.newaxis, :],
        cmap="Blues",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Max. probability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_extraction_rates(
    pz_list: Sequence[float] | np.ndarray,
    n_list: Sequence[int] | np.ndarray | None = None,
    p_list: Sequence[float] | np.ndarray | None = None,
):
    """Return matrix of (n, p)-discoverable extraction rates.

    Args
    ----
    pz_list: 1-D iterable of suffix probabilities for each example.
    n_list:  Iterable of *n* values (number of draws).  Defaults to
             ``range(1, 10_001, 10)``.
    p_list:  Iterable of target probabilities *p* we want the extraction
             probability to exceed.  Defaults to ``[0.1, 0.5, 0.9, 0.99,
             0.999]``.

    Returns
    -------
    List[List[tuple]]
        ``outer list`` iterates over *p*; ``inner list`` is ``[(rate, p,
        n), ...]`` for every *n*.
    """

    if n_list is None:
        n_list = list(range(1, 10_001, 10))
    if p_list is None:
        p_list = [0.1, 0.5, 0.9, 0.99, 0.999]

    pz = np.asarray(pz_list, dtype=np.float64)
    # NumPy cannot cast Python ints larger than int64_max to int64; values like 2**69
    # (generated by the default `n_list = [2**x for x in range(70)]`) will overflow and
    # raise `OverflowError`.  We clip anything above the maximum representable int64 so
    # the metric computation still succeeds instead of crashing after a long run.
    int64_max = np.iinfo(np.int64).max
    # Clip each candidate n to int64_max to avoid OverflowError during np.asarray cast.
    n_sanitized = [min(int64_max, int(x)) for x in n_list]
    n = np.asarray(n_sanitized, dtype=np.int64)

    # Broadcast: shape (examples, n)
    prob_matrix = 1 - np.power(1 - pz[:, None], n[None, :])

    results: list[list[tuple[float, float, int]]] = []
    for p in tqdm(p_list, desc="extract-rate p loop"):
        meets = prob_matrix >= p
        rates = meets.mean(axis=0)  # proportion of examples where P≥p
        results.append([(float(rate), float(p), int(nn)) for rate, nn in zip(rates, n)])
    return results


def compute_max_extraction_rates(
    pz_list: Sequence[float] | np.ndarray,
    n_list: Sequence[int] | np.ndarray | None = None,
    p_list: Sequence[float] | np.ndarray | None = None,
):
    """Wrapper with exponentially-spaced *n* to find global max rates."""
    if n_list is None:
        n_list = [2**x for x in range(70)]
    return compute_extraction_rates(pz_list, n_list=n_list, p_list=p_list)


# -----------------------------------------------------------------------------
# Analytical helpers ----------------------------------------------------------
# -----------------------------------------------------------------------------


def compute_n_for_pz_and_p(pz_list: Sequence[float], p: float):
    """For each pᶻ in *pz_list* find *n* s.t. 1 – (1 – pᶻ)ⁿ ≥ *p*.

    Returns an ``np.ndarray`` of integers (ceil).  If the input array has
    only one finite entry, a scalar int is returned for convenience.
    """
    pz = np.asarray(pz_list, dtype=np.float64)
    valid = (pz > 0) & (pz < 1)
    pz = pz[valid]
    log1m_p = np.log(1 - p)
    log1m_pz = np.log(1 - pz)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_vals = np.ceil(log1m_p / log1m_pz)
    n_vals = n_vals[np.isfinite(n_vals)].astype(int)
    if n_vals.size == 1:
        return int(n_vals[0])
    return n_vals


def compute_n_values_for_pz_and_p_list(pz: float, p_list: Sequence[float] | None = None):
    """Return arrays (p_array, n_array) for a single pᶻ over many p thresholds."""
    if p_list is None:
        p_list = np.arange(0.01, 1.0, 0.01)
    p_arr = np.asarray(p_list, dtype=np.float64)
    log1m_p = np.log(1 - p_arr)
    log1m_pz = np.log(1 - pz)
    n_vals = np.ceil(log1m_p / log1m_pz).astype(int)

    valid = np.isfinite(n_vals)
    return p_arr[valid], n_vals[valid]


def compute_greedy_extraction_rate(pz_list: Sequence[float]):
    """Special-case extraction rate when decoding is deterministic greedy."""
    return float(np.sum(pz_list)) / len(pz_list)


# -----------------------------------------------------------------------------
# Sliding-window helpers (adapted from careless.py) ----------------------------
# -----------------------------------------------------------------------------


def chunk_text_to_sliding_window_token_chunks(
    text: str,
    tokenizer,
    *,
    chunk_size: int = 100,
    slice_length: int = 2000,
    cursor_inc: int = 10,
) -> list[Dict[str, Any]]:
    """Tokenise *text* into overlapping `chunk_size`-token windows.

    Replicates the logic in ``careless.py`` almost verbatim but drops the
    torch-specific bits.  Returns a list of dictionaries with keys:

    ``input_ids``          – *list[int]* of length ``chunk_size``
    ``start_idx``          – start character index in *text*
    ``end_idx``            – end character index (inclusive) in *text*
    ``text``               – decoded chunk text (useful for debugging)
    ``attention_mask``     – list[int] same length as ``input_ids``
    ``text_len``           – length of decoded text in characters
    """

    all_chunks: list[Dict[str, Any]] = []
    text_cursor = 0
    text_len = len(text)

    progress_markers = {i for i in range(10, 101, 10)}

    while text_cursor < text_len:
        start_idx = text_cursor
        end_idx_plus_one = min(text_cursor + slice_length, text_len)
        text_slice = text[start_idx:end_idx_plus_one]

        enc = tokenizer(text_slice, add_special_tokens=False, return_attention_mask=True)
        input_ids: list[int] = enc["input_ids"][:chunk_size]
        attention_mask: list[int] = enc.get("attention_mask", [1] * len(input_ids))[:chunk_size]

        if len(input_ids) == chunk_size:
            decoded_chunk = tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_len = len(decoded_chunk)
            all_chunks.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_idx": start_idx,
                    "end_idx": start_idx + decoded_len - 1,
                    "text_len": decoded_len,
                    "text": decoded_chunk,
                }
            )

        text_cursor += cursor_inc
        pct_complete = int(100 * text_cursor / text_len)
        if pct_complete in progress_markers:
            logging.getLogger(__name__).info("Sliding-window progress: %s%%", pct_complete)
            progress_markers.remove(pct_complete)

    return all_chunks


def sliding_lm_examples(
    text: str,
    tokenizer,
    Pos: hax.Axis,
    pad_id: int,
    *,
    chunk_size: int = 100,
    slice_length: int = 2000,
    cursor_inc: int = 10,
) -> Generator[Tuple[LmExample, Tuple[int, int]], None, None]:
    """Yield ``(LmExample, (char_start, char_end))`` pairs for training/eval.

    The first *half* (50 tokens when ``chunk_size=100``) serve as the
    *prompt*, the second half as the *suffix* whose probability will be
    evaluated.
    """
    half = chunk_size // 2
    for chunk in chunk_text_to_sliding_window_token_chunks(
        text,
        tokenizer,
        chunk_size=chunk_size,
        slice_length=slice_length,
        cursor_inc=cursor_inc,
    ):
        ids = chunk["input_ids"]
        if len(ids) < Pos.size:
            ids = ids + [pad_id] * (Pos.size - len(ids))
        tokens_named = hax.named(np.array(ids, dtype=np.int32), Pos)
        example = LmExample.from_prompt_and_completion(Pos, tokens_named, prompt_length=half)
        yield example, (chunk["start_idx"], chunk["end_idx"])


def chunk_token_ids_to_sliding_windows(
    token_ids: Sequence[int],
    tokenizer,
    *,
    chunk_size: int = 100,
    cursor_inc: int = 1,
) -> list[Dict[str, Any]]:
    """Create overlapping ``chunk_size`` token windows from pre-tokenised ids."""

    all_chunks: list[Dict[str, Any]] = []
    token_cursor = 0
    token_len = len(token_ids)

    progress_markers = {i for i in range(10, 101, 10)}

    while token_cursor < token_len:
        ids = list(token_ids[token_cursor : token_cursor + chunk_size])
        if len(ids) < chunk_size:
            break

        decoded_chunk = tokenizer.decode(ids, skip_special_tokens=True)
        all_chunks.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "start_token": token_cursor,
                "end_token": token_cursor + chunk_size - 1,
                "text_len": len(decoded_chunk),
                "text": decoded_chunk,
            }
        )

        token_cursor += cursor_inc
        pct_complete = int(100 * token_cursor / token_len)
        if pct_complete in progress_markers:
            logging.getLogger(__name__).info("Sliding-window progress: %s%%", pct_complete)
            progress_markers.remove(pct_complete)

    return all_chunks


def token_sliding_lm_examples(
    token_ids: Sequence[int],
    tokenizer,
    Pos: hax.Axis,
    pad_id: int,
    *,
    chunk_size: int = 100,
    cursor_inc: int = 1,
    prompt_tokens: int = 50,
) -> Generator[Tuple[LmExample, Tuple[int, int]], None, None]:
    """Yield ``(LmExample, (token_start, token_end))`` pairs from token ids."""

    for chunk in chunk_token_ids_to_sliding_windows(
        token_ids,
        tokenizer,
        chunk_size=chunk_size,
        cursor_inc=cursor_inc,
    ):
        ids = chunk["input_ids"]
        if len(ids) < Pos.size:
            ids = ids + [pad_id] * (Pos.size - len(ids))
        tokens_named = hax.named(np.array(ids, dtype=np.int32), Pos)
        example = LmExample.from_prompt_and_completion(
            Pos, tokens_named, prompt_length=prompt_tokens, ignore_id=pad_id
        )
        yield example, (chunk["start_token"], chunk["end_token"])
