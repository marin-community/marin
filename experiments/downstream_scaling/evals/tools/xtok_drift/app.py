# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradio browser for scored xtok advisor-drift rollouts.

Reads the ``.npz``/``.json`` cache written by ``scoring`` and renders, per
rollout: the completion as a chunk strip colored by KL(canonical || forced) at
each boundary, a sortable boundary table, and a per-boundary detail pane with
the two top-k distributions. An ad-hoc probe tab re-scores an editable chunked
prefix (``|`` marks chunk boundaries) against the resident model — the only
feature that needs a GPU; browsing works cache-only.

    python -m experiments.downstream_scaling.evals.tools.xtok_drift.app \\
        --cache-dir ~/xtok_drift_cache --port 7860
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import os
from typing import Any

import gradio as gr
import numpy as np

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.tools.xtok_drift.scoring import (
    ADVISOR_MODEL,
    ADVISOR_REVISION,
    Comparison,
    load_advisor,
    score_chunked_prefix,
    token_label,
)

# Sequential blue ramp (dataviz palette steps 100..700); lightest = near-zero
# KL and recedes toward the surface. Ink flips to white on the darker steps.
KL_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
WHITE_INK_FROM = 8  # step 500
SKIPPED_COLOR = "#f0efec"  # neutral gray: boundary not scored (mid-codepoint)
TOP_ROWS_SHOWN = 32
CHUNK_SEPARATOR = "|"

BOUNDARY_HEADERS = [
    "step",
    "offset",
    "chunk",
    "coincide",
    "KL(C||F)",
    "KL(F||C)",
    "H forced",
    "H canon",
    "pm forced",
    "pm canon",
]
TOPK_HEADERS = ["token", "logprob", "prob", "Δprob vs other"]

# The model is loaded on first probe use so cache-only browsing never touches
# the GPU. Module-level cache: gradio handlers share the process.
_ADVISOR: dict[str, tuple[Any, Any, xtok_selection.Vocab]] = {}


def _advisor(model: str, revision: str, device: str) -> tuple[Any, Any, xtok_selection.Vocab]:
    if "advisor" not in _ADVISOR:
        tokenizer, lm = load_advisor(model, revision, device)
        _ADVISOR["advisor"] = (tokenizer, lm, xtok_selection.load_vocab(tokenizer))
    return _ADVISOR["advisor"]


def list_rollouts(cache_dir: str) -> list[str]:
    keys = sorted(os.path.basename(path)[: -len(".json")] for path in glob.glob(os.path.join(cache_dir, "*.json")))
    if not keys:
        raise ValueError(f"no scored rollouts in {cache_dir}; run scoring first")
    return keys


def read_scored(cache_dir: str, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with open(os.path.join(cache_dir, key + ".json")) as f:
        meta = json.load(f)
    with np.load(os.path.join(cache_dir, key + ".npz")) as data:
        arrays = {name: data[name] for name in data.files}
    return meta, arrays


def strip_html(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    kl_by_step = {
        int(step): float(kl) for step, kl in zip(arrays["step_index"], arrays["kl_canonical_forced"], strict=True)
    }
    drifted = [kl for step, kl in kl_by_step.items() if kl > 0]
    cap = max(float(np.percentile(drifted, 95)), 1e-6) if drifted else 1e-6
    parts = ['<div style="font-family:monospace; white-space:pre-wrap; word-break:break-all; line-height:1.8;">']
    for step, chunk_hex in enumerate(meta["chunks_hex"], start=1):
        text = bytes.fromhex(chunk_hex).decode("utf-8", errors="replace")
        kl = kl_by_step.get(step)
        if kl is None:
            color, ink = SKIPPED_COLOR, "#555555"
            tip = f"step {step}: boundary skipped (mid-codepoint)"
        else:
            index = min(len(KL_RAMP) - 1, int(kl / cap * (len(KL_RAMP) - 1)))
            color, ink = KL_RAMP[index], "#ffffff" if index >= WHITE_INK_FROM else "#1a1a1a"
            tip = f"step {step} · KL(C||F)={kl:.4f} nats"
        parts.append(
            f'<span title="{html.escape(tip)}" style="background:{color}; color:{ink};">{html.escape(text)}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def boundary_rows(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[list[Any]]:
    rows = []
    for i in range(len(arrays["step_index"])):
        step = int(arrays["step_index"][i])
        chunk = bytes.fromhex(meta["chunks_hex"][step - 1]).decode("utf-8", errors="replace")
        mass_f, mass_c = float(arrays["prefix_mass_forced"][i]), float(arrays["prefix_mass_canonical"][i])
        rows.append(
            [
                step,
                int(arrays["byte_offset"][i]),
                chunk[:24],
                "yes" if bool(arrays["coincide"][i]) else "",
                round(float(arrays["kl_canonical_forced"][i]), 4),
                round(float(arrays["kl_forced_canonical"][i]), 4),
                round(float(arrays["entropy_forced"][i]), 3),
                round(float(arrays["entropy_canonical"][i]), 3),
                "" if math.isnan(mass_f) else round(mass_f, 4),
                "" if math.isnan(mass_c) else round(mass_c, 4),
            ]
        )
    return rows


def _topk_rows(
    ids: np.ndarray, logprobs: np.ndarray, other_ids: np.ndarray, other_logprobs: np.ndarray, labels: dict[str, str]
) -> list[list[Any]]:
    other = {int(i): float(lp) for i, lp in zip(other_ids, other_logprobs, strict=True)}
    rows = []
    for token_id, logprob in zip(ids[:TOP_ROWS_SHOWN], logprobs[:TOP_ROWS_SHOWN], strict=True):
        prob = math.exp(float(logprob))
        other_logprob = other.get(int(token_id))
        delta = "" if other_logprob is None else round(prob - math.exp(other_logprob), 4)
        rows.append(
            [labels.get(str(int(token_id)), str(int(token_id))), round(float(logprob), 3), round(prob, 4), delta]
        )
    return rows


def boundary_detail(meta: dict[str, Any], arrays: dict[str, np.ndarray], row: int) -> tuple[str, list, list]:
    step = int(arrays["step_index"][row])
    labels = meta["token_labels"]
    summary = (
        f"**step {step}** · offset {int(arrays['byte_offset'][row])} · "
        f"coincide: {bool(arrays['coincide'][row])} · "
        f"KL(C||F)={float(arrays['kl_canonical_forced'][row]):.4f} · "
        f"KL(F||C)={float(arrays['kl_forced_canonical'][row]):.4f} · "
        f"H forced={float(arrays['entropy_forced'][row]):.3f} · "
        f"H canonical={float(arrays['entropy_canonical'][row]):.3f} · "
        f"prefix mass forced={float(arrays['prefix_mass_forced'][row]):.4f} / "
        f"canonical={float(arrays['prefix_mass_canonical'][row]):.4f}"
    )
    forced = _topk_rows(
        arrays["topk_forced_ids"][row],
        arrays["topk_forced_logprobs"][row],
        arrays["topk_canonical_ids"][row],
        arrays["topk_canonical_logprobs"][row],
        labels,
    )
    canonical = _topk_rows(
        arrays["topk_canonical_ids"][row],
        arrays["topk_canonical_logprobs"][row],
        arrays["topk_forced_ids"][row],
        arrays["topk_forced_logprobs"][row],
        labels,
    )
    return summary, forced, canonical


def probe_prefill(meta: dict[str, Any], arrays: dict[str, np.ndarray], row: int) -> tuple[str, str]:
    step = int(arrays["step_index"][row])
    chunks = [bytes.fromhex(chunk_hex).decode("utf-8", errors="replace") for chunk_hex in meta["chunks_hex"][:step]]
    return meta["prompt"], CHUNK_SEPARATOR.join(chunks)


def _comparison_outputs(
    comparison: Comparison, coincide: bool, vocab: xtok_selection.Vocab, tokenizer: Any
) -> tuple[str, list, list]:
    shown = np.concatenate([comparison.topk_forced_ids[:TOP_ROWS_SHOWN], comparison.topk_canonical_ids[:TOP_ROWS_SHOWN]])
    labels = {str(int(token_id)): token_label(int(token_id), vocab, tokenizer) for token_id in np.unique(shown)}
    summary = (
        f"coincide: {coincide} · KL(C||F)={comparison.kl_canonical_forced:.4f} · "
        f"KL(F||C)={comparison.kl_forced_canonical:.4f} · H forced={comparison.entropy_forced:.3f} · "
        f"H canonical={comparison.entropy_canonical:.3f}"
    )
    forced = _topk_rows(
        comparison.topk_forced_ids,
        comparison.topk_forced_logprobs,
        comparison.topk_canonical_ids,
        comparison.topk_canonical_logprobs,
        labels,
    )
    canonical = _topk_rows(
        comparison.topk_canonical_ids,
        comparison.topk_canonical_logprobs,
        comparison.topk_forced_ids,
        comparison.topk_forced_logprobs,
        labels,
    )
    return summary, forced, canonical


def build_app(cache_dir: str, model: str, revision: str, device: str) -> gr.Blocks:
    keys = list_rollouts(cache_dir)

    def on_rollout(key: str):
        meta, arrays = read_scored(cache_dir, key)
        return strip_html(meta, arrays), boundary_rows(meta, arrays), "", [], [], -1

    def on_select(key: str, evt: gr.SelectData):
        meta, arrays = read_scored(cache_dir, key)
        row = evt.index[0]
        summary, forced, canonical = boundary_detail(meta, arrays, row)
        return summary, forced, canonical, row

    def on_prefill(key: str, row: int):
        if row < 0:
            raise gr.Error("select a boundary row first")
        meta, arrays = read_scored(cache_dir, key)
        return probe_prefill(meta, arrays, row)

    def on_probe(prompt: str, chunked: str):
        tokenizer, lm, vocab = _advisor(model, revision, device)
        chunks = [part.encode("utf-8") for part in chunked.split(CHUNK_SEPARATOR) if part]
        if not chunks:
            raise gr.Error(f"enter a prefix with {CHUNK_SEPARATOR!r}-separated chunks")
        comparison, coincide = score_chunked_prefix(lm, tokenizer, vocab, prompt, chunks)
        return _comparison_outputs(comparison, coincide, vocab, tokenizer)

    with gr.Blocks(title="xtok advisor drift") as demo:
        gr.Markdown("## xtok advisor drift — forced vs canonical conditioning")
        rollout = gr.Dropdown(choices=keys, value=keys[0], label="rollout")
        selected_row = gr.State(-1)
        with gr.Tab("rollout"):
            strip = gr.HTML(label="completion (color = KL(C||F) at each chunk boundary)")
            boundaries = gr.Dataframe(headers=BOUNDARY_HEADERS, interactive=False, label="boundaries (click a row)")
            detail = gr.Markdown()
            with gr.Row():
                topk_forced = gr.Dataframe(headers=TOPK_HEADERS, interactive=False, label="forced conditioning")
                topk_canonical = gr.Dataframe(headers=TOPK_HEADERS, interactive=False, label="canonical conditioning")
        with gr.Tab("probe"):
            gr.Markdown(
                f"Edit a `{CHUNK_SEPARATOR}`-chunked prefix and re-score: each chunk is greedy-segmented "
                "(pipeline-style forcing) and compared against the canonical encoding of the same text. "
                "First use loads the model."
            )
            prefill = gr.Button("load selected boundary into probe")
            with gr.Accordion("prompt", open=False):
                probe_prompt = gr.Textbox(lines=6, label="prompt (bare-encoded, worker parity)")
            probe_chunks = gr.Textbox(lines=8, label=f"chunked prefix ({CHUNK_SEPARATOR} = chunk boundary)")
            probe_button = gr.Button("score", variant="primary")
            probe_summary = gr.Markdown()
            with gr.Row():
                probe_forced = gr.Dataframe(headers=TOPK_HEADERS, interactive=False, label="forced conditioning")
                probe_canonical = gr.Dataframe(headers=TOPK_HEADERS, interactive=False, label="canonical conditioning")

        rollout_outputs = [strip, boundaries, detail, topk_forced, topk_canonical, selected_row]
        rollout.change(on_rollout, inputs=[rollout], outputs=rollout_outputs)
        boundaries.select(on_select, inputs=[rollout], outputs=[detail, topk_forced, topk_canonical, selected_row])
        prefill.click(on_prefill, inputs=[rollout, selected_row], outputs=[probe_prompt, probe_chunks])
        probe_button.click(
            on_probe, inputs=[probe_prompt, probe_chunks], outputs=[probe_summary, probe_forced, probe_canonical]
        )
        demo.load(on_rollout, inputs=[rollout], outputs=rollout_outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", default=ADVISOR_MODEL, help="HF repo id or local model path (probe tab only)")
    parser.add_argument("--revision", default=ADVISOR_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", default="127.0.0.1")
    args = parser.parse_args()
    demo = build_app(args.cache_dir, args.model, args.revision, args.device)
    demo.launch(server_name=args.server_name, server_port=args.port)


if __name__ == "__main__":
    main()
