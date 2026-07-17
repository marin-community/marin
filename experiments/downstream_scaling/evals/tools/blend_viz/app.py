# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browse scored A/B rollouts and probe the same resident model pair."""

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
from experiments.downstream_scaling.evals.tools.blend_viz.scoring import (
    ROW_CUT,
    ROW_EOS,
    TOPK_STORE,
    ScoredRows,
    load_models,
    resolve_cache_dir,
    score_probe,
    token_label,
)

COLOR_RAMP = (
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
NEUTRAL_COLOR = "#f0efec"
WHITE_INK_FROM = 8

STEP_HEADERS = [
    "row",
    "committed",
    "A logprob",
    "A rank",
    "B logprob",
    "B rank",
    "H(A)",
    "H(B)",
    "KL(A||B)",
    "blend argmax",
    "flip alpha",
]
SIDE_HEADERS = ["token", "id", "logprob", "probability", "delta probability vs other"]
BLEND_HEADERS = ["token", "id", "A score", "B score", "blend score", "probability", "source"]
METRICS = ("KL(A || B)", "committed logprob B - A", "A entropy", "flip_alpha")
CANONICAL_PREFIX_MODE = "Prefix mode: canonically encoded text"
RECORDED_PREFIX_MODE = "Prefix mode: recorded token IDs"

_MODEL_PAIRS: dict[tuple[str, str, str, str, str], tuple[Any, Any, Any, xtok_selection.Vocab]] = {}


def list_rollouts(cache_dir: str) -> list[tuple[str, str]]:
    choices = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        with open(path) as f:
            meta = json.load(f)
        key = os.path.basename(path)[: -len(".json")]
        result = "pass" if meta["passed"] else "fail"
        choices.append(
            (f"{meta['problem_id']} · alpha={meta['advisor_weight']:g} · {result} · r{meta['sample_rank']}", key)
        )
    if not choices:
        raise ValueError(f"no scored rollouts in {cache_dir}; run scoring first")
    return choices


def read_scored(cache_dir: str, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with open(os.path.join(cache_dir, key + ".json")) as f:
        meta = json.load(f)
    with np.load(os.path.join(cache_dir, key + ".npz")) as data:
        arrays = {name: data[name] for name in data.files}
    return meta, arrays


def _candidates(
    meta: dict[str, Any], ids: np.ndarray, logprobs: np.ndarray, k: int
) -> dict[xtok_selection.Key, xtok_selection.Candidate]:
    out: dict[xtok_selection.Key, xtok_selection.Candidate] = {}
    token_bytes = meta["token_bytes_hex"]
    eos_id = int(meta["eos_token_id"])
    for token_id_value, logprob_value in zip(ids[:k], logprobs[:k], strict=True):
        token_id = int(token_id_value)
        logprob = float(logprob_value)
        if not math.isfinite(logprob):
            continue
        if token_id == eos_id:
            out[xtok_selection.EOS_KEY] = xtok_selection.Candidate(token_id, logprob)
        elif (piece := token_bytes.get(str(token_id))) is not None:
            out[bytes.fromhex(piece)] = xtok_selection.Candidate(token_id, logprob)
    if not out:
        raise ValueError("chosen top-k contains no ordinary or EOS candidates")
    return out


def _token_id(
    key: xtok_selection.Key,
    a: dict[xtok_selection.Key, xtok_selection.Candidate],
    b: dict[xtok_selection.Key, xtok_selection.Candidate],
) -> int:
    return (a.get(key) or b[key]).token_id


def _argmax(
    scores: dict[xtok_selection.Key, float],
    a: dict[xtok_selection.Key, xtok_selection.Candidate],
    b: dict[xtok_selection.Key, xtok_selection.Candidate],
) -> xtok_selection.Key:
    return min(scores, key=lambda key: (-scores[key], _token_id(key, a, b)))


def _blend(
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    row: int,
    alpha: float,
    temperature: float,
    k_a: int,
    k_b: int,
) -> tuple[
    dict[xtok_selection.Key, xtok_selection.Candidate],
    dict[xtok_selection.Key, xtok_selection.Candidate],
    dict[xtok_selection.Key, float],
    dict[xtok_selection.Key, float],
]:
    if temperature < 0:
        raise gr.Error("temperature must be non-negative")
    a = _candidates(meta, arrays["a_topk_ids"][row], arrays["a_topk_logprobs"][row], int(k_a))
    b = _candidates(meta, arrays["b_topk_ids"][row], arrays["b_topk_logprobs"][row], int(k_b))
    scores = xtok_selection.avg_bytes_union_scores(a, b, float(alpha))
    if temperature == 0:
        winner = _argmax(scores, a, b)
        probs = {key: float(key == winner) for key in scores}
    else:
        high = max(scores.values())
        exps = {key: math.exp((score - high) / temperature) for key, score in scores.items()}
        total = sum(exps.values())
        probs = {key: value / total for key, value in exps.items()}
    return a, b, scores, probs


def flip_alpha(meta: dict[str, Any], arrays: dict[str, np.ndarray], row: int) -> float | None:
    a = _candidates(meta, arrays["a_topk_ids"][row], arrays["a_topk_logprobs"][row], 16)
    b = _candidates(meta, arrays["b_topk_ids"][row], arrays["b_topk_logprobs"][row], 16)
    base_scores = xtok_selection.avg_bytes_union_scores(a, b, 0.0)
    base = _argmax(base_scores, a, b)
    for step in range(1, 101):
        alpha = step / 100.0
        scores = xtok_selection.avg_bytes_union_scores(a, b, alpha)
        if _argmax(scores, a, b) != base:
            return alpha
    return None


def _label(meta: dict[str, Any], token_id: int) -> str:
    if token_id == meta["eos_token_id"]:
        return "<eos>"
    return meta["token_labels"].get(str(token_id), str(token_id))


def _side_rows(
    meta: dict[str, Any], ids: np.ndarray, logprobs: np.ndarray, other_logprobs: np.ndarray, k: int
) -> list[list[Any]]:
    rows = []
    for token_id_value, logprob_value, other_value in zip(ids[:k], logprobs[:k], other_logprobs[:k], strict=True):
        token_id = int(token_id_value)
        logprob = float(logprob_value)
        other = float(other_value)
        rows.append(
            [
                _label(meta, token_id),
                token_id,
                round(logprob, 4),
                round(math.exp(logprob), 6),
                round(math.exp(logprob) - math.exp(other), 6),
            ]
        )
    return rows


def detail(
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    row: int,
    alpha: float,
    temperature: float,
    k_a: int,
    k_b: int,
) -> tuple[str, list[list[Any]], list[list[Any]], list[list[Any]]]:
    a, b, scores, probs = _blend(meta, arrays, row, alpha, temperature, k_a, k_b)
    committed_id = int(arrays["committed_ids"][row])
    committed = "<cut>" if committed_id < 0 else _label(meta, committed_id)
    summary = (
        f"**row {row}** · committed {committed} · KL(A||B)={float(arrays['kl_a_b'][row]):.5f} · "
        f"H(A)={float(arrays['a_entropy'][row]):.4f} · H(B)={float(arrays['b_entropy'][row]):.4f}"
    )
    rows_a = _side_rows(
        meta,
        arrays["a_topk_ids"][row],
        arrays["a_topk_logprobs"][row],
        arrays["b_logprobs_at_a_topk"][row],
        int(k_a),
    )
    rows_b = _side_rows(
        meta,
        arrays["b_topk_ids"][row],
        arrays["b_topk_logprobs"][row],
        arrays["a_logprobs_at_b_topk"][row],
        int(k_b),
    )
    a_floor = min(candidate.logit for candidate in a.values())
    b_floor = min(candidate.logit for candidate in b.values())
    ordered = sorted(scores, key=lambda key: (-probs[key], _token_id(key, a, b)))
    blend_rows = []
    for key in ordered:
        token_id = _token_id(key, a, b)
        source = "A+B" if key in a and key in b else "A; B=floor" if key in a else "B; A=floor"
        blend_rows.append(
            [
                _label(meta, token_id),
                token_id,
                round(a[key].logit if key in a else a_floor, 4),
                round(b[key].logit if key in b else b_floor, 4),
                round(scores[key], 4),
                round(probs[key], 6),
                source,
            ]
        )
    return summary, rows_a, rows_b, blend_rows


def step_rows(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[list[Any]]:
    rows = []
    alpha = float(meta["advisor_weight"])
    temperature = float(meta["temperature"])
    for row in range(len(arrays["row_kind"])):
        kind = int(arrays["row_kind"][row])
        token_id = int(arrays["committed_ids"][row])
        committed = "<cut>" if kind == ROW_CUT else "<eos>" if kind == ROW_EOS else _label(meta, token_id)
        a, b, scores, _ = _blend(
            meta,
            arrays,
            row,
            alpha,
            temperature,
            int(meta["top_k_a"]),
            int(meta["top_k_b"]),
        )
        winner = _token_id(_argmax(scores, a, b), a, b)
        flip = flip_alpha(meta, arrays, row)
        a_lp = float(arrays["a_committed_logprob"][row])
        b_lp = float(arrays["b_committed_logprob"][row])
        rows.append(
            [
                row,
                committed,
                "" if math.isnan(a_lp) else round(a_lp, 4),
                "" if int(arrays["a_committed_rank"][row]) < 0 else int(arrays["a_committed_rank"][row]),
                "" if math.isnan(b_lp) else round(b_lp, 4),
                "" if int(arrays["b_committed_rank"][row]) < 0 else int(arrays["b_committed_rank"][row]),
                round(float(arrays["a_entropy"][row]), 4),
                round(float(arrays["b_entropy"][row]), 4),
                round(float(arrays["kl_a_b"][row]), 5),
                _label(meta, winner),
                "never" if flip is None else f"{flip:.2f}",
            ]
        )
    return rows


def _metric_values(meta: dict[str, Any], arrays: dict[str, np.ndarray], metric: str) -> list[float | None]:
    token_rows = len(meta["chunks_hex"])
    if metric == "KL(A || B)":
        return [float(value) for value in arrays["kl_a_b"][:token_rows]]
    if metric == "committed logprob B - A":
        return [
            float(b - a)
            for a, b in zip(
                arrays["a_committed_logprob"][:token_rows], arrays["b_committed_logprob"][:token_rows], strict=True
            )
        ]
    if metric == "A entropy":
        return [float(value) for value in arrays["a_entropy"][:token_rows]]
    return [flip_alpha(meta, arrays, row) for row in range(token_rows)]


def strip_html(meta: dict[str, Any], arrays: dict[str, np.ndarray], metric: str) -> str:
    values = _metric_values(meta, arrays, metric)
    finite = [value for value in values if value is not None and math.isfinite(value)]
    low, high = (min(finite), max(finite)) if finite else (0.0, 0.0)
    parts = ['<div style="font-family:monospace; white-space:pre-wrap; word-break:break-all; line-height:1.8;">']
    for row, (chunk_hex, value) in enumerate(zip(meta["chunks_hex"], values, strict=True)):
        text = bytes.fromhex(chunk_hex).decode("utf-8", errors="replace")
        if value is None or not math.isfinite(value):
            color, ink, shown = NEUTRAL_COLOR, "#555555", "never"
        else:
            fraction = 0.0 if high == low else (value - low) / (high - low)
            index = min(len(COLOR_RAMP) - 1, int(fraction * (len(COLOR_RAMP) - 1)))
            color = COLOR_RAMP[index]
            ink = "#ffffff" if index >= WHITE_INK_FROM else "#1a1a1a"
            shown = f"{value:.5g}"
        tip = html.escape(f"row {row} · {metric}={shown}")
        parts.append(f'<span title="{tip}" style="background:{color}; color:{ink};">{html.escape(text)}</span>')
    parts.append("</div>")
    return "".join(parts)


def probe_prefill(meta: dict[str, Any], arrays: dict[str, np.ndarray], row: int) -> tuple[str, str, list[int], str]:
    prefix_bytes = b"".join(bytes.fromhex(chunk) for chunk in meta["chunks_hex"][:row])
    prefix = prefix_bytes.decode("utf-8", errors="backslashreplace")
    prefix_ids = [int(token_id) for token_id in arrays["committed_ids"][:row]]
    return meta["prompt"], prefix, prefix_ids, RECORDED_PREFIX_MODE


def _models(
    meta: dict[str, Any], decoder_override: str | None, advisor_override: str | None, device: str
) -> tuple[Any, Any, Any, xtok_selection.Vocab]:
    decoder_model = decoder_override or meta["decoder_model"]
    advisor_model = advisor_override or meta["advisor_model"]
    key = (decoder_model, meta["decoder_revision"], advisor_model, meta["advisor_revision"], device)
    if key not in _MODEL_PAIRS:
        tokenizer, vocab, decoder, advisor = load_models(
            decoder_model,
            advisor_model,
            device=device,
            decoder_revision=meta["decoder_revision"],
            advisor_revision=meta["advisor_revision"],
        )
        _MODEL_PAIRS[key] = tokenizer, decoder, advisor, vocab
    return _MODEL_PAIRS[key]


def _probe_metadata(
    meta: dict[str, Any], scored: ScoredRows, tokenizer: Any, vocab: xtok_selection.Vocab
) -> dict[str, Any]:
    token_ids = set(scored.a_topk_ids.ravel().tolist()) | set(scored.b_topk_ids.ravel().tolist())
    token_bytes_hex = {}
    labels = {}
    for token_id in token_ids:
        labels[str(token_id)] = token_label(token_id, vocab, tokenizer)
        if token_id == vocab.eos_id:
            continue
        piece = vocab.token_bytes[token_id] if token_id < len(vocab.token_bytes) else None
        if piece is not None:
            token_bytes_hex[str(token_id)] = piece.hex()
    return {**meta, "eos_token_id": vocab.eos_id, "token_bytes_hex": token_bytes_hex, "token_labels": labels}


def build_app(cache_dir: str, *, decoder_model: str | None, advisor_model: str | None, device: str) -> gr.Blocks:
    choices = list_rollouts(cache_dir)
    first_key = choices[0][1]

    def on_rollout(key: str, metric: str):
        meta, arrays = read_scored(cache_dir, key)
        alpha = float(meta["advisor_weight"])
        temperature = float(meta["temperature"])
        k_a, k_b = int(meta["top_k_a"]), int(meta["top_k_b"])
        summary, rows_a, rows_b, blend_rows = detail(meta, arrays, 0, alpha, temperature, k_a, k_b)
        return (
            strip_html(meta, arrays, metric),
            step_rows(meta, arrays),
            summary,
            rows_a,
            rows_b,
            blend_rows,
            0,
            alpha,
            temperature,
            k_a,
            k_b,
        )

    def on_metric(key: str, metric: str):
        meta, arrays = read_scored(cache_dir, key)
        return strip_html(meta, arrays, metric)

    def on_select(key: str, alpha: float, temperature: float, k_a: int, k_b: int, evt: gr.SelectData):
        meta, arrays = read_scored(cache_dir, key)
        row = int(evt.index[0])
        return (*detail(meta, arrays, row, alpha, temperature, k_a, k_b), row)

    def on_controls(key: str, row: int, alpha: float, temperature: float, k_a: int, k_b: int):
        meta, arrays = read_scored(cache_dir, key)
        return detail(meta, arrays, int(row), alpha, temperature, k_a, k_b)

    def on_prefill(key: str, row: int):
        meta, arrays = read_scored(cache_dir, key)
        return probe_prefill(meta, arrays, int(row))

    def on_prefix_edit() -> tuple[None, str]:
        return None, CANONICAL_PREFIX_MODE

    def on_probe(
        key: str,
        prompt: str,
        prefix: str,
        recorded_prefix_ids: list[int] | None,
        alpha: float,
        temperature: float,
        k_a: int,
        k_b: int,
    ):
        meta, _ = read_scored(cache_dir, key)
        tokenizer, decoder, advisor, vocab = _models(meta, decoder_model, advisor_model, device)
        scored = score_probe(
            decoder,
            advisor,
            tokenizer,
            prompt,
            prefix,
            recorded_prefix_ids=recorded_prefix_ids,
        )
        probe_meta = _probe_metadata(meta, scored, tokenizer, vocab)
        return detail(
            probe_meta,
            {name: getattr(scored, name) for name in scored.__dataclass_fields__},
            0,
            alpha,
            temperature,
            k_a,
            k_b,
        )

    with gr.Blocks(title="A/B blend visualizer") as demo:
        gr.Markdown("## A/B blend visualizer")
        rollout = gr.Dropdown(choices=choices, value=first_key, label="rollout")
        selected_row = gr.State(0)
        recorded_prefix_ids = gr.State(None)
        with gr.Row():
            metric = gr.Dropdown(choices=METRICS, value=METRICS[0], label="strip metric")
            alpha = gr.Slider(0.0, 1.0, step=0.01, label="alpha")
            temperature = gr.Number(minimum=0.0, label="temperature")
            k_a = gr.Slider(1, TOPK_STORE, step=1, label="k A")
            k_b = gr.Slider(1, TOPK_STORE, step=1, label="k B")
        with gr.Tab("rollout"):
            strip = gr.HTML(label="completion")
            steps = gr.Dataframe(headers=STEP_HEADERS, interactive=False, label="decision rows (click a row)")
            summary = gr.Markdown()
            with gr.Row():
                table_a = gr.Dataframe(headers=SIDE_HEADERS, interactive=False, label="A top-k")
                table_b = gr.Dataframe(headers=SIDE_HEADERS, interactive=False, label="B top-k")
            blend_table = gr.Dataframe(headers=BLEND_HEADERS, interactive=False, label="production byte-keyed blend")
        with gr.Tab("probe"):
            gr.Markdown(
                "Loading a selected row uses its recorded prefix IDs until the generated prefix is edited. "
                "First use loads both models and keeps them resident."
            )
            prefill = gr.Button("load selected row into probe")
            probe_prompt = gr.Textbox(lines=8, label="prompt")
            probe_prefix = gr.Textbox(lines=8, label="generated prefix")
            prefix_mode = gr.Markdown(CANONICAL_PREFIX_MODE)
            probe_button = gr.Button("score next token", variant="primary")
            probe_summary = gr.Markdown()
            with gr.Row():
                probe_a = gr.Dataframe(headers=SIDE_HEADERS, interactive=False, label="A top-k")
                probe_b = gr.Dataframe(headers=SIDE_HEADERS, interactive=False, label="B top-k")
            probe_blend = gr.Dataframe(headers=BLEND_HEADERS, interactive=False, label="production byte-keyed blend")

        rollout_outputs = [
            strip,
            steps,
            summary,
            table_a,
            table_b,
            blend_table,
            selected_row,
            alpha,
            temperature,
            k_a,
            k_b,
        ]
        rollout.change(on_rollout, inputs=[rollout, metric], outputs=rollout_outputs)
        metric.change(on_metric, inputs=[rollout, metric], outputs=[strip])
        steps.select(
            on_select,
            inputs=[rollout, alpha, temperature, k_a, k_b],
            outputs=[summary, table_a, table_b, blend_table, selected_row],
        )
        for control in (alpha, temperature, k_a, k_b):
            control.change(
                on_controls,
                inputs=[rollout, selected_row, alpha, temperature, k_a, k_b],
                outputs=[summary, table_a, table_b, blend_table],
            )
        prefill.click(
            on_prefill,
            inputs=[rollout, selected_row],
            outputs=[probe_prompt, probe_prefix, recorded_prefix_ids, prefix_mode],
        )
        probe_prefix.input(on_prefix_edit, outputs=[recorded_prefix_ids, prefix_mode])
        probe_button.click(
            on_probe,
            inputs=[rollout, probe_prompt, probe_prefix, recorded_prefix_ids, alpha, temperature, k_a, k_b],
            outputs=[probe_summary, probe_a, probe_b, probe_blend],
        )
        demo.load(on_rollout, inputs=[rollout, metric], outputs=rollout_outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None, help="cache dir; default $BLEND_VIZ_CACHE")
    parser.add_argument("--decoder-model", default=None, help="optional local/HF override for live probes")
    parser.add_argument("--advisor-model", default=None, help="optional local/HF override for live probes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", default="127.0.0.1")
    args = parser.parse_args()
    app = build_app(
        resolve_cache_dir(args.cache_dir),
        decoder_model=args.decoder_model,
        advisor_model=args.advisor_model,
        device=args.device,
    )
    app.launch(server_name=args.server_name, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
