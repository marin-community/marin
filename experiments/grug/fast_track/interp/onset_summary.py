# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-tag summary for the layer onset study, computed next to the data so only a small JSON leaves the region.

From ``stats.npz``, ``atlas.npz`` and ``ngram_score.npz``: the atlas unit categories per layer (``atlas_report``; n-gram
units by the census screening criteria ``ngram_validate.eligible`` only), per-layer firing and routing statistics, and
the n-gram screening counts. Writes ``onset_summary.json``. CPU only.
"""

import argparse
import json
import logging

import fsspec
import numpy as np

from experiments.grug.fast_track.interp import atlas_report, common, forward, ngram, ngram_validate

logger = logging.getLogger(__name__)

L = forward.NUM_LAYERS


def entropy_bits(p: np.ndarray) -> float:
    p = p / p.sum()
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def layer_stats(s: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Per layer: routed-unit firing (share of the expert's tokens with a positive pre-activation), dense / dead units,
    MI(expert; current token) over routing assignments, load entropy, mean effective experts, QB-bias-driven slots."""
    tokens_e = s["expert_tokens"].astype(np.float64)
    fire = s["fire"].astype(np.float64)
    rate = fire / np.maximum(tokens_e[..., None], 1)
    eti = s["expert_token_ids"].astype(np.float64)
    eff = s["eff_hist"].astype(np.float64)
    edges = np.linspace(1, 11, eff.shape[1] + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    out = []
    for l in range(L):
        live = tokens_e[l] > 0
        h_e, h_v, h_ev = entropy_bits(eti[l].sum(1)), entropy_bits(eti[l].sum(0)), entropy_bits(eti[l].reshape(-1))
        out.append(
            dict(
                fire_mean=float(np.mean(rate[l][live])),
                fire_median=float(np.median(rate[l][live])),
                dense_frac=float(np.mean(rate[l][live] > 0.5)),
                dead_units=int(np.sum((fire[l] == 0) & live[:, None])),
                dead_experts=int(np.sum(~live)),
                mi_expert_token=h_e + h_v - h_ev,
                mi_norm=(h_e + h_v - h_ev) / h_e,
                load_entropy_norm=h_e / np.log2(eti.shape[1]),
                eff_experts=float(np.sum(eff[l] * centres) / np.sum(eff[l])),
                bias_driven=float(np.sum(s["bias_driven"][l]) / max(np.sum(tokens_e[l]), 1)),
            )
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/vocab.json") as f:
        vocab = json.load(f)
    score = ngram_validate.score_metrics(ngram.load_npz(args.tag, "ngram_score"))
    ok = ngram_validate.eligible(score)
    ui = score["unit_index"]
    routed = ok & (ui[:, 1] >= 0)
    ok_mask = np.zeros((L, forward.NUM_EXPERTS, forward.EXPERT_WIDTH), bool)
    ok_mask[ui[routed, 0], ui[routed, 1], ui[routed, 2]] = True
    screen = np.where(ok, score["concentration"] * np.maximum(score["r2_gain"], 0.0), 0.0)
    z = ngram.load_npz(args.tag, "atlas")
    um = atlas_report.unit_metrics(z, vocab, ok_mask)
    cats = atlas_report.CATEGORIES
    summary = {
        "categories": {c: [int((um["category"][l] == i).sum()) for l in range(L)] for i, c in enumerate(cats)},
        "ngram_eligible": [int(np.sum(routed & (ui[:, 0] == l))) for l in range(L)],
        "ngram_screen_sum": [float(np.sum(screen[(ui[:, 0] == l) & (ui[:, 1] >= 0)])) for l in range(L)],
        "stats": layer_stats(ngram.load_npz(args.tag, "stats")),
    }
    logger.info("categories %s", summary["categories"])
    common.save_json(args.tag, "onset_summary", summary)


if __name__ == "__main__":
    main()
