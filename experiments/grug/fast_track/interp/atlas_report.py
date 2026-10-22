# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron and expert atlas from local copies of ``atlas.npz``, ``atlas_tokens.npz``, ``vocab.json`` and the n-gram
census inputs (``bigram/metrics.npz``, ``bigram/rows.json``): per-unit metrics and an automatic category, per-expert
profiles and labels, per-layer expert clusters, and the topic-unit table the routing pass reads.

Unit categories, first match wins:

- ``ngram``: an eligible (a, b) -> c unit of the n-gram census (``ngram_validate.eligible``);
- ``structural``: at least ``DOC_START_SHARE`` of its high activations in a document's first 4 tokens, or at least
  ``STRUCT_SHARE`` of its high mass on whitespace / newline / indentation tokens;
- ``token``: at least ``TOKEN_SHARE`` of its high mass on one current token;
- ``output``: its logit-lens top token is at least ``OUTPUT_Z`` logit standard deviations above the vocabulary mean
  and ``OUTPUT_MARGIN`` standard deviations above the runner-up;
- ``topic``: its Paloma high activations concentrated in one domain group (share at least ``DOMAIN_SHARE`` and lift
  at least ``DOMAIN_LIFT`` over Paloma's token mix) or its top tokens a semantic cluster (mean embedding cosine of its
  top tokens at least ``CLUSTER_COS`` and ``CLUSTER_LIFT`` times its expert's);
- ``dense``: fires (pre-activation > 0) on at least ``DENSE_FIRE`` of its expert's tokens with its top token under
  ``DENSE_CONCENTRATION`` of its high mass;
- ``polysemantic``: anything else.

"High" activations are above ``atlas.HIGH_FRACTION`` of the unit's mean top-16 activation (about the top 0.6% of
its expert's tokens).

Usage: ``python -m experiments.grug.fast_track.interp.atlas_report <tag data dir> <bigram dir> <out dir>``.
"""

import json
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from experiments.grug.fast_track.interp import atlas, ngram_validate

CATEGORIES = ("ngram", "structural", "token", "output", "topic", "dense", "polysemantic")
DOC_START_SHARE = 0.5
STRUCT_SHARE = 0.6
TOKEN_SHARE = 0.5
OUTPUT_Z = 8.0
OUTPUT_MARGIN = 1.5
DOMAIN_SHARE = 0.5
DOMAIN_LIFT = 2.5
MIN_PALOMA_HIGH = 30
CLUSTER_COS = 0.3
CLUSTER_LIFT = 1.5
CLUSTER_TOP = 8
DENSE_FIRE = 0.5
DENSE_CONCENTRATION = 0.2
MIN_HIGH = 20
CLUSTERS_PER_LAYER = 24
GROUP_OF_SOURCE = np.asarray([atlas.GROUPS.index(atlas.SOURCE_GROUP[s]) for s in atlas.SOURCES])


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path: str, **kwargs) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)


def structural_tokens(vocab: list[str]) -> np.ndarray:
    return np.asarray([(s != "" and s.strip(" \t") == "") or ("\n" in s and s.strip() == "") for s in vocab])


def group_counts(per_source: np.ndarray) -> np.ndarray:
    """``[..., D]`` per-source counts to ``[..., G]`` per group."""
    out = np.zeros((*per_source.shape[:-1], len(atlas.GROUPS)), np.float64)
    for d, g in enumerate(GROUP_OF_SOURCE):
        out[..., g] += per_source[..., d]
    return out


def embedding_cosines(embed: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean pairwise cosine (off-diagonal) of the token embeddings ``ids [..., n]``."""
    e = embed[ids]  # [..., n, d]
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-6)
    cos = np.einsum("...id,...jd->...ij", e, e)
    w = weights / np.maximum(weights.sum(-1, keepdims=True), 1e-9)
    ww = w[..., :, None] * w[..., None, :]
    n = ids.shape[-1]
    off = 1.0 - np.eye(n)
    return (cos * ww * off).sum((-1, -2)) / np.maximum((ww * off).sum((-1, -2)), 1e-9)


def unit_metrics(z: dict, vocab: list[str], ngram_ok: np.ndarray) -> dict[str, np.ndarray]:
    """Per routed unit ``[L, E, I]`` metrics and the category index."""
    routed = z["expert_token_ids"].sum(-1).astype(np.float64)  # [L, E]
    n = np.maximum(routed[..., None], 1)
    high = z["high"].astype(np.float64)
    hm = np.maximum(z["high_mass"].astype(np.float64), 1e-9)
    tok_mass = z["tok_mass"].astype(np.float64)  # [L, E, NT, I]
    share = tok_mass / hm[:, :, None, :]
    top_slot = np.argsort(-tok_mass[:, :, :-1, :], axis=2)[:, :, :3, :]  # [L, E, 3, I]
    top_ids = np.take_along_axis(z["expert_top_ids"][..., None].astype(np.int64), top_slot, axis=2)
    top_share = np.take_along_axis(share[:, :, :-1, :], top_slot, axis=2)
    struct_tok = structural_tokens(vocab)
    struct_share = (share[:, :, :-1, :] * struct_tok[z["expert_top_ids"].astype(np.int64)][..., None]).sum(2)
    pos = z["high_pos"].astype(np.float64)  # [L, E, P, I]
    doc_start = pos[:, :, :2, :].sum(2) / np.maximum(high, 1)
    dom = z["high_dom"].astype(np.float64)  # [L, E, I, D]
    pal = dom.copy()
    pal[..., 0] = 0
    pal_g = group_counts(pal)  # [L, E, I, G] (train group zero)
    pal_n = pal_g.sum(-1)
    base = group_counts(np.where(np.arange(len(atlas.SOURCES)) == 0, 0, z["dom_tokens"]).astype(np.float64))
    base = base / base.sum()
    pal_share = pal_g / np.maximum(pal_n[..., None], 1)
    lift = pal_share / np.maximum(base, 1e-9)
    dom_g = np.argmax(pal_share, axis=-1)
    dom_share = np.max(pal_share, axis=-1)
    dom_lift = np.take_along_axis(lift, dom_g[..., None], axis=-1)[..., 0]
    embed = z["token_embed"].astype(np.float32)
    embed = embed / np.maximum(np.linalg.norm(embed, axis=-1, keepdims=True), 1e-6)
    tm = tok_mass[:, :, :-1, :]  # [L, E, 64, I]
    kth = -np.sort(-tm, axis=2)[:, :, CLUSTER_TOP - 1 : CLUSTER_TOP, :]
    w = np.where(tm >= kth, tm, 0.0)
    w = w / np.maximum(w.sum(2, keepdims=True), 1e-12)
    unit_cos = np.zeros(high.shape, np.float32)
    expert_cos = np.zeros(routed.shape, np.float32)
    counts = z["expert_token_ids"]
    for layer in range(high.shape[0]):
        e64 = embed[z["expert_top_ids"][layer].astype(np.int64)]  # [E, 64, d]
        cos = np.einsum("ejd,ekd->ejk", e64, e64)
        quad = np.einsum("eji,ejk,eki->ei", w[layer], cos, w[layer])
        sq = np.sum(w[layer] ** 2, axis=1)
        unit_cos[layer] = (quad - sq) / np.maximum(1 - sq, 1e-9)
        ids = z["expert_top_ids"][layer][:, :CLUSTER_TOP].astype(np.int64)
        expert_cos[layer] = embedding_cosines(embed, ids, np.take_along_axis(counts[layer], ids, -1).astype(np.float64))
    lens_z = z["lens_top_val"][..., 0] / np.maximum(z["lens_logit_std"], 1e-9)
    lens_margin = (z["lens_top_val"][..., 0] - z["lens_top_val"][..., 1]) / np.maximum(z["lens_logit_std"], 1e-9)
    high_rate = high / n
    enough = high >= MIN_HIGH

    cat = np.full(high.shape, CATEGORIES.index("polysemantic"), np.int8)
    rules = [
        ("dense", enough & (z["fire"] / n >= DENSE_FIRE) & (top_share[:, :, 0, :] < DENSE_CONCENTRATION)),
        (
            "topic",
            enough
            & (
                ((pal_n >= MIN_PALOMA_HIGH) & (dom_share >= DOMAIN_SHARE) & (dom_lift >= DOMAIN_LIFT))
                | ((unit_cos >= CLUSTER_COS) & (unit_cos >= CLUSTER_LIFT * expert_cos[..., None]))
            ),
        ),
        ("output", (lens_z >= OUTPUT_Z) & (lens_margin >= OUTPUT_MARGIN)),
        ("token", enough & (top_share[:, :, 0, :] >= TOKEN_SHARE)),
        ("structural", enough & ((doc_start >= DOC_START_SHARE) | (struct_share >= STRUCT_SHARE))),
        ("ngram", ngram_ok),
    ]
    for name, mask in rules:  # reverse precedence: later rules overwrite
        cat[mask] = CATEGORIES.index(name)
    topic_by_domain = (pal_n >= MIN_PALOMA_HIGH) & (dom_share >= DOMAIN_SHARE) & (dom_lift >= DOMAIN_LIFT)
    return dict(
        category=cat,
        fire_rate=z["fire"] / n,
        mean_act=z["act_sum"] / n,
        max_act=z["act_max"],
        high=high,
        high_rate=high_rate,
        concentration=top_share[:, :, 0, :],
        top_ids=np.moveaxis(top_ids, 2, -1),  # [L, E, I, 3]
        top_share=np.moveaxis(top_share, 2, -1),
        struct_share=struct_share,
        doc_start=doc_start,
        dom_group=dom_g,
        dom_share=dom_share,
        dom_lift=dom_lift,
        pal_share=pal_share,
        topic_by_domain=topic_by_domain,
        unit_cos=unit_cos,
        expert_cos=expert_cos,
        lens_z=lens_z,
        lens_margin=lens_margin,
        routed=routed,
    )


def ngram_eligible(bigram_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """``[L, E, I]`` eligible census units and their validated-clear flag (-1 unknown)."""
    m = dict(np.load(f"{bigram_dir}/metrics.npz"))
    ok = ngram_validate.eligible(m)
    shape = (atlas.forward.NUM_LAYERS, atlas.forward.NUM_EXPERTS, atlas.forward.EXPERT_WIDTH)
    out = np.zeros(shape, bool)
    clear = np.full(shape, -1, np.int8)
    ui = m["unit_index"]
    routed = ok & (ui[:, 1] >= 0)
    out[ui[routed, 0], ui[routed, 1], ui[routed, 2]] = True
    for r in read_json(f"{bigram_dir}/rows.json"):
        if r["shared"]:
            continue
        lyr, e, u = (int(x[1:]) for x in r["unit"].split("·"))
        clear[lyr, e, u] = int(r["clear"])
    return out, clear


def topic_tables(um: dict, z: dict) -> dict[str, np.ndarray]:
    """The routing pass's topic-unit inputs: ``topic_dom [L, E, I, G]`` (domain topic units' group) and
    ``topic_slot [L, E, I, NT]`` (semantic-cluster topic units' top tokens, as expert top-token slots)."""
    topic = um["category"] == CATEGORIES.index("topic")
    dom = np.zeros((*topic.shape, len(atlas.GROUPS)), bool)
    by_dom = topic & um["topic_by_domain"]
    li, ei, ii = np.nonzero(by_dom)
    dom[li, ei, ii, um["dom_group"][li, ei, ii]] = True
    slot = np.zeros((*topic.shape, atlas.TOP_TOKENS + 1), bool)
    by_tok = topic & ~um["topic_by_domain"]
    tm = z["tok_mass"][:, :, :-1, :].astype(np.float64)  # [L, E, 64, I]
    order = np.argsort(-tm, axis=2)[:, :, :CLUSTER_TOP, :]
    li, ei, ii = np.nonzero(by_tok)
    for j in range(CLUSTER_TOP):
        slot[li, ei, ii, order[li, ei, j, ii]] = True
    return dict(topic_dom=dom, topic_slot=slot)


def expert_profiles(z: dict, um: dict, vocab: list[str]) -> dict:
    counts = z["expert_token_ids"].astype(np.float64)  # [L, E, V]
    total = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(total, 1)
    entropy = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-30)), 0)).sum(-1)
    top = np.argsort(-counts, axis=-1)[..., :20]
    top_p = np.take_along_axis(p, top, -1)
    dom = z["expert_dom"].astype(np.float64)
    pal = dom.copy()
    pal[..., 0] = 0
    pal_g = group_counts(pal)
    base = group_counts(np.where(np.arange(len(atlas.SOURCES)) == 0, 0, z["dom_tokens"]).astype(np.float64))
    base = base / base.sum()
    pal_share = pal_g / np.maximum(pal_g.sum(-1, keepdims=True), 1)
    lift = pal_share / base
    all_tokens = z["dom_tokens"].sum()
    util = total[..., 0] / (all_tokens * atlas.forward.TOP_K / atlas.forward.NUM_EXPERTS)
    mix = np.stack([(um["category"] == c).sum(-1) for c in range(len(CATEGORIES))], -1)  # [L, E, C]
    labels = []
    for layer in range(counts.shape[0]):
        row = []
        for e in range(counts.shape[1]):
            row.append(
                expert_label(layer, e, top[layer, e], top_p[layer, e], pal_share[layer, e], lift[layer, e], vocab)
            )
        labels.append(row)
    return dict(
        entropy=entropy,
        top=top,
        top_p=top_p,
        pal_share=pal_share,
        lift=lift,
        util=util,
        mix=mix,
        labels=labels,
        weight=z["expert_weight"] / np.maximum(total[..., 0], 1),
        bias_driven=z["expert_bias_driven"] / np.maximum(total[..., 0], 1),
    )


def show(s: str) -> str:
    """A token as label text: whitespace-only tokens escaped, a leading space shown as an open box."""
    if s.strip() == "":
        return repr(s)[1:-1].replace(" ", "\u2423")
    return "\u2423" + s[1:] if s.startswith(" ") else s


def expert_label(layer, e, top, top_p, pal_share, lift, vocab) -> str:
    g = int(np.argmax(np.where(lift >= 2.5, pal_share, -1)))
    domain = atlas.GROUPS[g] if lift[g] >= 2.5 and pal_share[g] >= 0.35 else ""
    picked, mass = [], 0.0
    for t, q in zip(top, top_p, strict=True):
        picked.append(vocab[int(t)])
        mass += q
        if mass >= 0.5 or len(picked) == 3:
            break
    words = ", ".join(f"'{show(s)}'" for s in picked)
    tail = f" ({mass:.0%})" if top_p[0] < 0.97 else ""
    head = f"{domain}: " if domain else ""
    return f"L{layer} E{e}: {head}{words}{tail}"


def cluster_layer(counts: np.ndarray, pal_share: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Average-linkage cosine clusters of the experts' sqrt token distributions (plus domain profile), and a 2-D
    PCA layout of the same features."""
    p = counts / np.maximum(counts.sum(-1, keepdims=True), 1)
    feat = np.concatenate([np.sqrt(p), 0.5 * np.sqrt(pal_share)], axis=1)
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-9)
    tree = linkage(feat, method="average", metric="cosine")
    labels = fcluster(tree, t=n_clusters, criterion="maxclust") - 1
    centred = feat - feat.mean(0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return labels, u[:, :2] * s[:2]


def main() -> None:
    data, bigram_dir, out_dir = sys.argv[1:4]
    z = dict(np.load(f"{data}/atlas.npz"))
    vocab = read_json(f"{data}/vocab.json")
    ok, clear = ngram_eligible(bigram_dir)
    um = unit_metrics(z, vocab, ok)
    um["ngram_clear"] = clear
    ep = expert_profiles(z, um, vocab)
    np.savez_compressed(f"{out_dir}/unit_metrics.npz", **{k: v for k, v in um.items()})
    np.savez_compressed(f"{out_dir}/topic_units.npz", **topic_tables(um, z))
    clusters = [
        cluster_layer(z["expert_token_ids"][l].astype(np.float64), ep["pal_share"][l], CLUSTERS_PER_LAYER)
        for l in range(6)
    ]
    np.savez_compressed(
        f"{out_dir}/expert_profiles.npz",
        **{k: v for k, v in ep.items() if k != "labels"},
        cluster=np.stack([c for c, _ in clusters]),
        layout=np.stack([xy for _, xy in clusters]),
    )
    write_json(ep["labels"], f"{out_dir}/expert_labels.json")
    counts = {c: [int((um["category"][l] == i).sum()) for l in range(6)] for i, c in enumerate(CATEGORIES)}
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
