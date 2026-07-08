#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""Turn the semantic episode labels into per-intervention token-budget uplift.

Consumes the labeled episodes (`_data/labels/*.json`) joined to their PPS sampling
weights (`episodes_sampled.parquet`) and the ground-truth budget anchor
(`budget_decomposition.json`). Because sampling was PPS on amplified carry cost,
each labeled episode carries a `weight_cost` slice of the tool budget, so a
cost-weighted saved fraction estimates the population directly.

Per episode: `saved_frac = clip(1 - substitute_size_ratio, 0, 1)` when the
substitute is not "none" and is at least partially sufficient, else 0. The size
ratio already includes the residual lookups + the substitute's own read, so
saved_frac is net. Results:

- Uplift by intervention (best_substitute), as a fraction of the tool surface and
  of the whole budget (via the ground-truth tool-addressable anchor).
- Wiki-topic yield: episodes clustered on the proposed `wiki_topic_slug`, with the
  forestalled tokens per topic vs the article's own carried cost — the concrete
  "is a one-paragraph article worth more than the greps it replaces" test.
- haiku vs sonnet agreement on the validation batches.
- Breakdowns by intent_category and answer_kind.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# Always-on prelude carry rate: a token parked in the prefix is written once and
# re-read every subsequent turn. Median session ~54 turns => ~0.10*54 + 1.25.
PRELUDE_CARRY_PER_TOKEN = 0.10 * 54 + 1.25
SUBSTITUTES = [
    "none", "shared-wiki", "semantic-code-index", "persistent-memory",
    "better-tool-or-flag", "result-compaction", "repo-map-or-docs",
]  # fmt: skip

# Realizability of each substitute. AUTOMATIC mechanisms are generated from the
# code/tooling with no per-item authoring, so their per-episode saving is
# realizable as-is. AUTHORED_REPO artifacts (repo map / structured docs) are one
# artifact per repo, amortized across every navigation episode in that repo.
# AUTHORED_TOPIC artifacts (wiki article, memory entry) are one artifact per fact,
# so their saving is gated by how often the topic RECURS across sessions.
AUTOMATIC = {"semantic-code-index", "better-tool-or-flag", "result-compaction"}
AUTHORED_REPO = {"repo-map-or-docs"}
AUTHORED_TOPIC = {"shared-wiki", "persistent-memory"}


def load_labels(label_dir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(label_dir, "*.json"))):
        try:
            with open(fp) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        for L in d.get("labels", []):
            if isinstance(L, dict) and L.get("episode_id"):
                rows.append(L)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("episode_id")
    df["substitute_size_ratio"] = pd.to_numeric(df.get("substitute_size_ratio"), errors="coerce").clip(0, 1).fillna(1.0)
    df["best_substitute"] = df.get("best_substitute", "none").fillna("none")
    df["substitute_sufficient"] = df.get("substitute_sufficient", "no").fillna("no")
    addressable = (df.best_substitute != "none") & (df.substitute_sufficient != "no")
    df["saved_frac"] = np.where(addressable, (1 - df.substitute_size_ratio).clip(lower=0), 0.0)
    return df


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--data", default=os.path.join(here, "_data"))
    ap.add_argument("--out", default=os.path.join(here, "_data", "semantic_analysis.json"))
    args = ap.parse_args()

    meta = pd.read_parquet(os.path.join(args.data, "episodes_sampled.parquet"))
    labels = load_labels(os.path.join(args.data, "labels"))
    if labels.empty:
        raise SystemExit("no labels found yet — run the labeling workflow first")
    df = meta.merge(labels, on="episode_id", how="inner")
    coverage = len(df) / len(meta)

    with open(os.path.join(args.data, "budget_decomposition.json")) as fh:
        bd = json.load(fh)
    budget = bd["observed_budget_input_equiv"]
    tool_pct = bd["tool_addressable_surface"]["tool_addressable_pct_of_budget_lower_bound"]

    W = df.weight_cost.sum()  # total tool cost represented by the labeled sample
    # fraction of the tool surface each intervention can save (cost-weighted)
    by_sub = {}
    for s in SUBSTITUTES:
        sub = df[df.best_substitute == s]
        frac = (sub.weight_cost * sub.saved_frac).sum() / W
        by_sub[s] = {
            "n_episodes": len(sub),
            "cost_share_of_tool_surface": round(float(sub.weight_cost.sum() / W), 3),
            "saved_frac_of_tool_surface": round(float(frac), 4),
            "saved_pct_of_budget": round(float(frac * tool_pct), 2),
            "mean_size_ratio": round(float(sub.substitute_size_ratio.mean()), 2) if len(sub) else None,
        }
    total_saved_frac = float((df.weight_cost * df.saved_frac).sum() / W)

    # realizability: not every "a wiki would help here" is realizable, because the
    # SAME article must recur across sessions to amortize its authoring.
    df["saved_cost"] = df.weight_cost * df.saved_frac
    topic_sessions = df[df.wiki_topic_slug.fillna("") != ""].groupby("wiki_topic_slug").session_id.nunique()
    recurring = set(topic_sessions[topic_sessions >= 2].index)
    df["topic_recurs"] = df.wiki_topic_slug.isin(recurring)

    # cluster the fragmented slugs into maintainable-doc topics (from a Sonnet pass)
    # so recurrence is measured per doc a team would keep, not per raw slug label.
    cluster_path = os.path.join(args.data, "topic_clusters.json")
    df["cluster"] = df.wiki_topic_slug
    if os.path.exists(cluster_path):
        with open(cluster_path) as fh:
            cmap = {a["slug"]: a["cluster"] for a in json.load(fh).get("assignments", [])}
        df["cluster"] = df.wiki_topic_slug.map(lambda s: cmap.get(s, s))
    clus_sessions = df[df.wiki_topic_slug.fillna("") != ""].groupby("cluster").session_id.nunique()
    recurring_clusters = set(clus_sessions[clus_sessions >= 2].index)
    df["cluster_recurs"] = df.cluster.isin(recurring_clusters)

    def bpct(mask):
        return round(float(df[mask].saved_cost.sum() / W * tool_pct), 2)

    automatic_pct = bpct(df.best_substitute.isin(AUTOMATIC))
    authored_repo_pct = bpct(df.best_substitute.isin(AUTHORED_REPO))
    authored_topic_potential = bpct(df.best_substitute.isin(AUTHORED_TOPIC))
    authored_topic_realizable = bpct(df.best_substitute.isin(AUTHORED_TOPIC) & df.topic_recurs)
    authored_topic_cluster = bpct(df.best_substitute.isin(AUTHORED_TOPIC) & df.cluster_recurs)
    realizability = {
        "automatic_index_tools_compaction_pct": automatic_pct,
        "authored_repo_map_docs_pct": authored_repo_pct,
        "authored_wiki_memory_potential_pct": authored_topic_potential,
        "authored_wiki_memory_recurrence_gated_pct_slug_lower": authored_topic_realizable,
        "authored_wiki_memory_recurrence_gated_pct_cluster_upper": authored_topic_cluster,
        "realizable_total_pct_of_budget": round(automatic_pct + authored_repo_pct + authored_topic_cluster, 2),
        "recurrence": {
            "distinct_slugs": int(topic_sessions.shape[0]),
            "recurring_slugs": len(recurring),
            "distinct_clusters": int(clus_sessions.shape[0]),
            "recurring_clusters": len(recurring_clusters),
        },
        "note": (
            "AUTOMATIC (semantic index, better tools, result compaction) needs no authoring, so its "
            "per-episode saving is realizable. AUTHORED_REPO (repo map/docs) amortizes over every "
            "navigation episode in a repo. AUTHORED wiki/memory is gated by topic recurrence, reported "
            "as a range: per-slug (lower, over-fragmented) to per-doc-cluster (upper, after grouping "
            "slugs into the docs a team would actually maintain)."
        ),
    }

    # wiki-topic yield: forestalled tokens per maintainable-doc cluster vs recurrence
    wt = df[(df.wiki_topic_slug.fillna("") != "") & (df.saved_frac > 0)].copy()
    topic = (
        wt.assign(saved=wt.weight_cost * wt.saved_frac)
        .groupby("cluster")
        .agg(
            episodes=("episode_id", "size"),
            sessions=("session_id", "nunique"),
            forestalled_cost=("saved", "sum"),
            repos=("repo", "nunique"),
        )
        .sort_values("forestalled_cost", ascending=False)
    )
    top_topics = [
        {
            "cluster": name,
            "episodes": int(r.episodes),
            "sessions": int(r.sessions),
            "repos": int(r.repos),
            "forestalled_input_equiv": int(r.forestalled_cost),
        }
        for name, r in topic.head(30).iterrows()
    ]
    n_topics = int(topic.shape[0])
    n_topics_multi = int((topic.sessions > 1).sum())

    # validation: haiku vs sonnet on shared batches
    val = load_labels(os.path.join(args.data, "labels_val"))
    agreement = None
    if not val.empty:
        m = df[["episode_id", "best_substitute", "saved_frac"]].merge(
            val[["episode_id", "best_substitute", "saved_frac"]], on="episode_id", suffixes=("_h", "_s")
        )
        if len(m):
            m["addr_h"] = m.best_substitute_h != "none"
            m["addr_s"] = m.best_substitute_s != "none"
            agreement = {
                "n_compared": len(m),
                "addressable_agreement": round(float((m.addr_h == m.addr_s).mean()), 3),
                "exact_substitute_agreement": round(float((m.best_substitute_h == m.best_substitute_s).mean()), 3),
                "mean_saved_frac_haiku": round(float(m.saved_frac_h.mean()), 3),
                "mean_saved_frac_sonnet": round(float(m.saved_frac_s.mean()), 3),
            }

    # Sonnet-calibration: Sonnet (a stronger judge) is systematically more
    # conservative than Haiku on the saved fraction. Scale the Haiku headline by
    # the Sonnet/Haiku ratio to get a conservative estimate; keep Haiku as the
    # optimistic bound. Neither is a measured replacement.
    calib = 1.0
    if agreement and agreement["mean_saved_frac_haiku"] > 0:
        calib = agreement["mean_saved_frac_sonnet"] / agreement["mean_saved_frac_haiku"]
    calibrated = {
        "sonnet_over_haiku_ratio": round(calib, 3),
        "per_episode_potential_pct_budget": {
            "haiku_optimistic": round(total_saved_frac * tool_pct, 2),
            "sonnet_calibrated": round(total_saved_frac * tool_pct * calib, 2),
        },
        "realizable_total_pct_budget": {
            "haiku_optimistic": realizability["realizable_total_pct_of_budget"],
            "sonnet_calibrated": round(realizability["realizable_total_pct_of_budget"] * calib, 2),
        },
        "automatic_pct_budget": {
            "haiku_optimistic": automatic_pct,
            "sonnet_calibrated": round(automatic_pct * calib, 2),
        },
        "authored_wiki_memory_pct_budget": {
            "haiku_optimistic_slug_to_cluster": [authored_topic_realizable, authored_topic_cluster],
            "sonnet_calibrated_slug_to_cluster": [
                round(authored_topic_realizable * calib, 2),
                round(authored_topic_cluster * calib, 2),
            ],
        },
    }

    def dist(col):
        g = df.groupby(col).apply(
            lambda d: pd.Series(
                {
                    "n": len(d),
                    "cost_share": round(float(d.weight_cost.sum() / W), 3),
                    "saved_frac": round(float((d.weight_cost * d.saved_frac).sum() / d.weight_cost.sum()), 3),
                }
            ),
            include_groups=False,
        )
        return {str(k): {"n": int(v.n), "cost_share": v.cost_share, "saved_frac": v.saved_frac} for k, v in g.iterrows()}

    result = {
        "sample": {
            "n_labeled": len(df),
            "n_sampled": len(meta),
            "label_coverage": round(coverage, 3),
            "tool_surface_pct_of_budget": tool_pct,
            "budget_input_equiv": budget,
        },
        "uplift_by_intervention": by_sub,
        "headline": {
            "irreducible_none_share_of_tool_surface": round(
                float(df[df.best_substitute == "none"].weight_cost.sum() / W), 3
            ),
            "per_episode_potential_saved_frac_of_tool_surface": round(total_saved_frac, 4),
            "per_episode_potential_saved_pct_of_budget": round(total_saved_frac * tool_pct, 2),
        },
        "realizability": realizability,
        "calibrated_headline": calibrated,
        "wiki_topics": {
            "n_distinct_topics": n_topics,
            "n_topics_recurring_multi_session": n_topics_multi,
            "top_by_forestalled_cost": top_topics,
            "note": (
                "forestalled cost assumes retrieval-gated delivery (article read only when relevant). "
                f"An always-on article of T tokens instead costs ~{PRELUDE_CARRY_PER_TOKEN:.0f}*T per session "
                "in prelude carry, so only recurring, multi-session topics clear the bar."
            ),
        },
        "by_intent_category": dist("intent_category"),
        "by_answer_kind": dist("answer_kind"),
        "validation_haiku_vs_sonnet": agreement,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
