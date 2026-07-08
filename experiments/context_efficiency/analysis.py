# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn the semantic episode labels into per-intervention token-budget uplift.

This is the pipeline's final output. It joins the labeled episodes to their PPS
sampling weights and the ground-truth budget anchor. Because sampling was PPS on
amplified carry cost, each labeled episode carries a ``weight_cost`` slice of the
tool budget, so a cost-weighted saved fraction estimates the population directly.

Per episode: ``saved_frac = clip(1 - substitute_size_ratio, 0, 1)`` when the
substitute is not "none" and is at least partially sufficient, else 0. The size
ratio already includes residual lookups + the substitute's own read, so saved_frac
is net. Results: uplift by intervention, realizability (automatic vs authored, with
authored wiki/memory gated by topic recurrence), wiki-topic yield per maintainable
cluster, Haiku-vs-stronger-model agreement, and a stronger-model-calibrated headline.
"""

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rigging.filesystem import StoragePath, prefix_join

from experiments.context_efficiency.accounting import P_READ, P_WRITE
from experiments.context_efficiency.schema import AUTHORED_REPO, AUTHORED_TOPIC, AUTOMATIC, SUBSTITUTES

logger = logging.getLogger(__name__)

# Always-on prelude carry rate: a token parked in the prefix is written once and
# re-read every subsequent turn. Median session ~54 turns.
MEDIAN_SESSION_TURNS = 54
PRELUDE_CARRY_PER_TOKEN = P_READ * MEDIAN_SESSION_TURNS + P_WRITE


def load_labels(label_dir: str) -> pd.DataFrame:
    """Load every ``*.json`` label file under a labeling step's output directory."""
    rows, bad = [], []
    for fp in sorted(str(p) for p in StoragePath(prefix_join(label_dir, "*.json")).glob()):
        try:
            d = json.loads(StoragePath(fp).read_text())
        except (json.JSONDecodeError, OSError):
            bad.append(fp)
            continue
        for lab in d.get("labels", []):
            if isinstance(lab, dict) and lab.get("episode_id"):
                rows.append(lab)
    if bad:
        logger.warning("%d unreadable label files under-count coverage: %s", len(bad), bad)
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


@dataclass(frozen=True)
class AnalysisConfig:
    episodes_path: str
    labels_path: str
    labels_val_path: str
    clusters_path: str
    budget_path: str
    output_path: str


def _uplift_by_intervention(df: pd.DataFrame, W: float, tool_pct: float) -> dict:
    """Per-substitute cost share and cost-weighted saved fraction of the tool surface."""
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
    return by_sub


def _assign_recurrence(df: pd.DataFrame, clusters_path: str) -> dict:
    """Add ``saved_cost`` and the recurrence columns in place, and return recurrence counts.

    A topic (raw slug) or cluster (maintainable doc) is *recurring* when it appears in ≥2
    distinct sessions — the bar an authored article must clear to amortize its authoring.
    The cluster mapping (from the clustering step) collapses fragmented slugs into the docs
    a team would actually keep, so recurrence is measured per doc, not per raw slug.
    """
    df["saved_cost"] = df.weight_cost * df.saved_frac
    topic_sessions = df[df.wiki_topic_slug.fillna("") != ""].groupby("wiki_topic_slug").session_id.nunique()
    recurring = set(topic_sessions[topic_sessions >= 2].index)
    df["topic_recurs"] = df.wiki_topic_slug.isin(recurring)

    cluster_path = prefix_join(clusters_path, "topic_clusters.json")
    df["cluster"] = df.wiki_topic_slug
    if StoragePath(cluster_path).exists():
        with StoragePath(cluster_path).open() as fh:
            cmap = {a["slug"]: a["cluster"] for a in json.load(fh).get("assignments", [])}
        df["cluster"] = df.wiki_topic_slug.map(lambda s: cmap.get(s, s))
    clus_sessions = df[df.wiki_topic_slug.fillna("") != ""].groupby("cluster").session_id.nunique()
    recurring_clusters = set(clus_sessions[clus_sessions >= 2].index)
    df["cluster_recurs"] = df.cluster.isin(recurring_clusters)
    return {
        "distinct_slugs": int(topic_sessions.shape[0]),
        "recurring_slugs": len(recurring),
        "distinct_clusters": int(clus_sessions.shape[0]),
        "recurring_clusters": len(recurring_clusters),
    }


def _realizability(df: pd.DataFrame, W: float, tool_pct: float, recurrence: dict) -> dict:
    """Saving by realizability class: automatic (no authoring), authored-per-repo, and
    authored-per-topic gated by recurrence (per-slug lower to per-cluster upper)."""

    def bpct(mask):
        return round(float(df[mask].saved_cost.sum() / W * tool_pct), 2)

    automatic_pct = bpct(df.best_substitute.isin(AUTOMATIC))
    authored_repo_pct = bpct(df.best_substitute.isin(AUTHORED_REPO))
    authored_topic_potential = bpct(df.best_substitute.isin(AUTHORED_TOPIC))
    authored_topic_realizable = bpct(df.best_substitute.isin(AUTHORED_TOPIC) & df.topic_recurs)
    authored_topic_cluster = bpct(df.best_substitute.isin(AUTHORED_TOPIC) & df.cluster_recurs)
    return {
        "automatic_index_tools_compaction_pct": automatic_pct,
        "authored_repo_map_docs_pct": authored_repo_pct,
        "authored_wiki_memory_potential_pct": authored_topic_potential,
        "authored_wiki_memory_recurrence_gated_pct_slug_lower": authored_topic_realizable,
        "authored_wiki_memory_recurrence_gated_pct_cluster_upper": authored_topic_cluster,
        "realizable_total_pct_of_budget": round(automatic_pct + authored_repo_pct + authored_topic_cluster, 2),
        "recurrence": recurrence,
        "note": (
            "AUTOMATIC (semantic index, better tools, result compaction) needs no authoring, so its "
            "per-episode saving is realizable. AUTHORED_REPO (repo map/docs) amortizes over every "
            "navigation episode in a repo. AUTHORED wiki/memory is gated by topic recurrence, reported "
            "as a range: per-slug (lower, over-fragmented) to per-doc-cluster (upper, after grouping "
            "slugs into the docs a team would actually maintain)."
        ),
    }


def _wiki_topic_yield(df: pd.DataFrame) -> dict:
    """Forestalled tokens per maintainable-doc cluster, ranked, with recurrence counts."""
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
    return {
        "n_distinct_topics": int(topic.shape[0]),
        "n_topics_recurring_multi_session": int((topic.sessions > 1).sum()),
        "top_by_forestalled_cost": top_topics,
        "note": (
            "forestalled cost assumes retrieval-gated delivery (article read only when relevant). "
            f"An always-on article of T tokens instead costs ~{PRELUDE_CARRY_PER_TOKEN:.0f}*T per session "
            "in prelude carry, so only recurring, multi-session topics clear the bar."
        ),
    }


def _validation_agreement(df: pd.DataFrame, val: pd.DataFrame) -> dict | None:
    """Bulk-labeler vs stronger-model agreement on the shared validation subset."""
    if val.empty:
        return None
    m = df[["episode_id", "best_substitute", "saved_frac"]].merge(
        val[["episode_id", "best_substitute", "saved_frac"]], on="episode_id", suffixes=("_h", "_s")
    )
    if not len(m):
        return None
    m["addr_h"] = m.best_substitute_h != "none"
    m["addr_s"] = m.best_substitute_s != "none"
    return {
        "n_compared": len(m),
        "addressable_agreement": round(float((m.addr_h == m.addr_s).mean()), 3),
        "exact_substitute_agreement": round(float((m.best_substitute_h == m.best_substitute_s).mean()), 3),
        "mean_saved_frac_haiku": round(float(m.saved_frac_h.mean()), 3),
        "mean_saved_frac_sonnet": round(float(m.saved_frac_s.mean()), 3),
    }


def _calibration(realizability: dict, total_saved_frac: float, tool_pct: float, agreement: dict | None) -> dict:
    """Scale the bulk headline by the stronger model's more conservative saved-fraction
    ratio; keep the bulk number as the optimistic bound. Neither is a measured replacement."""
    calib = 1.0
    if agreement and agreement["mean_saved_frac_haiku"] > 0:
        calib = agreement["mean_saved_frac_sonnet"] / agreement["mean_saved_frac_haiku"]
    automatic_pct = realizability["automatic_index_tools_compaction_pct"]
    realizable_total = realizability["realizable_total_pct_of_budget"]
    slug_lower = realizability["authored_wiki_memory_recurrence_gated_pct_slug_lower"]
    cluster_upper = realizability["authored_wiki_memory_recurrence_gated_pct_cluster_upper"]
    return {
        "sonnet_over_haiku_ratio": round(calib, 3),
        "per_episode_potential_pct_budget": {
            "haiku_optimistic": round(total_saved_frac * tool_pct, 2),
            "sonnet_calibrated": round(total_saved_frac * tool_pct * calib, 2),
        },
        "realizable_total_pct_budget": {
            "haiku_optimistic": realizable_total,
            "sonnet_calibrated": round(realizable_total * calib, 2),
        },
        "automatic_pct_budget": {
            "haiku_optimistic": automatic_pct,
            "sonnet_calibrated": round(automatic_pct * calib, 2),
        },
        "authored_wiki_memory_pct_budget": {
            "haiku_optimistic_slug_to_cluster": [slug_lower, cluster_upper],
            "sonnet_calibrated_slug_to_cluster": [round(slug_lower * calib, 2), round(cluster_upper * calib, 2)],
        },
    }


def _distribution(df: pd.DataFrame, W: float, col: str) -> dict:
    """Per-value episode count, cost share, and cost-weighted saved fraction."""
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


def run_analysis(cfg: AnalysisConfig) -> None:
    meta = pd.read_parquet(prefix_join(cfg.episodes_path, "episodes_sampled.parquet"))
    labels = load_labels(cfg.labels_path)
    if labels.empty:
        raise ValueError(f"no labels found under {cfg.labels_path} — run the labeling step first")
    df = meta.merge(labels, on="episode_id", how="inner")
    coverage = len(df) / len(meta)

    with StoragePath(prefix_join(cfg.budget_path, "budget_decomposition.json")).open() as fh:
        bd = json.load(fh)
    budget = bd["observed_budget_input_equiv"]
    tool_pct = bd["tool_addressable_surface"]["tool_addressable_pct_of_budget_lower_bound"]

    W = df.weight_cost.sum()  # total tool cost represented by the labeled sample
    total_saved_frac = float((df.weight_cost * df.saved_frac).sum() / W)

    recurrence = _assign_recurrence(df, cfg.clusters_path)
    realizability = _realizability(df, W, tool_pct, recurrence)
    agreement = _validation_agreement(df, load_labels(cfg.labels_val_path))
    calibrated = _calibration(realizability, total_saved_frac, tool_pct, agreement)

    result = {
        "sample": {
            "n_labeled": len(df),
            "n_sampled": len(meta),
            "label_coverage": round(coverage, 3),
            "tool_surface_pct_of_budget": tool_pct,
            "budget_input_equiv": budget,
        },
        "uplift_by_intervention": _uplift_by_intervention(df, W, tool_pct),
        "headline": {
            "irreducible_none_share_of_tool_surface": round(
                float(df[df.best_substitute == "none"].weight_cost.sum() / W), 3
            ),
            "per_episode_potential_saved_frac_of_tool_surface": round(total_saved_frac, 4),
            "per_episode_potential_saved_pct_of_budget": round(total_saved_frac * tool_pct, 2),
        },
        "realizability": realizability,
        "calibrated_headline": calibrated,
        "wiki_topics": _wiki_topic_yield(df),
        "by_intent_category": _distribution(df, W, "intent_category"),
        "by_answer_kind": _distribution(df, W, "answer_kind"),
        "validation_haiku_vs_sonnet": agreement,
    }
    StoragePath(cfg.output_path).mkdirs()
    with StoragePath(prefix_join(cfg.output_path, "semantic_analysis.json")).open("w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "analysis: %d labeled (%.0f%% coverage); realizable %.2f%% (calibrated %.2f%%)",
        len(df),
        100 * coverage,
        realizability["realizable_total_pct_of_budget"],
        calibrated["realizable_total_pct_budget"]["sonnet_calibrated"],
    )
