# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for the pooled fast-transformer quality scores.

Aggregates the per-source :class:`QualityScores` artifacts into one page: the
score distribution (histogram + fixed-0.2 bucket bars), per-domain bucket mix,
a sortable per-source table with anomaly flags (``uninformative`` = near-constant
score, the variance-gate case; ``homogeneous`` = spread but one dominant bucket),
and an interactive spot-check drawer over the samples side output (which already
carries text, so no separate fetch). Reads are bounded per source.
"""

from collections import defaultdict

import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, QualityScores
from experiments.datakit.cluster.quality.fast_transformer.score import SAMPLE_PCT
from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

NB = len(BUCKET_EDGES) + 1
HBINS = 25  # score histogram bins
MAIN_ROWS_PER_SOURCE = 500  # scored rows read per source (bounds memory + gives representative stats)
SAMPLE_ROWS_PER_SOURCE = 60  # rows read from the samples side output per source
SAMPLE_PER_CELL = 5  # docs shown per (source, bucket) in the spot-check
TEXT_CHARS = 1600
UNINFORMATIVE_STD = 0.03  # below this within-source std the FT can't discriminate (variance gate)
HOMOGENEOUS_FRAC = 0.9  # one bucket holding >= this share = genuinely uniform source

# Heuristic content-domain for a source, checked in order (first match wins).
_DOMAIN_RULES = [
    ("multilingual", ("finepdfs/", "translated", "climblab-ja", "cmn_", "_translated")),
    ("math", ("math", "arxiv", "numina")),
    ("code", ("starcoder2/", "code", "stackv2", "github", "swe-", "coderforge", "svg", "kaggle", "transpilation")),
    ("formal", ("formal_logic", "unconditional_algorithmic", "infinibyte_reasoning", "superior-reasoning", "rqa")),
    ("wiki", ("wikiteam", "wiki_rewrite", "wikipedia")),
    ("web", ("nemotron_cc_v2", "refuseweb", "safeweb", "diverse_qa")),
    ("safety", ("safety_pt", "moral_education")),
    (
        "reference",
        (
            "cp/",
            "libretexts",
            "doab",
            "caselaw",
            "nsf_awards",
            "library_of_congress",
            "biodiversity",
            "foodista",
            "data_provenance",
            "economics",
        ),
    ),
    (
        "reasoning",
        (
            "sft",
            "student_teacher",
            "question_answering",
            "multiple_choice",
            "stem",
            "synthetic-1",
            "scientific_coding",
            "rewriting",
            "code_review",
            "concepts",
        ),
    ),
]


def _domain_of(source: str) -> str:
    s = source.lower()
    for domain, pats in _DOMAIN_RULES:
        if any(p in s for p in pats):
            return domain
    return "other"


def _hist(scores: np.ndarray) -> list[int]:
    h, _ = np.histogram(np.clip(scores, 0, 1), bins=HBINS, range=(0, 1))
    return h.tolist()


def _bucket_counts(buckets: np.ndarray) -> list[int]:
    return [int((buckets == b).sum()) for b in range(NB)]


def _anomaly_flags(mix: list[int], scores: np.ndarray) -> list[tuple[str, str]]:
    """`uninformative` = near-constant score (FT can't discriminate -> variance gate);
    `homogeneous` = spread exists but clusters in one bucket (source is uniform quality)."""
    std = float(np.std(scores))
    frac = np.array(mix) / len(scores)
    if std < UNINFORMATIVE_STD:
        return [("uninformative", f"std={std:.03f}")]
    if frac.max() > HOMOGENEOUS_FRAC:
        return [("homogeneous", f"q{int(frac.argmax())} {frac.max():.0%}")]
    return []


def _spot_check_docs(sources: dict[str, QualityScores]) -> list[dict]:
    """Bounded read of each source's samples side output (which carries text),
    keeping up to SAMPLE_PER_CELL docs per (source, bucket)."""
    kept: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for qs in sources.values():
        rows = sample_rows(
            qs.samples_output_dir, ["source", "id", "score", "quality_bucket", "text"], SAMPLE_ROWS_PER_SOURCE
        )
        for r in rows:
            cell = (r["source"], int(r["quality_bucket"]))
            if len(kept[cell]) >= SAMPLE_PER_CELL:
                continue
            kept[cell].append(
                {
                    "source": r["source"],
                    "domain": _domain_of(r["source"]),
                    "id": str(r["id"]),
                    "ft": round(float(r["score"]), 3),
                    "ft_bucket": int(r["quality_bucket"]),
                    "text": r["text"][:TEXT_CHARS],
                }
            )
    return [doc for docs in kept.values() for doc in docs]


def _build_report_data(rows: list[dict], docs: list[dict], *, scorer: str) -> dict:
    """Aggregate the lean scored rows (distribution + per-source + per-domain) and the
    spot-check docs into the template payload."""
    src = np.array([r["source"] for r in rows])
    fts = np.array([r["score"] for r in rows], float)
    ftb = np.array([r["quality_bucket"] for r in rows], int)
    dom = np.array([_domain_of(s) for s in src])
    n = len(rows)

    overall = {
        "n": n,
        "ft_hist": _hist(fts),
        "ft_buckets": _bucket_counts(ftb),
        "ft_mean": float(fts.mean()),
        "ft_std": float(fts.std()),
    }

    domains = {}
    for d in sorted(set(dom)):
        m = dom == d
        domains[d] = {
            "n": int(m.sum()),
            "ft_mean": float(fts[m].mean()),
            "ft_buckets": _bucket_counts(ftb[m]),
            "ft_hist": _hist(fts[m]),
        }

    per_source = []
    for s in sorted(set(src)):
        m = src == s
        mix = _bucket_counts(ftb[m])
        per_source.append(
            {
                "source": str(s),
                "domain": _domain_of(s),
                "n": int(m.sum()),
                "ft_mean": float(fts[m].mean()),
                "ft_buckets": mix,
                "flags": _anomaly_flags(mix, fts[m]),
            }
        )
    per_source.sort(key=lambda r: r["ft_mean"])

    return {
        "meta": {
            "scorer": scorer,
            "n": n,
            "nsrc": len(per_source),
            "ndom": len(domains),
            "sampling": (
                f"distribution + per-source stats from the first {MAIN_ROWS_PER_SOURCE} scored rows "
                f"per source (file order); spot-check docs from the scorer's ~{SAMPLE_PCT:.0%} systematic "
                f"sample side output (≤{SAMPLE_PER_CELL} per source x bucket)"
            ),
        },
        "overall": overall,
        "domains": domains,
        "sources": per_source,
        "docs": docs,
    }


def quality_report(output_path: str, sources: dict[str, QualityScores]) -> StageReport:
    scorers = {qs.model_dir for qs in sources.values()}
    assert len(scorers) == 1, f"mixed scorer model dirs across sources: {scorers}"
    (scorer,) = scorers

    rows: list[dict] = []
    for qs in sources.values():
        rows.extend(sample_rows(qs.main_output_dir, ["source", "id", "score", "quality_bucket"], MAIN_ROWS_PER_SOURCE))
    docs = _spot_check_docs(sources)
    data = _build_report_data(rows, docs, scorer=scorer)

    o = data["overall"]
    stats = {
        "total_scored": sum(qs.counters.get("ft_quality/scored", 0) for qs in sources.values()),
        "total_sampled": sum(qs.counters.get("ft_quality/sampled", 0) for qs in sources.values()),
        "docs_sampled": o["n"],
        "n_sources": data["meta"]["nsrc"],
        "n_domains": data["meta"]["ndom"],
        "score_mean": o["ft_mean"],
        "score_std": o["ft_std"],
        "q0_share": o["ft_buckets"][0] / o["n"],
        "q4_share": o["ft_buckets"][4] / o["n"],
        "spot_check_docs": len(docs),
        "flagged_sources": sum(1 for r in data["sources"] if r["flags"]),
    }
    page = render_template("quality.html", title="Datakit quality", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
