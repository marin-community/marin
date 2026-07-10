# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a self-contained quality-score debugging report (single-page HTML app).

Reads ``score.py``'s two per-source outputs directly: ``outputs/main/`` (lean
``source``/``id``/``score``/``quality_bucket`` -> the distribution + per-source stats)
and ``outputs/samples/`` (the systematic sample *with text* -> the spot-check docs, no
separate text fetch needed). The output is one standalone ``.html`` file -- all CSS/JS
inlined and the data embedded as JSON, so it works offline and can be hosted anywhere --
showing:

  - the score distribution (histogram + fixed-0.2 bucket bars)
  - per-domain bucket mix + means
  - a sortable per-source table with anomaly flags (``uninformative`` = near-constant
    score, the variance-gate case; ``homogeneous`` = spread but one dominant bucket)
  - an interactive spot-check drawer sampling docs per source x bucket (text fetched
    from the sample)

Usage::

    python -m experiments.datakit.cluster.quality.fast_transformer.ops.report \\
        --scored s3://.../quality/scored_1t \\
        --out report.html --scorer "pooled_junkgate2 (bme)" --sample "sample_1t"
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath

_TEMPLATE = Path(__file__).with_name("report_template.html")

NB = 5  # buckets 0..4
HBINS = 25  # score histogram bins
SAMPLE_PER_CELL = 5  # docs shown per (source, bucket) in the spot-check
PER_SOURCE = 500  # scored rows read per source (bounds memory + gives representative stats)
READ_BATCH = 4096
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


def domain_of(source: str) -> str:
    s = source.lower()
    for domain, pats in _DOMAIN_RULES:
        if any(p in s for p in pats):
            return domain
    return "other"


def _hist(scores: np.ndarray, bins: int = HBINS) -> list[int]:
    h, _ = np.histogram(np.clip(scores, 0, 1), bins=bins, range=(0, 1))
    return h.tolist()


def _bucket_counts(buckets: np.ndarray) -> list[int]:
    return [int((buckets == b).sum()) for b in range(NB)]


def _anomaly_flags(n: int, ft_mix: list[int], ft_scores: np.ndarray) -> list[tuple[str, str]]:
    """`uninformative` = near-constant score (FT can't discriminate -> variance gate);
    `homogeneous` = spread exists but clusters in one bucket (source is uniform quality)."""
    std = float(np.std(ft_scores))
    frac = np.array(ft_mix) / max(n, 1)
    if std < UNINFORMATIVE_STD:
        return [("uninformative", f"std={std:.03f}")]
    if frac.max() > HOMOGENEOUS_FRAC:
        return [("homogeneous", f"q{int(frac.argmax())} {frac.max():.0%}")]
    return []


def read_scored(scored_prefix: str, per_source: int) -> dict[str, list]:
    """Read the per-source scored parquets under ``scored_prefix``, keeping up to
    ``per_source`` rows per source (source is the dir name, used to skip capped sources)."""
    prefix = scored_prefix.rstrip("/")
    out: dict[str, list] = {k: [] for k in ("source", "id", "score", "quality_bucket")}
    per: dict[str, int] = defaultdict(int)
    for f in sorted(str(m) for m in StoragePath(f"{prefix}/**/outputs/main/**/*.parquet").glob()):
        src_dir = f.split(prefix + "/", 1)[1].split("/outputs/main/", 1)[0]
        if per[src_dir] >= per_source:
            continue
        with StoragePath(f).open("rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(
                batch_size=READ_BATCH, columns=["source", "id", "score", "quality_bucket"]
            ):
                d = batch.to_pydict()
                for s, i, sc, qb in zip(d["source"], d["id"], d["score"], d["quality_bucket"], strict=True):
                    if per[s] >= per_source:
                        continue
                    per[s] += 1
                    out["source"].append(s)
                    out["id"].append(str(i))
                    out["score"].append(float(sc))
                    out["quality_bucket"].append(int(qb))
    return out


def read_samples(scored_prefix: str, per_cell: int) -> list[dict]:
    """Read the samples side output (``source``/``id``/``score``/``quality_bucket``/``text``)
    that ``score.py`` writes to ``outputs/samples/``, keeping up to ``per_cell`` docs per
    (source, bucket) for the spot-check. No separate text fetch is needed."""
    prefix = scored_prefix.rstrip("/")
    kept: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for f in sorted(str(m) for m in StoragePath(f"{prefix}/**/outputs/samples/**/*.parquet").glob()):
        with StoragePath(f).open("rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(
                batch_size=READ_BATCH, columns=["source", "id", "score", "quality_bucket", "text"]
            ):
                d = batch.to_pydict()
                for s, i, sc, qb, t in zip(
                    d["source"], d["id"], d["score"], d["quality_bucket"], d["text"], strict=True
                ):
                    cell = (s, int(qb))
                    if len(kept[cell]) >= per_cell:
                        continue
                    kept[cell].append(
                        {
                            "source": s,
                            "domain": domain_of(s),
                            "id": str(i),
                            "ft": round(float(sc), 3),
                            "ft_bucket": int(qb),
                            "old_bucket": -1,
                            "old": None,
                            "text": (t or "")[:TEXT_CHARS],
                        }
                    )
    return [doc for docs in kept.values() for doc in docs]


def build_report_data(scored: dict[str, list], samples: list[dict], *, scorer: str, sample: str) -> dict:
    """Aggregate the lean scored rows (distribution + per-source) and the samples side
    output (spot-check docs with text) into the report payload."""
    src = np.array(scored["source"])
    fts = np.array(scored["score"], float)
    ftb = np.array(scored["quality_bucket"], int)
    dom = np.array([domain_of(s) for s in src])
    n = len(src)

    overall = {
        "n": n,
        "has_old": False,
        "ft_hist": _hist(fts),
        "old_hist": [],
        "ft_buckets": _bucket_counts(ftb),
        "old_buckets": [],
        "ft_mean": float(fts.mean()),
        "old_mean": 0.0,
        "ft_std": float(fts.std()),
        "old_std": 0.0,
    }

    domains = {}
    for d in sorted(set(dom)):
        m = dom == d
        domains[d] = {
            "n": int(m.sum()),
            "ft_mean": float(fts[m].mean()),
            "old_mean": 0.0,
            "ft_buckets": _bucket_counts(ftb[m]),
            "old_buckets": [],
            "ft_hist": _hist(fts[m]),
        }

    sources = []
    for s in sorted(set(src)):
        m = src == s
        mix = _bucket_counts(ftb[m])
        sources.append(
            {
                "source": s,
                "domain": domain_of(s),
                "n": int(m.sum()),
                "ft_mean": float(fts[m].mean()),
                "old_mean": 0.0,
                "ft_buckets": mix,
                "flags": _anomaly_flags(int(m.sum()), mix, fts[m]),
            }
        )
    sources.sort(key=lambda r: r["ft_mean"])

    # the samples side output IS the spot-check pool (already carries text)
    docs = samples

    return {
        "meta": {
            "scorer": scorer,
            "baseline": "",
            "sample": sample,
            "n": n,
            "nsrc": len(sources),
            "ndom": len(domains),
            "has_old": False,
        },
        "overall": overall,
        "conf": [],
        "domains": domains,
        "sources": sources,
        "docs": docs,
    }


def render_html(data: dict, *, title: str) -> str:
    blob = json.dumps(data).replace("</", "<\\/")
    return _TEMPLATE.read_text().replace("__DATA__", blob).replace("__TITLE__", html.escape(title))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored", required=True, help="score.py output prefix (per-source outputs/main + outputs/samples)")
    p.add_argument("--out", required=True, help="output .html path (local or storage url)")
    p.add_argument("--per-source", type=int, default=PER_SOURCE, help="scored rows read per source (from outputs/main)")
    p.add_argument("--title", default="Quality-Score Debugging Report")
    p.add_argument("--scorer", default="pooled fast-transformer (calibrated)")
    p.add_argument("--sample", default="")
    args = p.parse_args()

    scored = read_scored(args.scored, args.per_source)  # outputs/main -> distribution + per-source
    samples = read_samples(args.scored, SAMPLE_PER_CELL)  # outputs/samples -> spot-check docs with text
    data = build_report_data(scored, samples, scorer=args.scorer, sample=args.sample)
    doc = render_html(data, title=args.title)
    with StoragePath(args.out).open("w") as fh:
        fh.write(doc)
    print(f"wrote {args.out}  ({data['meta']['n']} docs, {data['meta']['nsrc']} sources, {len(data['docs'])} samples)")


if __name__ == "__main__":
    main()
