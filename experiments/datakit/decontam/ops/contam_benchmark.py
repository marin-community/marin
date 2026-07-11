# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Precision/recall benchmark for decon on artificial data with known labels (marin#6852).

A self-contained pipeline: generate a labeled corpus → run the *real* decon
(`decon_to_parquet`) → score precision / recall / F1 against the ground truth.

Runs two arms on the same corpus and reports them side by side:

* A — core decon, no common-ngram filter (the baseline precision/recall).
* B — with the DF filter: a drop-set built from the benchmark corpus itself
  (`build_source_drop_set`) excludes ngrams common across the corpus. Boilerplate
  hard negatives (a famous quote, an instruction template) appear in enough docs
  to clear the DF cutoff and get dropped → precision rises; distinctive injected
  eval items appear too rarely to be dropped → recall is unchanged.

Doc families (each labeled contaminated or clean):

* POSITIVE — a real eval item injected into filler in four forms (verbatim,
  its own paragraph, short-line-wrapped, inline-embedded). Contaminated → recall.
* EASY_NEGATIVE — filler only, no eval overlap. Clean.
* HARD_NEGATIVE — a doc that *shares non-distinctive text* with the eval suite
  (a legal enacting clause, a famous quote, a truth-table instruction template,
  a standard math identity) but is not a leak. Clean → precision. These are the
  false-positive families catalogued in marin#6852; the benchmark measures how
  often the algorithm mistakes them for contamination.

Reports headline P/R/F1 at the 0.5 overlap threshold, a per-mechanism breakdown
(recall per positive form, false-positive rate per hard-negative family), and a
PR curve over overlap thresholds (from each doc's ``max_overlap``).

    uv run iris --cluster=cw-rno2a job run --cpu 2 --memory 4GB --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/contam_benchmark.py \\
           --work s3://marin-us-east-02a/marin/user/rav/decon_benchmark \\
           --out s3://marin-us-east-02a/marin/user/rav/decon_viewer/benchmark.json
"""

import argparse
import json
import logging
import random
from collections import defaultdict

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.decon import (
    NGramConfig,
    _bloom_hash,
    _discover_eval_files,
    _extract_features,
    bloom_paths,
    build_eval_bloom,
    build_source_drop_set,
    decon_to_parquet,
)
from marin.datakit.normalize import NormalizedData
from rigging.filesystem import StoragePath, marin_prefix, url_to_fs
from zephyr.readers import load_file

from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS
from experiments.datakit.testbed.decon_arm import (
    DF_COMMON_FRAC,
    DF_COMMON_MIN_ABS,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
    PARAGRAPH_DELIMITER,
)

logger = logging.getLogger(__name__)
_EVALS_RELATIVE = "datakit/decontam/evals"

# Generic prose that shares no eval n-grams — the padding around injected items.
# Bloom-hitting sentences are dropped before use so filler never self-flags.
_FILLER_POOL = [
    "The afternoon market sold ripe tomatoes and fresh basil to a handful of unhurried regulars.",
    "Rain tapped the tin roof while the kettle warmed and the cat stretched across the windowsill.",
    "She repainted the fence a pale green over the weekend and planted marigolds along the walk.",
    "The commuter train was six minutes late, which gave everyone time to finish their coffee.",
    "A gentle breeze carried the smell of cut grass across the empty football pitch at dusk.",
    "He sorted the mismatched socks into a shoebox and labelled it with a faded blue marker.",
    "The library's reading room hummed with the quiet turning of pages and the occasional cough.",
    "Two gulls argued over a chip near the harbour wall as the ferry idled at the dock.",
    "Grandmother's recipe called for a pinch of nutmeg and an hour of patience by the stove.",
    "The bicycle's front tyre had gone soft again, so he walked it the last block home.",
    "Streetlights flickered on one by one as the shopkeepers rolled down their metal shutters.",
    "A toddler chased pigeons across the plaza while her father photographed the old clock tower.",
    "The hiking club rescheduled the ridge walk after the forecast promised an afternoon of hail.",
    "Fresh bread cooled on a rack behind the counter and the whole street smelled of yeast.",
    "The mechanic wiped his hands on a rag and said the alternator would last another winter.",
    "Autumn leaves clogged the gutter, so they spent Sunday morning up ladders with old buckets.",
    "The choir rehearsed the same eight bars until the tenor finally landed the high note.",
    "A single lamp burned in the lighthouse as the tide crept back over the mudflats.",
    "The bakery ran out of croissants by nine, which surprised no one who arrived at ten.",
    "He learned to whittle small birds from driftwood during the long quiet ferry crossings.",
]


def _sentence_ngrams_hit(text: str, bf: dupekit.Bloom, ngram: NGramConfig) -> bool:
    return any(_bloom_hash(feat) in bf for feat in _extract_features(text, ngram))


def _short_line_wrap(text: str, words_per_line: int = 8) -> str:
    words = text.split()
    return "\n".join(" ".join(words[i : i + words_per_line]) for i in range(0, len(words), words_per_line))


# Non-distinctive text shared with the eval suite — the FP families (marin#6852).
# Each string is verified to hit the bloom before use (else it can't test precision).
_HARD_NEGATIVES = {
    "enacting_clause": (
        "Be it enacted by the Senate and House of Representatives of the "
        "United States of America in Congress assembled, That this Act may be cited by its short title."
    ),
    "gettysburg": (
        "Four score and seven years ago our fathers brought forth on this continent a new nation, "
        "conceived in liberty, and dedicated to the proposition that all men are created equal."
    ),
    "truth_table_template": (
        "Construct a complete truth table for the following argument. Then, using the truth "
        "table, determine whether the argument is valid or invalid. If the argument is invalid, choose an option "
        "which presents a counterexample."
    ),
    "sum_of_cubes_identity": (
        "Recall the factoring identity x^3 + y^3 + z^3 - 3xyz = "
        "(x + y + z)(x^2 + y^2 + z^2 - xy - yz - zx), which shows up in many contest problems."
    ),
    # Verbatim recurring formula from hendrycks/minerva math evals (~20 items),
    # so it is guaranteed in the bloom yet non-distinctive across problems.
    "arithmetic_series_formula": (
        "The sum of an arithmetic series is equal to the average of the first and "
        "last term, multiplied by the number of terms, a fact worth memorizing before the exam."
    ),
}


def _build_filler(rng: random.Random, bf: dupekit.Bloom, ngram: NGramConfig) -> list[str]:
    pool = [s for s in _FILLER_POOL if not _sentence_ngrams_hit(s, bf, ngram)]
    dropped = len(_FILLER_POOL) - len(pool)
    if dropped:
        logger.warning("dropped %d filler sentences that hit the bloom", dropped)
    rng.shuffle(pool)
    return pool


def _filler_block(filler: list[str], rng: random.Random, n: int = 4) -> str:
    return "\n\n".join(rng.sample(filler, min(n, len(filler))))


def _generate(
    eval_files: list[str], bf: dupekit.Bloom, ngram: NGramConfig, n_positive: int, n_hard: int, n_easy: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Return (docs[{id,text,partition_id}], labels[{id,contaminated,mechanism,eval_id}])."""
    rng = random.Random(seed)
    filler = _build_filler(rng, bf, ngram)

    # Eval items with enough tokens to form a matchable n-gram, deduped.
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in eval_files:
        for rec in load_file(path):
            text = str(rec.get("text") or "")
            if len(text.split()) >= NGRAM_LENGTH and text not in seen and _sentence_ngrams_hit(text, bf, ngram):
                seen.add(text)
                items.append((str(rec.get("id") or f"{path}::{len(items)}"), text))
        if len(items) >= n_positive * 3:
            break
    if len(items) < n_positive:
        raise ValueError(f"only {len(items)} usable eval items found; need {n_positive}")
    picked = rng.sample(items, n_positive)

    docs: list[dict] = []
    labels: list[dict] = []

    def add(text: str, contaminated: bool, mechanism: str, eval_id: str | None) -> None:
        did = f"bench-{len(docs):06d}"
        docs.append({"id": did, "text": text, "partition_id": 0})
        labels.append({"id": did, "contaminated": contaminated, "mechanism": mechanism, "eval_id": eval_id or ""})

    forms = {
        "verbatim": lambda t: t,
        "own_paragraph": lambda t: f"{_filler_block(filler, rng)}\n\n{t}\n\n{_filler_block(filler, rng)}",
        "short_line": lambda t: f"{_filler_block(filler, rng)}\n\n{_short_line_wrap(t)}\n\n{_filler_block(filler, rng)}",
        "embedded": lambda t: f"{_filler_block(filler, rng)} {t.replace(chr(10), ' ')} {_filler_block(filler, rng)}",
    }
    for eid, text in picked:
        for form, fn in forms.items():
            add(fn(text), True, f"positive_{form}", eid)

    for name, boiler in _HARD_NEGATIVES.items():
        if not _sentence_ngrams_hit(boiler, bf, ngram):
            logger.warning("hard negative %s does not hit the bloom — it can't be a false positive here", name)
        for _ in range(n_hard):
            add(f"{_filler_block(filler, rng)}\n\n{boiler}\n\n{_filler_block(filler, rng)}", False, f"hard_{name}", None)

    for _ in range(n_easy):
        add(_filler_block(filler, rng, n=6), False, "easy_negative", None)

    logger.info(
        "generated %d docs (%d positive, %d hard-neg, %d easy-neg)",
        len(docs),
        n_positive * 4,
        n_hard * len(_HARD_NEGATIVES),
        n_easy,
    )
    return docs, labels


def _write_docs(docs: list[dict], out_dir: str) -> str:
    path = f"{out_dir.rstrip('/')}/docs/part-00000-of-00001.parquet"
    StoragePath(f"{out_dir.rstrip('/')}/docs").mkdirs()
    with StoragePath(path).open("wb") as fh:
        pq.write_table(pa.Table.from_pylist(docs), fh, compression="zstd")
    return f"{out_dir.rstrip('/')}/docs"


def _read_predictions(decon_out: str) -> dict[str, float]:
    """id -> max_overlap from the decon attributes (under outputs/main)."""
    fs, resolved = url_to_fs(f"{decon_out.rstrip('/')}/outputs/main")
    preds: dict[str, float] = {}
    for f in sorted(x for x in fs.find(resolved) if x.endswith(".parquet")):
        with fs.open(f, "rb") as fh:
            tbl = pq.read_table(fh, columns=["id", "attributes"])
        for did, a in zip(tbl.column("id").to_pylist(), tbl.column("attributes").to_pylist(), strict=True):
            preds[did] = (a or {}).get("max_overlap", 0.0) or 0.0
    return preds


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def _score(labels: list[dict], preds: dict[str, float], threshold: float) -> dict:
    def flagged(did: str, t: float) -> bool:
        return preds.get(did, 0.0) >= t and preds.get(did, 0.0) > 0

    tp = fp = fn = 0
    per_mech: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # mechanism -> [n, n_flagged]
    for lab in labels:
        pred = flagged(lab["id"], threshold)
        per_mech[lab["mechanism"]][0] += 1
        per_mech[lab["mechanism"]][1] += int(pred)
        if lab["contaminated"]:
            tp += int(pred)
            fn += int(not pred)
        else:
            fp += int(pred)

    mech_report = {}
    for mech, (n, nf) in sorted(per_mech.items()):
        is_pos = mech.startswith("positive_")
        mech_report[mech] = {
            "n": n,
            "flagged": nf,
            ("recall" if is_pos else "false_positive_rate"): round(nf / n, 4) if n else 0.0,
        }

    curve = []
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        ctp = sum(1 for x in labels if x["contaminated"] and flagged(x["id"], t))
        cfp = sum(1 for x in labels if not x["contaminated"] and flagged(x["id"], t))
        cfn = sum(1 for x in labels if x["contaminated"] and not flagged(x["id"], t))
        m = _prf(ctp, cfp, cfn)
        curve.append({"threshold": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})

    return {"overall": _prf(tp, fp, fn), "threshold": threshold, "by_mechanism": mech_report, "pr_curve": curve}


def _log_comparison(a: dict, b: dict, n_dropped: int, delimiter: str) -> None:
    """Print arm A (no filter) vs arm B (DF filter) side by side."""
    logger.info("=== decon contamination benchmark (delimiter=%r) ===", delimiter)
    logger.info("DF filter dropped %d common ngrams (built from the benchmark corpus)", n_dropped)
    oa, ob = a["overall"], b["overall"]
    logger.info(
        "OVERALL @%.2f   A(no filter): P=%.3f R=%.3f F1=%.3f (fp=%d)   B(DF filter): P=%.3f R=%.3f F1=%.3f (fp=%d)",
        OVERLAP_THRESHOLD,
        oa["precision"],
        oa["recall"],
        oa["f1"],
        oa["fp"],
        ob["precision"],
        ob["recall"],
        ob["f1"],
        ob["fp"],
    )
    logger.info("%-28s %13s %13s", "mechanism", "A flagged", "B flagged")
    for mech in a["by_mechanism"]:
        ra, rb = a["by_mechanism"][mech], b["by_mechanism"][mech]
        logger.info("  %-26s n=%-5d %6d %13d", mech, ra["n"], ra["flagged"], rb["flagged"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="scratch dir for bloom + docs + decon output")
    ap.add_argument("--out", required=True, help="metrics JSON path")
    ap.add_argument("--n-positive", type=int, default=300)
    ap.add_argument("--n-hard", type=int, default=40, help="docs per hard-negative family")
    ap.add_argument("--n-easy", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--delimiter",
        default=None,
        help=r"paragraph delimiter override (backslash-escaped, e.g. '\n' per-line or '\n\n' true-paragraph); "
        f"default is the production {PARAGRAPH_DELIMITER!r}",
    )
    args = ap.parse_args()
    delimiter = args.delimiter.encode().decode("unicode_escape") if args.delimiter else PARAGRAPH_DELIMITER
    ngram = NGramConfig(ngram_length=NGRAM_LENGTH, overlap_threshold=OVERLAP_THRESHOLD, paragraph_delimiter=delimiter)

    eval_root = f"{marin_prefix()}/{_EVALS_RELATIVE}"
    eval_files = list(_discover_eval_files([eval_root], DECON_EXCLUDED_EVAL_TASKS))

    bloom_dir = f"{args.work.rstrip('/')}/bloom"
    build_eval_bloom(
        eval_data_sources=[eval_root],
        output_path=bloom_dir,
        ngram=ngram,
        estimated_doc_count=50_000_000,
        false_positive_rate=1e-9,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )
    bloom_path, _ = bloom_paths(bloom_dir)
    bf = dupekit.Bloom.load_bytes(StoragePath(bloom_path).read_bytes())

    docs, labels = _generate(eval_files, bf, ngram, args.n_positive, args.n_hard, args.n_easy, args.seed)
    docs_dir = _write_docs(docs, args.work)
    norm = NormalizedData(main_output_dir=docs_dir, dup_output_dir="", counters={})

    # Arm A: core decon, no common-ngram filter.
    decon_to_parquet(
        normalized_data=norm, prebuilt_bloom_dir=bloom_dir, output_path=f"{args.work.rstrip('/')}/decon_a", ngram=ngram
    )
    metrics_a = _score(labels, _read_predictions(f"{args.work.rstrip('/')}/decon_a"), OVERLAP_THRESHOLD)

    # Arm B: DF filter. Build a drop-set from the benchmark corpus itself (the
    # "source"), then decon with the common ngrams excluded from every overlap.
    drop_dir = f"{args.work.rstrip('/')}/drop"
    dropped = build_source_drop_set(
        df_sample_dir=docs_dir,
        prebuilt_bloom_dir=bloom_dir,
        output_path=drop_dir,
        ngram=ngram,
        sample_docs=len(docs),
        common_frac=DF_COMMON_FRAC,
        common_min_abs=DF_COMMON_MIN_ABS,
    )
    decon_to_parquet(
        normalized_data=norm,
        prebuilt_bloom_dir=bloom_dir,
        output_path=f"{args.work.rstrip('/')}/decon_b",
        ngram=ngram,
        drop_set_dir=drop_dir,
    )
    metrics_b = _score(labels, _read_predictions(f"{args.work.rstrip('/')}/decon_b"), OVERLAP_THRESHOLD)

    metrics = {
        "arm_a_no_filter": metrics_a,
        "arm_b_df_filter": metrics_b,
        "config": {
            "paragraph_delimiter": delimiter,
            "ngram_length": NGRAM_LENGTH,
            "n_docs": len(docs),
            "df_common_frac": DF_COMMON_FRAC,
            "df_common_min_abs": DF_COMMON_MIN_ABS,
            "df_ngrams_dropped": dropped.n_dropped,
        },
    }
    ofs, opath = url_to_fs(args.out)
    with ofs.open(opath, "w") as fh:
        json.dump(metrics, fh, indent=2)
    _log_comparison(metrics_a, metrics_b, dropped.n_dropped, delimiter)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
