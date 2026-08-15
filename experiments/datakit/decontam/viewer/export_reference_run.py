# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export completed reference-pipeline decontamination marks for inspection."""

import argparse
import hashlib
import html
import json
import logging
import random
import time
from collections import Counter

import pyarrow.parquet as pq
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import read_artifact
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.decontam.prepare_eval_corpus import AA_EVALS
from experiments.datakit.decontam.viewer.export_run import (
    _overlapping_ngrams,
    _window,
    eval_id_to_family,
)
from experiments.datakit.decontam.viewer.report import _single
from experiments.datakit.reference_pipeline import (
    AA_BENCHMARK_NAMES,
    EVAL_CORPUS_VERSION,
    EVAL_ROOT,
    decontamination_steps,
    select_sources,
)

logger = logging.getLogger(__name__)

INDEX_BATCH_ROWS = 131_072
GALLERY_ROWS_PER_SOURCE = 5
MAX_MATCHED_EVALS = 5
AA_SUBDIR_TO_NAME = {config.subdir: config.name for config in AA_EVALS}


def _eval_family(eval_id: str) -> str:
    for subdir, name in AA_SUBDIR_TO_NAME.items():
        if eval_id == subdir or eval_id.startswith(f"{subdir}-"):
            return name
    return eval_id_to_family(eval_id)


def _seed_for_source(seed: int, source: str) -> int:
    digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_flagged_rows(output_dir: str, limit: int, seed: int) -> tuple[list[dict], list[str]]:
    files = sorted(str(path) for path in StoragePath(f"{output_dir.rstrip('/')}/*.parquet").glob())
    random.Random(seed).shuffle(files)
    rows: list[dict] = []
    used_files: list[str] = []
    columns = ["id", "text", "max_overlap", "matched_hashes"]
    for path in files:
        used_files.append(path)
        with StoragePath(path).open("rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=min(limit, 4096), columns=columns):
                rows.extend(batch.to_pylist()[: limit - len(rows)])
                if len(rows) >= limit:
                    return rows, used_files
    return rows, used_files


def _completed_marks() -> tuple[dict[str, object], dict[str, object]]:
    sources = select_sources(None)
    decon = decontamination_steps(sources)
    complete: dict[str, object] = {}
    for name, step in decon.marks.items():
        if StoragePath(f"{step.output_path.rstrip('/')}/.artifact.json").exists():
            complete[name] = step
    return complete, decon.marks


def _wait_for_marks(minimum_sources: int, timeout: int) -> tuple[dict[str, object], dict[str, object]]:
    deadline = time.monotonic() + timeout
    while True:
        complete, all_marks = _completed_marks()
        logger.info("completed marks: %d/%d; gate=%d", len(complete), len(all_marks), minimum_sources)
        if len(complete) >= minimum_sources:
            return complete, all_marks
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {len(complete)}/{len(all_marks)} marks completed before the timeout")
        time.sleep(30)


def _hash_to_evals(index_path: str, needed_hashes: set[int]) -> dict[int, set[str]]:
    mapping: dict[int, set[str]] = {}
    with StoragePath(index_path).open("rb") as fh:
        for batch in pq.ParquetFile(fh).iter_batches(
            batch_size=INDEX_BATCH_ROWS,
            columns=["hash", "eval_id"],
        ):
            for feature_hash, eval_id in zip(
                batch.column("hash").to_pylist(),
                batch.column("eval_id").to_pylist(),
                strict=True,
            ):
                if feature_hash in needed_hashes:
                    mapping.setdefault(feature_hash, set()).add(str(eval_id))
    return mapping


def _load_eval_texts(eval_ids: set[str]) -> dict[str, str]:
    fs, root = url_to_fs(EVAL_ROOT)
    texts: dict[str, str] = {}
    for path in sorted(path for path in fs.find(root) if path.endswith(".parquet")):
        with fs.open(path, "rb") as fh:
            parquet = pq.ParquetFile(fh)
            if not {"id", "text"}.issubset(parquet.schema_arrow.names):
                continue
            for batch in parquet.iter_batches(batch_size=INDEX_BATCH_ROWS, columns=["id", "text"]):
                for eval_id, text in zip(
                    batch.column("id").to_pylist(),
                    batch.column("text").to_pylist(),
                    strict=True,
                ):
                    if eval_id in eval_ids:
                        texts[str(eval_id)] = str(text)
    return texts


def _rate_histogram(rates: list[float]) -> str:
    bounds = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1]
    labels = [
        "0",
        "(0,1e-8]",
        "(1e-8,1e-7]",
        "(1e-7,1e-6]",
        "(1e-6,1e-5]",
        "(1e-5,1e-4]",
        "(1e-4,1e-3]",
        "(1e-3,1e-2]",
        ">1e-2",
    ]
    counts = [0] * len(labels)
    for rate in rates:
        if rate == 0:
            counts[0] += 1
            continue
        placed = False
        for index, bound in enumerate(bounds[1:], start=1):
            if rate <= bound:
                counts[index] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    maximum = max(counts, default=1) or 1
    rows = "".join(
        f"<div style='display:grid;grid-template-columns:110px 1fr 45px;gap:8px;margin:3px 0'>"
        f"<span>{html.escape(label)}</span>"
        f"<span style='background:#26384b;width:{100 * count / maximum:.1f}%'>&nbsp;</span>"
        f"<span>{count}</span></div>"
        for label, count in zip(labels, counts, strict=True)
    )
    return f"<div style='max-width:620px'>{rows}</div>"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-sources", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=28_800)
    parser.add_argument("--samples-per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    complete, all_marks = _wait_for_marks(args.minimum_sources, args.timeout)
    sampled_by_source: dict[str, list[dict]] = {}
    sample_files: dict[str, list[str]] = {}
    source_hash_counts: dict[str, Counter[int]] = {}
    attrs_by_source: dict[str, DeconAttributes] = {}
    needed_hashes: set[int] = set()
    manifest_rows: list[dict] = []

    for source, step in sorted(complete.items()):
        attrs = read_artifact(step.output_path, DeconAttributes)
        attrs_by_source[source] = attrs
        rows, files = _sample_flagged_rows(
            str(attrs.flagged_output_dir),
            args.samples_per_source,
            _seed_for_source(args.seed, source),
        )
        sampled_by_source[source] = rows
        sample_files[source] = files
        counts: Counter[int] = Counter()
        for row in rows:
            hashes = [int(value) for value in row.get("matched_hashes") or []]
            counts.update(hashes)
            manifest_rows.append(
                {
                    "source": source,
                    "id": str(row["id"]),
                    "max_overlap": float(row["max_overlap"]),
                    "matched_hashes": hashes,
                }
            )
        source_hash_counts[source] = counts
        needed_hashes.update(counts)

    index_paths = {str(attrs.eval_hash_index_path) for attrs in attrs_by_source.values()}
    if len(index_paths) != 1:
        raise ValueError(f"expected one eval hash index, found {sorted(index_paths)}")
    index_path = index_paths.pop()
    hash_to_evals = _hash_to_evals(index_path, needed_hashes)

    gallery: dict[str, list[dict]] = {}
    gallery_eval_ids: set[str] = set()
    for source, rows in sampled_by_source.items():
        selected = rows[:GALLERY_ROWS_PER_SOURCE]
        docs: list[dict] = []
        for row in selected:
            matched_hashes = [int(value) for value in row.get("matched_hashes") or []]
            eval_hits: Counter[str] = Counter()
            families: Counter[str] = Counter()
            for feature_hash in matched_hashes:
                for eval_id in hash_to_evals.get(feature_hash, ()):
                    eval_hits[eval_id] += 1
                    families[_eval_family(eval_id)] += 1
            chosen_evals = eval_hits.most_common(MAX_MATCHED_EVALS)
            gallery_eval_ids.update(eval_id for eval_id, _ in chosen_evals)
            text = str(row["text"])
            matched_ngrams = _overlapping_ngrams(text, set(matched_hashes))
            docs.append(
                {
                    "id": str(row["id"]),
                    "max_overlap": float(row["max_overlap"]),
                    "n_matched": len(matched_hashes),
                    "families": families.most_common(8),
                    "matched_ngrams": matched_ngrams,
                    "matched_eval_hits": chosen_evals,
                    "text": _window(text, matched_ngrams),
                }
            )
        gallery[source] = docs

    eval_texts = _load_eval_texts(gallery_eval_ids)
    per_source: list[dict] = []
    total_aa_hits: Counter[str] = Counter()
    for source, attrs in attrs_by_source.items():
        contaminated = int(attrs.counters.get("decon/contaminated", 0))
        clean = int(attrs.counters.get("decon/clean", 0))
        total = contaminated + clean
        family_counts: Counter[str] = Counter()
        for feature_hash, count in source_hash_counts[source].items():
            for eval_id in hash_to_evals.get(feature_hash, ()):
                family_counts[_eval_family(eval_id)] += count
        for family, count in family_counts.items():
            if family in AA_BENCHMARK_NAMES:
                total_aa_hits[family] += count
        docs = gallery[source]
        for doc in docs:
            matched_ngrams = doc["matched_ngrams"]
            doc["matched_evals"] = [
                {
                    "eval_id": eval_id,
                    "family": _eval_family(eval_id),
                    "hits": hits,
                    "text": _window(eval_texts.get(eval_id, ""), matched_ngrams),
                }
                for eval_id, hits in doc.pop("matched_eval_hits")
            ]
        per_source.append(
            {
                "name": source,
                "docs": total,
                "flagged": contaminated,
                "rate": contaminated / total if total else 0.0,
                "top_families": family_counts.most_common(12),
                "samples": docs,
            }
        )

    per_source.sort(key=lambda item: item["rate"], reverse=True)
    run = {
        "label": args.label,
        "target_tokens_b": 0,
        "exclude": [],
        "root": str(next(iter(attrs_by_source.values())).main_output_dir),
        "sources": per_source,
    }
    exact_docs = sum(source["docs"] for source in per_source)
    exact_flagged = sum(source["flagged"] for source in per_source)
    coverage = (
        f"{len(complete)}/{len(all_marks)} completed sources; exact counters cover {exact_docs:,} documents. "
        f"The sample manifest has {len(manifest_rows):,} flagged rows from shuffled shard files with seed {args.seed}."
    )
    aa_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{total_aa_hits.get(name, 0):,}</td></tr>" for name in AA_BENCHMARK_NAMES
    )
    summary = (
        "<section style='margin:0 0 18px;padding:12px;background:#171a21;border:1px solid #262b34'>"
        f"<h2 style='font-size:16px'>Coverage and sampling</h2><p>{html.escape(coverage)}</p>"
        f"<p>Eval corpus: {html.escape(EVAL_CORPUS_VERSION)}. Exact flagged count: {exact_flagged:,}. "
        "AA hit counts below are sample-based feature attributions, not full-corpus recall counts.</p>"
        "<h3 style='font-size:14px'>Source contamination-rate distribution</h3>"
        f"{_rate_histogram([source['rate'] for source in per_source])}"
        "<h3 style='font-size:14px'>Mandatory AA sample attributions</h3>"
        f"<table style='max-width:620px'><thead><tr><th>benchmark</th><th>sample feature hits</th></tr></thead>"
        f"<tbody>{aa_rows}</tbody></table></section>"
    )
    page = _single(run).replace("<main>", f"<main>{summary}", 1)

    output = args.out.rstrip("/")
    with StoragePath(f"{output}/report.json").open("w") as fh:
        json.dump(run, fh)
    with StoragePath(f"{output}/sample_manifest.jsonl").open("w") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row) + "\n")
    sampling = {
        "label": args.label,
        "eval_corpus_version": EVAL_CORPUS_VERSION,
        "eval_root": EVAL_ROOT,
        "eval_hash_index": index_path,
        "completed_sources": len(complete),
        "total_sources": len(all_marks),
        "samples_per_source": args.samples_per_source,
        "gallery_rows_per_source": GALLERY_ROWS_PER_SOURCE,
        "seed": args.seed,
        "sample_files": sample_files,
        "source_outputs": {source: step.output_path for source, step in complete.items()},
    }
    with StoragePath(f"{output}/sampling.json").open("w") as fh:
        json.dump(sampling, fh, indent=2)
    with StoragePath(f"{output}/report.html").open("w") as fh:
        fh.write(page)
    logger.info("wrote %s with %d sources and %d sampled flagged rows", output, len(complete), len(manifest_rows))


if __name__ == "__main__":
    main()
