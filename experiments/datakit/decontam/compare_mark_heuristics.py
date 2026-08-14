# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare decontamination mark rules on an existing flagged-document sample."""

import argparse
import hashlib
import html
import json
import logging
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.datakit.decon import (
    DeconAttributes,
    NGramConfig,
    _bloom_hash,
    _document_overlap_matches_by_minimum,
    _extract_features,
    _load_drop_sets,
    bloom_paths,
)
from marin.execution.artifact import read_artifact
from rigging.filesystem import StoragePath, marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.datakit.decontam.prepare_eval_corpus import AA_EVALS
from experiments.datakit.decontam.viewer.export_run import _flagged_from_sidecar, _window, eval_id_to_family
from experiments.datakit.reference_pipeline import FLAGGED_SAMPLE_SIZE, select_sources
from experiments.datakit.reports.common import sample_rows

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_REPORT = "s3://marin-us-east-02a/marin/datakit/report/decontam_899ee8ee/report.html"
DEFAULT_BLOOM_DIR = "s3://marin-us-east-02a/marin/datakit/bloom/_combined_fixed_e8043260"
DEFAULT_DROP_SET_DIR = "s3://marin-us-east-02a/marin/datakit/decon_drop/_combined_50513514"
DEFAULT_OUTPUT_DIR = "s3://marin-us-east-02a/marin/datakit/decontam/reports/mark-heuristic-sample-v1-20260814"
BASELINE_ATTRIBUTE_SCHEMA_VERSION = 4
FEATURE_FILTER_VERSION = 3
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
PARAGRAPH_DELIMITER = "\n\n"
ROWS_PER_SOURCE = 40
INDEX_BATCH_ROWS = 131_072
EXAMPLES_PER_SECTION = 30
TEXT_CONTEXT_CHARS = 700
VARIANTS = (1, 2, 3)
_PROCESS_BLOOM: dupekit.Bloom | None = None
FLAGGED_COLUMNS = ["id", "text", "max_overlap", "matched_hashes"]


def _report_data(report_path: str) -> dict[str, Any]:
    page = StoragePath(report_path).read_text()
    start_marker = "const D = "
    end_marker = ";\nconst fmt"
    start = page.index(start_marker) + len(start_marker)
    end = page.index(end_marker, start)
    return json.loads(page[start:end])


def _baseline_mark_path(source: str, normalized_name: str) -> str:
    attrs = {
        "text_field": "text",
        "ngram_length": NGRAM_LENGTH,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "paragraph_delimiter": PARAGRAPH_DELIMITER,
        "feature_filter_version": FEATURE_FILTER_VERSION,
        "attribute_schema_version": BASELINE_ATTRIBUTE_SCHEMA_VERSION,
        "input_dir": None,
        "flagged_sample_size": FLAGGED_SAMPLE_SIZE,
    }
    payload = json.dumps(
        {
            "name": f"datakit/decontam/{source}",
            "attrs": attrs,
            "deps": sorted(
                [
                    normalized_name,
                    "datakit/bloom/_combined_fixed_e8043260",
                    "datakit/decon_drop/_combined_50513514",
                ]
            ),
        },
        sort_keys=True,
    )
    hash_id = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{marin_prefix().rstrip('/')}/datakit/decontam/{source}_{hash_id}"


def _sample_flagged_shards(mark_path: str, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    files = list(StoragePath(f"{mark_path.rstrip('/')}/outputs/flagged_sample/*.parquet").glob())
    rng.shuffle(files)
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("rb") as stream:
            file_rows = pq.read_table(stream, columns=FLAGGED_COLUMNS).to_pylist()
        rng.shuffle(file_rows)
        rows.extend(file_rows[: limit - len(rows)])
        if len(rows) == limit:
            break
    return rows


def _sample_documents(
    report: dict[str, Any],
    *,
    rows_per_source: int,
    all_sources: bool,
    reservoir: bool,
    shard_sample: bool,
    seed: int,
) -> list[dict[str, Any]]:
    source_names = (
        [source["name"] for source in report["sources"]]
        if all_sources
        else [group["source"] for group in report["flagged"]]
    )
    sources = select_sources(source_names)
    documents: list[dict[str, Any]] = []
    for index, source in enumerate(source_names, start=1):
        mark_path = _baseline_mark_path(source, sources[source].name_with_hash)
        attrs = read_artifact(mark_path, DeconAttributes)
        rng = random.Random(f"{seed}:{source}")
        if shard_sample:
            rows = _sample_flagged_shards(mark_path, rows_per_source, rng)
        elif reservoir:
            rows = _flagged_from_sidecar(mark_path, rows_per_source, rng) or []
        else:
            rows = sample_rows(
                attrs.flagged_output_dir,
                FLAGGED_COLUMNS,
                rows_per_source,
            )
        logger.info("loaded %d documents for source %d/%d: %s", len(rows), index, len(source_names), source)
        documents.extend({"source": source, **row} for row in rows)
    return documents


def _matching_feature_text(text: str, matched_hashes: list[int]) -> list[str]:
    wanted = set(matched_hashes)
    seen: set[str] = set()
    features: list[str] = []
    ngram = NGramConfig(
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        min_matched_features=1,
        paragraph_delimiter=PARAGRAPH_DELIMITER,
    )
    for feature in _extract_features(text, ngram):
        if feature in seen or _bloom_hash(feature) not in wanted:
            continue
        seen.add(feature)
        features.append(feature)
    return features


def _score_source_documents(
    documents: list[dict[str, Any]], bloom: dupekit.Bloom, drop_hashes: frozenset[int]
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    ngram = NGramConfig(
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        min_matched_features=1,
        paragraph_delimiter=PARAGRAPH_DELIMITER,
    )
    for document in documents:
        max_overlap, matches = _document_overlap_matches_by_minimum(
            document["text"], bloom, ngram, VARIANTS, drop_hashes
        )
        baseline_features = _matching_feature_text(document["text"], matches[1])
        variants: dict[str, Any] = {}
        for minimum in VARIANTS:
            matched = matches[minimum]
            wanted = set(matched)
            variants[str(minimum)] = {
                "marked": bool(matched),
                "max_overlap": max_overlap,
                "matched_hashes": matched,
                "matched_features": [feature for feature in baseline_features if _bloom_hash(feature) in wanted],
            }
        scored.append({**document, "variants": variants})
    return scored


def _score_source_process(payload: tuple[list[dict[str, Any]], frozenset[int]]) -> list[dict[str, Any]]:
    if _PROCESS_BLOOM is None:
        raise RuntimeError("worker Bloom is not initialized")
    documents, drop_hashes = payload
    return _score_source_documents(documents, _PROCESS_BLOOM, drop_hashes)


def _score_documents(
    documents: list[dict[str, Any]], bloom: dupekit.Bloom, drop_set_dir: str, processes: int
) -> list[dict[str, Any]]:
    global _PROCESS_BLOOM

    global_drop_dir = f"{drop_set_dir.rstrip('/')}/_global"
    by_source: dict[str, list[dict[str, Any]]] = {}
    for order, document in enumerate(documents):
        by_source.setdefault(document["source"], []).append({**document, "_sample_order": order})
    payloads = []
    for source, source_documents in by_source.items():
        drop_hashes = _load_drop_sets([f"{drop_set_dir.rstrip('/')}/{source}", global_drop_dir])
        payloads.append((source_documents, drop_hashes))

    if processes == 1:
        scored = [
            document
            for source_documents, drop_hashes in payloads
            for document in _score_source_documents(source_documents, bloom, drop_hashes)
        ]
    else:
        _PROCESS_BLOOM = bloom
        scored = []
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=processes, mp_context=context) as executor:
            futures = [executor.submit(_score_source_process, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                scored.extend(future.result())
                logger.info("scored %d/%d sources", completed, len(futures))
        _PROCESS_BLOOM = None
    scored.sort(key=lambda document: document["_sample_order"])
    for document in scored:
        del document["_sample_order"]
    return scored


def _hash_to_eval_ids(index_path: str, hashes: set[int]) -> dict[int, set[str]]:
    if not hashes:
        return {}
    wanted = pa.array(sorted(hashes), type=pa.uint64())
    mapping: dict[int, set[str]] = {}
    with StoragePath(index_path).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=INDEX_BATCH_ROWS, columns=["hash", "eval_id"]):
            selected = batch.filter(pc.is_in(batch.column("hash"), value_set=wanted))
            for hash_value, eval_id in zip(
                selected.column("hash").to_pylist(),
                selected.column("eval_id").to_pylist(),
                strict=True,
            ):
                mapping.setdefault(hash_value, set()).add(str(eval_id))
    return mapping


def _aa_benchmark(eval_id: str) -> str | None:
    family = eval_id_to_family(eval_id)
    for config in AA_EVALS:
        if family == config.subdir or family.startswith(f"{config.subdir}-"):
            return config.name
    return None


def _add_attribution(scored: list[dict[str, Any]], mapping: dict[int, set[str]]) -> None:
    for document in scored:
        for variant in document["variants"].values():
            eval_ids = sorted(
                {eval_id for hash_value in variant["matched_hashes"] for eval_id in mapping.get(hash_value, ())}
            )
            variant["eval_ids"] = eval_ids
            variant["eval_families"] = sorted({eval_id_to_family(eval_id) for eval_id in eval_ids})
            variant["aa_benchmarks"] = sorted(
                {benchmark for eval_id in eval_ids if (benchmark := _aa_benchmark(eval_id)) is not None}
            )


def _summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    variant_counts = {
        str(minimum): sum(document["variants"][str(minimum)]["marked"] for document in scored) for minimum in VARIANTS
    }
    aa_counts = {
        config.name: {
            str(minimum): sum(config.name in document["variants"][str(minimum)]["aa_benchmarks"] for document in scored)
            for minimum in VARIANTS
        }
        for config in AA_EVALS
    }
    source_counts = []
    for source in sorted({document["source"] for document in scored}):
        source_documents = [document for document in scored if document["source"] == source]
        source_counts.append(
            {
                "source": source,
                "sampled": len(source_documents),
                **{
                    str(minimum): sum(document["variants"][str(minimum)]["marked"] for document in source_documents)
                    for minimum in VARIANTS
                },
            }
        )
    return {
        "sampled_documents": len(scored),
        "sampled_sources": len({document["source"] for document in scored}),
        "variant_counts": variant_counts,
        "removed_by_two": sum(
            document["variants"]["1"]["marked"] and not document["variants"]["2"]["marked"] for document in scored
        ),
        "removed_by_three": sum(
            document["variants"]["2"]["marked"] and not document["variants"]["3"]["marked"] for document in scored
        ),
        "aa_counts": aa_counts,
        "source_counts": source_counts,
    }


def _example_rows(documents: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for document in documents:
        baseline = document["variants"]["1"]
        two = document["variants"]["2"]
        three = document["variants"]["3"]
        features = baseline["matched_features"][:12]
        window = _window(document["text"], features, TEXT_CONTEXT_CHARS)
        families = ", ".join(baseline["eval_families"][:12]) or "none"
        aa = ", ".join(baseline["aa_benchmarks"]) or "none"
        feature_text = "\n".join(features) or "none"
        rows.append(
            "<details><summary>"
            f"<code>{html.escape(document['source'])}</code> · <code>{html.escape(str(document['id']))}</code> · "
            f"rules 1/2/3: {int(baseline['marked'])}/{int(two['marked'])}/{int(three['marked'])} · "
            f"features: {len(baseline['matched_hashes'])}/{len(two['matched_hashes'])}/{len(three['matched_hashes'])}"
            "</summary>"
            f"<p><b>AA:</b> {html.escape(aa)}<br><b>Eval families:</b> {html.escape(families)}</p>"
            f"<h4>Shared features</h4><pre>{html.escape(feature_text)}</pre>"
            f"<h4>Corpus text</h4><pre>{html.escape(window)}</pre>"
            "</details>"
        )
    return "".join(rows) or "<p>None.</p>"


def _stratified_examples(documents: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        by_source.setdefault(document["source"], []).append(document)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        added = False
        for source in sorted(by_source):
            if by_source[source]:
                selected.append(by_source[source].pop(0))
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def _render_report(scored: list[dict[str, Any]], summary: dict[str, Any], examples_per_section: int) -> str:
    removed = _stratified_examples(
        [
            document
            for document in scored
            if document["variants"]["1"]["marked"] and not document["variants"]["2"]["marked"]
        ],
        examples_per_section,
    )
    retained = _stratified_examples(
        [document for document in scored if document["variants"]["2"]["marked"]], examples_per_section
    )
    aa_documents = _stratified_examples(
        [document for document in scored if document["variants"]["1"]["aa_benchmarks"]], examples_per_section
    )
    rule_rows = "".join(
        f"<tr><td>{minimum}</td><td>{summary['variant_counts'][str(minimum)]}</td>"
        f"<td>{summary['variant_counts'][str(minimum)] / max(1, summary['sampled_documents']):.1%}</td></tr>"
        for minimum in VARIANTS
    )
    aa_rows = "".join(
        f"<tr><td>{html.escape(config.name)}</td>"
        + "".join(f"<td>{summary['aa_counts'][config.name][str(minimum)]}</td>" for minimum in VARIANTS)
        + "</tr>"
        for config in AA_EVALS
    )
    source_rows = "".join(
        f"<tr><td>{html.escape(row['source'])}</td><td>{row['sampled']}</td>"
        + "".join(f"<td>{row[str(minimum)]}</td>" for minimum in VARIANTS)
        + "</tr>"
        for row in summary["source_counts"]
    )
    return f"""<!doctype html>
<meta charset="utf-8"><title>Decontamination mark heuristic sample</title>
<style>
body{{font:14px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#18202a}}
table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #d7dde5;padding:7px;text-align:left}}
th{{background:#f3f5f8}}details{{border:1px solid #d7dde5;border-radius:8px;margin:8px 0;padding:10px}}
summary{{cursor:pointer}}pre{{white-space:pre-wrap;word-break:break-word;background:#f6f7f9;padding:12px;max-height:420px;overflow:auto}}
.note{{background:#fff5d8;border:1px solid #e4ca78;border-radius:8px;padding:12px}}code{{font-size:12px}}
</style>
<h1>Decontamination mark heuristic sample</h1>
<p>Compared one, two, and three distinct matching features per paragraph on
{summary["sampled_documents"]} documents from {summary["sampled_sources"]} sources. These documents were flagged by
the current one-feature rule.</p>
<p class="note"><b>Limit:</b> This sample tests precision on earlier marks. It cannot measure recall or show documents
that the earlier rule did not flag.</p>
<h2>Rule comparison</h2>
<table><tr><th>Minimum distinct features</th><th>Documents marked</th><th>Share of sample</th></tr>{rule_rows}</table>
<p>The two-feature rule removed {summary["removed_by_two"]} earlier marks. The three-feature rule removed
{summary["removed_by_three"]} more marks.</p>
<h2>AA benchmark attribution</h2>
<table><tr><th>AA benchmark</th><th>One feature</th><th>Two features</th><th>Three features</th></tr>{aa_rows}</table>
<h2>Source summary</h2>
<table><tr><th>Source</th><th>Sampled</th><th>One feature</th><th>Two features</th><th>Three features</th></tr>
{source_rows}</table>
<h2>AA-attributed documents</h2>{_example_rows(aa_documents)}
<h2>Examples removed by the two-feature rule</h2>{_example_rows(removed)}
<h2>Examples retained by the two-feature rule</h2>{_example_rows(retained)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-report", default=DEFAULT_CORPUS_REPORT)
    parser.add_argument("--bloom-dir", default=DEFAULT_BLOOM_DIR)
    parser.add_argument("--drop-set-dir", default=DEFAULT_DROP_SET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rows-per-source", type=int, default=ROWS_PER_SOURCE)
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--reservoir", action="store_true")
    parser.add_argument("--shard-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-section", type=int, default=EXAMPLES_PER_SECTION)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    if args.rows_per_source < 1 or args.examples_per_section < 1 or args.processes < 1:
        parser.error("row, example, and process counts must be positive")
    if args.reservoir and args.shard_sample:
        parser.error("choose at most one sampling method")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()
    report = _report_data(args.corpus_report)
    documents = _sample_documents(
        report,
        rows_per_source=args.rows_per_source,
        all_sources=args.all_sources,
        reservoir=args.reservoir,
        shard_sample=args.shard_sample,
        seed=args.seed,
    )
    expected = int(report["stats"]["n_flagged_sampled"])
    if (
        not args.all_sources
        and not args.reservoir
        and args.rows_per_source == ROWS_PER_SOURCE
        and len(documents) != expected
    ):
        raise ValueError(f"loaded {len(documents)} documents, expected {expected} from the corpus report")

    bloom_path, index_path = bloom_paths(args.bloom_dir)
    logger.info("loading Bloom from %s", bloom_path)
    bloom = dupekit.Bloom.load_bytes(StoragePath(bloom_path).read_bytes())
    scored = _score_documents(documents, bloom, args.drop_set_dir, args.processes)
    needed_hashes = {
        hash_value
        for document in scored
        for variant in document["variants"].values()
        for hash_value in variant["matched_hashes"]
    }
    logger.info("resolving %d matched hashes against %s", len(needed_hashes), index_path)
    _add_attribution(scored, _hash_to_eval_ids(index_path, needed_hashes))
    summary = _summary(scored)

    output_dir = args.output_dir.rstrip("/")
    StoragePath(output_dir).mkdirs()
    StoragePath(f"{output_dir}/report.html").write_text(_render_report(scored, summary, args.examples_per_section))
    manifest = {
        "schema_version": 1,
        "corpus_report": args.corpus_report,
        "bloom_dir": args.bloom_dir,
        "drop_set_dir": args.drop_set_dir,
        "sampling": {
            "all_sources": args.all_sources,
            "reservoir": args.reservoir,
            "shard_sample": args.shard_sample,
            "rows_per_source": args.rows_per_source,
            "seed": args.seed,
            "processes": args.processes,
        },
        "summary": summary,
        "documents": [
            {
                "source": document["source"],
                "id": document["id"],
                "variants": {
                    minimum: {
                        key: variant[key]
                        for key in ("marked", "max_overlap", "matched_features", "eval_families", "aa_benchmarks")
                    }
                    for minimum, variant in document["variants"].items()
                },
            }
            for document in scored
        ],
    }
    StoragePath(f"{output_dir}/manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s", output_dir)
    logger.info("summary: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
