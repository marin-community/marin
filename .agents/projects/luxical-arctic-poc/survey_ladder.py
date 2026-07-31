# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Survey the fixed Luxical ladder data with bounded private text views."""

import base64
import gzip
import hashlib
import html
import json
import logging
import math
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import dupekit
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ladder_config import MANIFEST_ROOT, SURVEY_ROWS_PER_SOURCE
from rigging.filesystem import atomic_rename

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
SURVEY_ROOT = f"{MANIFEST_ROOT}/survey"
SURVEY_JSON_URL = f"{SURVEY_ROOT}/survey.json.gz"
SURVEY_HTML_URL = f"{SURVEY_ROOT}/report.html"
RANDOM_ROWS_PER_SOURCE = 80
SHORTEST_ROWS_PER_SOURCE = 10
LONGEST_ROWS_PER_SOURCE = 10
SNIPPET_CHARS = 500
MINHASH_PERMUTATIONS = 128
MINHASH_BANDS = 16
MINHASH_NGRAM_SIZE = 5
MINHASH_TEXT_CAP = 6_000
NEAR_DUPLICATE_THRESHOLD = 0.80
MAX_BUCKET_SIZE_FOR_PAIRS = 100
MAX_CANDIDATE_PAIRS = 2_000_000
RESULT_FILE = Path("/tmp/luxical-arctic-survey")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def selected_survey_rows(table: pa.Table) -> list[tuple[str, dict[str, Any]]]:
    """Return random, short, and long evaluation strata."""
    evaluation = table.filter(pc.equal(table["split"], "eval"))
    rows = evaluation.to_pylist()
    random_rows = [row for row in rows if row["eval_rank"] < RANDOM_ROWS_PER_SOURCE]
    remaining = [row for row in rows if row["eval_rank"] >= RANDOM_ROWS_PER_SOURCE]
    shortest = sorted(remaining, key=lambda row: (row["raw_characters"], row["eval_rank"]))[:SHORTEST_ROWS_PER_SOURCE]
    shortest_ranks = {row["eval_rank"] for row in shortest}
    longest_candidates = [row for row in remaining if row["eval_rank"] not in shortest_ranks]
    longest = sorted(
        longest_candidates,
        key=lambda row: (-row["raw_characters"], row["eval_rank"]),
    )[:LONGEST_ROWS_PER_SOURCE]
    selected = (
        [("random", row) for row in random_rows]
        + [("shortest", row) for row in shortest]
        + [("longest", row) for row in longest]
    )
    if len(selected) != SURVEY_ROWS_PER_SOURCE:
        raise ValueError(f"Survey selected {len(selected)} rows; expected {SURVEY_ROWS_PER_SOURCE}")
    return selected


def script_name(character: str) -> str:
    """Return a coarse Unicode script name."""
    name = unicodedata.name(character, "")
    for script in (
        "ARABIC",
        "CJK",
        "CYRILLIC",
        "DEVANAGARI",
        "HANGUL",
        "HEBREW",
        "HIRAGANA",
        "KATAKANA",
        "LATIN",
        "THAI",
    ):
        if script in name:
            return script
    return "OTHER"


def bounded_text_views(text: str) -> dict[str, str]:
    """Return private bounded views for direct inspection."""
    middle_start = max(0, len(text) // 2 - SNIPPET_CHARS // 2)
    return {
        "head": text[:SNIPPET_CHARS],
        "middle": text[middle_start : middle_start + SNIPPET_CHARS],
        "tail": text[-SNIPPET_CHARS:],
    }


def document_report(method: str, row: dict[str, Any]) -> dict[str, Any]:
    """Return survey metrics and bounded text views for one document."""
    text = row["text"]
    scripts = Counter(script_name(character) for character in text if character.isalpha())
    non_whitespace = "".join(text.split())
    return {
        "source": row["source"],
        "source_category": row["source_category"],
        "sample_method": method,
        "id_sha256": hashlib.sha256(row["id"].encode()).hexdigest(),
        "eval_rank": row["eval_rank"],
        "raw_characters": row["raw_characters"],
        "view_characters": len(text),
        "raw_sha256": row["raw_sha256"],
        "normalized_sha256": row["normalized_sha256"],
        "nonconstant": len(set(non_whitespace)) >= 2,
        "scripts": dict(sorted(scripts.items())),
        "input_path": row["input_path"],
        "input_row_group": row["input_row_group"],
        "input_row_in_group": row["input_row_in_group"],
        "views": bounded_text_views(text),
        "text_for_minhash": text[:MINHASH_TEXT_CAP],
    }


def load_documents(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Load all fixed survey strata."""
    documents = []
    columns = [
        "id",
        "source",
        "source_category",
        "split",
        "eval_rank",
        "raw_characters",
        "raw_sha256",
        "normalized_sha256",
        "input_path",
        "input_row_group",
        "input_row_in_group",
        "text",
    ]
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Surveying source %d/%d: %s", index, len(manifest["sources"]), source)
        filesystem, path = fsspec.core.url_to_fs(result["output_url"])
        table = pq.read_table(path, filesystem=filesystem, columns=columns)
        documents.extend(document_report(method, row) for method, row in selected_survey_rows(table))
    return documents


def duplicate_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Return raw and normalized exact duplicate statistics."""
    raw_groups: dict[str, list[int]] = defaultdict(list)
    normalized_groups: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        raw_groups[document["raw_sha256"]].append(index)
        normalized_groups[document["normalized_sha256"]].append(index)

    def summary(groups: dict[str, list[int]]) -> dict[str, Any]:
        duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
        duplicate_documents = {index for indices in duplicate_groups for index in indices}
        cross_source_groups = [
            indices for indices in duplicate_groups if len({documents[index]["source"] for index in indices}) > 1
        ]
        return {
            "unique_fraction": 1.0 - len(duplicate_documents) / len(documents),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_document_count": len(duplicate_documents),
            "cross_source_group_count": len(cross_source_groups),
        }

    return {"raw": summary(raw_groups), "normalized": summary(normalized_groups)}


def near_duplicate_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Return bounded MinHash-LSH near-duplicate statistics."""
    batch = pa.RecordBatch.from_pydict(
        {
            "row_index": list(range(len(documents))),
            "text": [document["text_for_minhash"] for document in documents],
        }
    )
    transformed = dupekit.transform(
        batch,
        [
            dupekit.Transformation.CleanText(input_col="text", output_col="clean_text"),
            dupekit.Transformation.MinHash(
                input_col="clean_text",
                output_col="signature",
                num_perms=MINHASH_PERMUTATIONS,
                ngram_size=MINHASH_NGRAM_SIZE,
                seed=42,
            ),
            dupekit.Transformation.MinHashLSH(
                input_col="signature",
                output_col="buckets",
                num_bands=MINHASH_BANDS,
            ),
        ],
    )
    signatures = [value.as_py() for value in transformed["signature"]]
    bucket_members: dict[int, list[int]] = defaultdict(list)
    for row_index, buckets in enumerate(transformed["buckets"]):
        if buckets.is_valid:
            for bucket in buckets.as_py():
                bucket_members[int(bucket)].append(row_index)

    candidate_pairs: set[tuple[int, int]] = set()
    oversized_buckets = 0
    candidates_truncated = False
    for members in bucket_members.values():
        unique_members = sorted(set(members))
        if len(unique_members) > MAX_BUCKET_SIZE_FOR_PAIRS:
            oversized_buckets += 1
            continue
        for left, right in combinations(unique_members, 2):
            candidate_pairs.add((left, right))
            if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                candidates_truncated = True
                break
        if candidates_truncated:
            break

    near_pairs = []
    near_documents = set()
    for left, right in sorted(candidate_pairs):
        left_signature = np.asarray(signatures[left], dtype=np.uint64)
        right_signature = np.asarray(signatures[right], dtype=np.uint64)
        similarity = float(np.mean(left_signature == right_signature))
        if similarity >= NEAR_DUPLICATE_THRESHOLD:
            near_pairs.append(
                {
                    "left_source": documents[left]["source"],
                    "left_id_sha256": documents[left]["id_sha256"],
                    "right_source": documents[right]["source"],
                    "right_id_sha256": documents[right]["id_sha256"],
                    "estimated_jaccard": similarity,
                }
            )
            near_documents.update((left, right))
    return {
        "method": "128-permutation, 5-character MinHash with 16 LSH bands",
        "estimated_jaccard_threshold": NEAR_DUPLICATE_THRESHOLD,
        "candidate_pair_count": len(candidate_pairs),
        "candidate_pairs_truncated": candidates_truncated,
        "oversized_bucket_count": oversized_buckets,
        "near_duplicate_pair_count": len(near_pairs),
        "near_duplicate_document_count": len(near_documents),
        "near_duplicate_document_fraction": len(near_documents) / len(documents),
        "example_pairs": near_pairs[:100],
    }


def source_summaries(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return length, script, and constant checks for every source."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        by_source[document["source"]].append(document)
    summaries = {}
    for source, rows in sorted(by_source.items()):
        lengths = np.asarray([row["raw_characters"] for row in rows])
        scripts = Counter()
        for row in rows:
            scripts.update(row["scripts"])
        summaries[source] = {
            "category": rows[0]["source_category"],
            "rows": len(rows),
            "nonconstant_fraction": float(np.mean([row["nonconstant"] for row in rows])),
            "minimum_characters": int(lengths.min()),
            "median_characters": float(np.median(lengths)),
            "p90_characters": float(np.quantile(lengths, 0.9)),
            "maximum_characters": int(lengths.max()),
            "scripts": dict(sorted(scripts.items())),
        }
    return summaries


def length_plot_svg(documents: list[dict[str, Any]]) -> str:
    """Return an SVG heat map of length distributions by category."""
    bins = (0, 2, 3, 4, 5, 6, math.inf)
    labels = ("<100", "100-999", "1K-9K", "10K-99K", "100K-999K", ">=1M")
    categories = sorted({document["source_category"] for document in documents})
    counts = {}
    for category in categories:
        lengths = [document["raw_characters"] for document in documents if document["source_category"] == category]
        log_lengths = [math.log10(max(1, length)) for length in lengths]
        category_counts = []
        for left, right in pairwise(bins):
            category_counts.append(sum(left <= value < right for value in log_lengths) / len(log_lengths))
        counts[category] = category_counts

    cell_width = 105
    cell_height = 42
    label_width = 120
    width = label_width + cell_width * len(labels) + 20
    height = 70 + cell_height * len(categories)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Document length heat map">']
    for column, label in enumerate(labels):
        x = label_width + column * cell_width + cell_width / 2
        parts.append(f'<text x="{x}" y="22" text-anchor="middle" font-size="12">{html.escape(label)}</text>')
    for row, category in enumerate(categories):
        y = 38 + row * cell_height
        parts.append(
            f'<text x="{label_width - 8}" y="{y + 25}" text-anchor="end" font-size="13">'
            f"{html.escape(category)}</text>"
        )
        for column, fraction in enumerate(counts[category]):
            x = label_width + column * cell_width
            opacity = 0.12 + 0.88 * fraction
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 3}" height="{cell_height - 3}" '
                f'fill="#2855a6" opacity="{opacity:.3f}"/>'
            )
            parts.append(
                f'<text x="{x + cell_width / 2}" y="{y + 25}" text-anchor="middle" '
                f'font-size="12" fill="{"white" if fraction > 0.45 else "black"}">{fraction:.1%}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def inspection_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one random, short, and long private view per source."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        by_source[document["source"]].append(document)
    selected = []
    for source_rows in by_source.values():
        for method in ("random", "shortest", "longest"):
            selected.append(next(row for row in source_rows if row["sample_method"] == method))
    return selected


def html_report(report: dict[str, Any]) -> str:
    """Render one standalone private data report."""
    source_rows = "".join(
        "<tr>"
        f"<td>{html.escape(source)}</td>"
        f"<td>{html.escape(result['category'])}</td>"
        f"<td>{result['rows']}</td>"
        f"<td>{result['nonconstant_fraction']:.1%}</td>"
        f"<td>{result['median_characters']:,.0f}</td>"
        f"<td>{result['p90_characters']:,.0f}</td>"
        f"<td>{result['maximum_characters']:,}</td>"
        "</tr>"
        for source, result in report["source_summaries"].items()
    )
    inspection = []
    for document in report["inspection_documents"]:
        views = document["views"]
        inspection.append(
            f"<details><summary>{html.escape(document['source'])} — "
            f"{html.escape(document['sample_method'])} — {document['raw_characters']:,} characters</summary>"
            f"<h4>Head</h4><pre>{html.escape(views['head'])}</pre>"
            f"<h4>Middle</h4><pre>{html.escape(views['middle'])}</pre>"
            f"<h4>Tail</h4><pre>{html.escape(views['tail'])}</pre></details>"
        )
    exact = report["exact_duplicates"]
    near = report["near_duplicates"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Luxical ladder data survey</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 100rem; }}
table {{ border-collapse: collapse; width: 100%; }} td, th {{ border: 1px solid #bbb; padding: .3rem; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f5f5; padding: .8rem; }}
svg {{ width: 100%; max-width: 900px; }}
</style></head><body>
<h1>Luxical ladder data survey</h1>
<p>{report["document_count"]:,} documents across {report["source_count"]} sources.</p>
<p>Nonconstant: {report["nonconstant_fraction"]:.3%}. Raw exact uniqueness:
{exact["raw"]["unique_fraction"]:.3%}. Normalized exact uniqueness:
{exact["normalized"]["unique_fraction"]:.3%}. Near-duplicate document fraction:
{near["near_duplicate_document_fraction"]:.3%}.</p>
<h2>Length distribution</h2>{report["length_plot_svg"]}
<h2>Source summary</h2>
<table><thead><tr><th>Source</th><th>Category</th><th>Rows</th><th>Nonconstant</th>
<th>Median chars</th><th>P90 chars</th><th>Max chars</th></tr></thead><tbody>{source_rows}</tbody></table>
<h2>Direct private inspection</h2>{''.join(inspection)}
</body></html>"""


def write_bytes(url: str, payload: bytes) -> None:
    """Write bytes atomically to private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "wb") as file:
            file.write(payload)


def main() -> None:
    """Build the full bounded data survey."""
    manifest = read_json(MANIFEST_URL)
    documents = load_documents(manifest)
    expected_rows = len(manifest["sources"]) * SURVEY_ROWS_PER_SOURCE
    if len(documents) != expected_rows:
        raise ValueError(f"Survey has {len(documents)} rows; expected {expected_rows}")
    exact_duplicates = duplicate_summary(documents)
    near_duplicates = near_duplicate_summary(documents)
    summaries = source_summaries(documents)
    report = {
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "source_count": len(manifest["sources"]),
        "document_count": len(documents),
        "sampling": {
            "random_rows_per_source": RANDOM_ROWS_PER_SOURCE,
            "shortest_rows_per_source": SHORTEST_ROWS_PER_SOURCE,
            "longest_rows_per_source": LONGEST_ROWS_PER_SOURCE,
        },
        "nonconstant_fraction": float(np.mean([document["nonconstant"] for document in documents])),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "source_summaries": summaries,
        "length_plot_svg": length_plot_svg(documents),
        "inspection_documents": inspection_rows(documents),
        "documents": [
            {key: value for key, value in document.items() if key != "text_for_minhash"} for document in documents
        ],
    }
    write_bytes(
        SURVEY_JSON_URL,
        gzip.compress(json.dumps(report, ensure_ascii=False, sort_keys=True).encode()),
    )
    write_bytes(SURVEY_HTML_URL, html_report(report).encode())
    filesystem, html_path = fsspec.core.url_to_fs(SURVEY_HTML_URL)
    _, json_path = fsspec.core.url_to_fs(SURVEY_JSON_URL)
    signed_html_url = filesystem.sign(html_path, expiration=7_200)
    signed_json_url = filesystem.sign(json_path, expiration=7_200)
    summary = {
        "source_count": report["source_count"],
        "document_count": report["document_count"],
        "nonconstant_fraction": report["nonconstant_fraction"],
        "raw_unique_fraction": exact_duplicates["raw"]["unique_fraction"],
        "normalized_unique_fraction": exact_duplicates["normalized"]["unique_fraction"],
        "near_duplicate_document_fraction": near_duplicates["near_duplicate_document_fraction"],
        "survey_json_url": SURVEY_JSON_URL,
        "survey_html_url": SURVEY_HTML_URL,
        "signed_json_url_base64": base64.b64encode(signed_json_url.encode()).decode(),
        "signed_html_url_base64": base64.b64encode(signed_html_url.encode()).decode(),
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_SURVEY=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
