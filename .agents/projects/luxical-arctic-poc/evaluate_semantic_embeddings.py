# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare saved embedding models on source-blind GLM semantic labels."""

import hashlib
import html
import json
import logging
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from evaluate_fast_student import load_student
from evaluate_ladder import (
    BASELINE_FILE,
    BASELINE_REPO,
    BASELINE_REVISION,
    MANIFEST_URL,
    TEACHER_EMBEDDING_DIMENSION,
    read_json,
    teacher_output_url,
)
from evaluate_teacher_candidate import (
    CANDIDATES,
    candidate_output_url,
    expected_metadata,
    quantized_vectors,
)
from evaluate_teacher_candidate import (
    normalized_vectors as normalized_candidate_vectors,
)
from glm_semantic_labels import Assignment, Bucket, SampleDocument, read_jsonl
from huggingface_hub import hf_hub_download
from luxical.embedder import Embedder
from rigging.filesystem import StoragePath, atomic_rename
from semantic_embedding_metrics import (
    cosine_order_fidelity,
    normalize_embeddings,
    semantic_metrics,
    stored_vector_rows,
    student_gates,
)

GLM_RUN_ROOT = StoragePath(
    "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/"
    "evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001"
)
OUTPUT_ROOT = GLM_RUN_ROOT / "embedding-screen-v1"
NEIGHBOR_COUNT = 10
GALLERY_DOCUMENTS = 25
GALLERY_NEIGHBORS = 3
SEED = 42
FAST_STUDENT_REPORT_URL = (
    "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/"
    "evaluation/fast-student/full/3m/report.json"
)
RESULT_FILE = Path("/tmp/luxical-semantic-embedding-screen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def semantic_sample() -> tuple[list[SampleDocument], list[Assignment], list[Bucket]]:
    """Load and validate the fixed semantic sample."""
    documents = [SampleDocument(**row) for row in read_jsonl(GLM_RUN_ROOT / "sample-private.jsonl.gz")]
    assignment_paths = sorted((GLM_RUN_ROOT / "assignments" / "*.jsonl.gz").glob(), key=str)
    assignments = [Assignment(**row) for path in assignment_paths for row in read_jsonl(path)]
    taxonomy = read_json(str(GLM_RUN_ROOT / "taxonomy.json"))
    buckets = [Bucket(**row) for row in taxonomy["buckets"]]
    documents.sort(key=lambda row: row.sample_index)
    assignments.sort(key=lambda row: row.sample_index)
    indices = list(range(len(documents)))
    if [row.sample_index for row in documents] != indices or [row.sample_index for row in assignments] != indices:
        raise ValueError("The semantic sample indices are not complete and ordered")
    if [row.sample_index for row in documents] != [row.sample_index for row in assignments]:
        raise ValueError("The semantic documents and assignments are not aligned")
    return documents, assignments, buckets


def rows_by_source(documents: list[SampleDocument]) -> dict[str, list[SampleDocument]]:
    """Group selected rows by source."""
    output: dict[str, list[SampleDocument]] = defaultdict(list)
    for document in documents:
        output[document.source].append(document)
    return dict(output)


def select_stored_vectors(
    table: pa.Table,
    documents: list[SampleDocument],
    dimension: int,
    vector_function: Callable[[pa.Table, int], np.ndarray],
) -> dict[int, np.ndarray]:
    """Select stored vectors by evaluation rank and validate their hashes."""
    hashes = table["raw_sha256"].to_pylist()
    rows = stored_vector_rows(
        hashes,
        table["eval_rank"].to_pylist(),
        [(document.eval_rank, document.raw_sha256) for document in documents],
    )
    matrix = vector_function(table, dimension)
    return {document.sample_index: matrix[row] for document, row in zip(documents, rows, strict=True)}


def arctic_vectors(manifest: dict[str, Any], documents: list[SampleDocument]) -> np.ndarray:
    """Load Arctic vectors for the semantic sample."""
    selected: dict[int, np.ndarray] = {}
    for index, (source, source_documents) in enumerate(sorted(rows_by_source(documents).items()), start=1):
        logger.info("Loading Arctic source %d/%d: %s", index, len(manifest["sources"]), source)
        output_url = teacher_output_url(manifest["sources"][source]["output_url"])
        filesystem, path = fsspec.core.url_to_fs(output_url)
        table = pq.read_table(
            path,
            filesystem=filesystem,
            columns=["raw_sha256", "split", "eval_rank", "embedding"],
        )
        table = table.filter(pc.equal(table["split"], "eval")).drop(["split"])
        selected.update(
            select_stored_vectors(
                table,
                source_documents,
                TEACHER_EMBEDDING_DIMENSION,
                lambda value, dimension: normalized_candidate_vectors(quantized_vectors(value, dimension)),
            )
        )
    return np.stack([selected[index] for index in range(len(documents))])


def candidate_vectors(candidate_name: str, manifest: dict[str, Any], documents: list[SampleDocument]) -> np.ndarray:
    """Load one saved teacher candidate for the semantic sample."""
    candidate = CANDIDATES[candidate_name]
    selected: dict[int, np.ndarray] = {}
    source_groups = rows_by_source(documents)
    for index, (source, source_documents) in enumerate(sorted(source_groups.items()), start=1):
        logger.info("Loading %s source %d/%d: %s", candidate_name, index, len(source_groups), source)
        output_url = candidate_output_url(candidate, manifest["sources"][source]["output_url"])
        filesystem, path = fsspec.core.url_to_fs(output_url)
        table = pq.read_table(path, filesystem=filesystem)
        metadata = table.schema.metadata or {}
        if any(metadata.get(key) != value for key, value in expected_metadata(candidate, manifest["sha256"]).items()):
            raise ValueError(f"Stored {candidate_name} metadata differs for {source}")
        selected.update(
            select_stored_vectors(
                table,
                source_documents,
                candidate.embedding_dimension,
                lambda value, dimension: normalized_candidate_vectors(quantized_vectors(value, dimension)),
            )
        )
    return np.stack([selected[index] for index in range(len(documents))])


def local_model_vectors(documents: list[SampleDocument]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Embed the semantic sample with the saved local-inference models."""
    texts = [document.text for document in documents]
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        baseline = Embedder.load(baseline_path)
        student, training_report = load_student("full", "full", "3m", directory)
        qwen_student, qwen_training_report = load_student("full", "full-qwen3-06b-1024-crossdim", "750k", directory)
        vectors = {
            "luxical_one": normalize_embeddings(baseline(texts, batch_size=4_096)),
            "fast_arctic_3m": normalize_embeddings(student(texts, batch_size=4_096)),
            "fast_qwen_crossdim_750k": normalize_embeddings(qwen_student(texts, batch_size=4_096)),
        }
    metadata = {
        "luxical_one": {
            "repo": BASELINE_REPO,
            "file": BASELINE_FILE,
            "revision": BASELINE_REVISION,
        },
        "fast_arctic_3m": training_report,
        "fast_qwen_crossdim_750k": qwen_training_report,
    }
    return vectors, metadata


def gallery_indices(assignments: list[Assignment], count: int) -> list[int]:
    """Select one stable-hash document from each primary bucket."""
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for assignment in assignments:
        by_bucket[assignment.primary_bucket_id].append(assignment.sample_index)
    selected = []
    for bucket_id in sorted(by_bucket, key=lambda value: hash_bytes(value)):
        selected.append(min(by_bucket[bucket_id], key=lambda value: hash_bytes(str(value))))
        if len(selected) == count:
            break
    return selected


def hash_bytes(value: str) -> bytes:
    """Return a stable order key."""
    return hashlib.sha256(f"{SEED}:{value}".encode()).digest()


def neighbor_gallery(
    models: dict[str, np.ndarray],
    neighbors: dict[str, np.ndarray],
    documents: list[SampleDocument],
    assignments: list[Assignment],
) -> list[dict[str, Any]]:
    """Return private examples for direct nearest-neighbor inspection."""
    output = []
    for sample_index in gallery_indices(assignments, GALLERY_DOCUMENTS):
        query = assignments[sample_index]
        row = {
            "sample_index": sample_index,
            "text": documents[sample_index].text[:500],
            "labels": [query.primary_bucket_id, *query.secondary_bucket_ids],
            "models": {},
        }
        for name in models:
            row["models"][name] = [
                {
                    "sample_index": int(neighbor),
                    "text": documents[int(neighbor)].text[:500],
                    "labels": [
                        assignments[int(neighbor)].primary_bucket_id,
                        *assignments[int(neighbor)].secondary_bucket_ids,
                    ],
                }
                for neighbor in neighbors[name][sample_index, :GALLERY_NEIGHBORS]
            ]
        output.append(row)
    return output


def report_html(report: dict[str, Any]) -> str:
    """Return a private single-page semantic screen report."""
    metric_names = (
        "neighbor_any_label_fraction",
        "neighbor_label_jaccard",
        "nearest_primary_macro_f1",
        "cluster_nmi",
        "cluster_purity",
        "effective_rank",
    )
    rows = []
    for name, metrics in report["models"].items():
        values = "".join(f"<td>{float(metrics[metric]):.4f}</td>" for metric in metric_names)
        rows.append(f"<tr><td>{html.escape(name)}</td>{values}</tr>")
    gallery = []
    for item in report["gallery"]:
        model_sections = []
        for name, neighbors in item["models"].items():
            neighbor_items = "".join(
                f"<li><code>{html.escape(', '.join(row['labels']))}</code> {html.escape(row['text'])}</li>"
                for row in neighbors
            )
            model_sections.append(f"<h4>{html.escape(name)}</h4><ol>{neighbor_items}</ol>")
        gallery.append(
            f"<details><summary><code>{html.escape(', '.join(item['labels']))}</code> "
            f"{html.escape(item['text'])}</summary>{''.join(model_sections)}</details>"
        )
    headings = "".join(f"<th>{html.escape(name)}</th>" for name in metric_names)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Semantic embedding screen</title>
<style>body{{font-family:sans-serif;margin:2rem;max-width:100rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:.35rem}}details{{margin:1rem 0}}li{{margin:.5rem 0}}</style></head>
<body><h1>Semantic embedding screen</h1>
<p>This private report compares source-blind semantic neighbors on {report['documents']} documents.</p>
<table><thead><tr><th>Model</th>{headings}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Nearest-neighbor gallery</h2>{''.join(gallery)}</body></html>"""


def write_report(report: dict[str, Any]) -> tuple[str, str]:
    """Write private JSON and HTML reports."""
    json_url = str(OUTPUT_ROOT / "report.json")
    html_url = str(OUTPUT_ROOT / "report.html")
    for url, payload in ((json_url, json.dumps(report, indent=2, sort_keys=True)), (html_url, report_html(report))):
        filesystem, path = fsspec.core.url_to_fs(url)
        with atomic_rename(path, fs=filesystem) as temporary_path:
            with filesystem.open(temporary_path, "w") as file:
                file.write(payload)
    return json_url, html_url


def main() -> None:
    documents, assignments, buckets = semantic_sample()
    manifest = read_json(MANIFEST_URL)
    models, metadata = local_model_vectors(documents)
    models["arctic_medium"] = arctic_vectors(manifest, documents)
    models["qwen3_embedding_0.6b"] = candidate_vectors("qwen3-embedding-0.6b", manifest, documents)
    models["lfm2.5_embedding_350m"] = candidate_vectors("lfm2.5-embedding-350m", manifest, documents)
    metadata["arctic_medium"] = {"vector_root": "teacher-arctic-v1"}
    metadata["qwen3_embedding_0.6b"] = asdict(CANDIDATES["qwen3-embedding-0.6b"])
    metadata["lfm2.5_embedding_350m"] = asdict(CANDIDATES["lfm2.5-embedding-350m"])

    primary_labels = np.asarray([assignment.primary_bucket_id for assignment in assignments])
    label_sets = [
        frozenset((assignment.primary_bucket_id, *assignment.secondary_bucket_ids)) for assignment in assignments
    ]
    model_metrics = {}
    model_neighbors = {}
    for name, vectors in models.items():
        logger.info("Measuring semantic coherence for %s", name)
        metrics, neighbors = semantic_metrics(
            vectors,
            primary_labels,
            label_sets,
            neighbor_count=NEIGHBOR_COUNT,
            cluster_count=len(buckets),
            seed=SEED,
        )
        model_metrics[name] = metrics
        model_neighbors[name] = neighbors
    qwen_vectors = models["qwen3_embedding_0.6b"]
    for name, vectors in models.items():
        model_metrics[name]["qwen_cosine_order_fidelity"] = cosine_order_fidelity(vectors, qwen_vectors)

    speed_report = read_json(FAST_STUDENT_REPORT_URL)
    speed_ratio = float(speed_report["comparison"]["speed_ratio"])
    gates = student_gates(model_metrics["fast_arctic_3m"], model_metrics["qwen3_embedding_0.6b"], speed_ratio)
    report = {
        "documents": len(documents),
        "taxonomy_buckets": len(buckets),
        "semantic_run_root": str(GLM_RUN_ROOT),
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "source_metadata_used_for_alignment_only": True,
        "neighbor_count": NEIGHBOR_COUNT,
        "models": model_metrics,
        "model_metadata": metadata,
        "fast_arctic_3m_cpu_speed_ratio": speed_ratio,
        "fast_arctic_3m_gates_against_qwen": gates,
        "fast_arctic_3m_all_screen_gates_passed": all(gates.values()),
        "gallery": neighbor_gallery(models, model_neighbors, documents, assignments),
    }
    json_url, html_url = write_report(report)
    summary = {
        "json_url": json_url,
        "html_url": html_url,
        "models": {name: metrics for name, metrics in model_metrics.items()},
        "fast_arctic_3m_gates_against_qwen": gates,
        "fast_arctic_3m_all_screen_gates_passed": all(gates.values()),
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("SEMANTIC_EMBEDDING_SCREEN=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
