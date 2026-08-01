# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed and evaluate the fixed holdout with Arctic Embed Large v2.0."""

import html
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from arctic import PinnedArcticEmbedder
from evaluate_ladder import (
    BASELINE_FILE,
    BASELINE_REPO,
    BASELINE_REVISION,
    BOOTSTRAP_SAMPLES,
    CLUSTER_COUNT,
    CLUSTER_MAX_SOURCE_SHARE,
    CLUSTER_SEEDS,
    CPU_THREADS,
    EVALUATION_ROOT,
    MANIFEST_URL,
    MIN_EFFECTIVE_RANK_RATIO,
    MIN_UNIQUE_FRACTION,
    MIN_VARIANCE_RATIO,
    PROBE_TRAIN_ROWS_PER_SOURCE,
    QUALITY_DELTA,
    TEACHER_EMBEDDING_DIMENSION,
    TEACHER_QUANTIZATION_LIMIT,
    model_metrics,
    pair_indices,
    read_json,
    representation_comparison,
    teacher_comparison_report,
    teacher_output_url,
    vector_metrics,
)
from evaluate_teacher import add_source_details, failure_summary, source_categories
from huggingface_hub import hf_hub_download
from ladder_config import (
    LARGE_TEACHER_ID,
    LARGE_TEACHER_REVISION,
    MANIFEST_ROOT,
    PREDECLARED_OOD_SOURCES,
    SEED,
    teacher_windows_from_view,
)
from luxical.embedder import Embedder
from luxical.teacher_embedder import fast_8bit_uniform_scalar_quantize
from luxical.training import dequantize_8bit_uniform_scalar_quantized
from rigging.filesystem import atomic_rename
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

OUTPUT_NAME = "teacher-arctic-l-v2.0-v2"
LARGE_TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-arctic-l-v2.0-eval-v2"
RESULT_FILE = Path("/tmp/luxical-arctic-large-teacher-evaluation")
MAX_TEACHER_TOKENS = 512
INFERENCE_BATCH_SIZE = 64
EXPECTED_EVALUATION_ROWS = 74_752
WINDOWS_PER_DOCUMENT = 3
ATTENTION_IMPLEMENTATION = "eager"
POOLING_IMPLEMENTATION = "arctic.PinnedArcticEmbedder._embed_batch"
LOG_CHUNK_CHARACTERS = 2_000

MANIFEST_METADATA_KEY = b"luxical_manifest_sha256"
TEACHER_ID_METADATA_KEY = b"luxical_teacher_id"
TEACHER_REVISION_METADATA_KEY = b"luxical_teacher_revision"
TEACHER_SCOPE_METADATA_KEY = b"luxical_teacher_scope"
TEACHER_MAX_TOKENS_METADATA_KEY = b"luxical_teacher_max_tokens"
TEACHER_WINDOWS_METADATA_KEY = b"luxical_teacher_windows_per_document"
TEACHER_DIMENSION_METADATA_KEY = b"luxical_teacher_embedding_dimension"
TEACHER_QUANTIZATION_METADATA_KEY = b"luxical_teacher_quantization_limit"
TEACHER_ATTENTION_METADATA_KEY = b"luxical_teacher_attention_implementation"
TEACHER_POOLING_METADATA_KEY = b"luxical_teacher_pooling_implementation"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class ArcticLargeEmbedder(PinnedArcticEmbedder):
    """Run one exact Arctic Embed Large v2.0 checkpoint."""

    HF_MODEL_ID = LARGE_TEACHER_ID
    EMBEDDING_DIM = 1024

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(
            LARGE_TEACHER_ID,
            revision=LARGE_TEACHER_REVISION,
        )
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise TypeError(f"Expected a fast tokenizer, got {type(self.tokenizer).__name__}")
        self.model = AutoModel.from_pretrained(
            LARGE_TEACHER_ID,
            revision=LARGE_TEACHER_REVISION,
            add_pooling_layer=False,
            dtype=torch.float32,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )
        self.device: str | torch.device = "cpu"
        self.max_seq_len = MAX_TEACHER_TOKENS
        self.to("cuda", dtype=torch.float32)
        self.model.eval()
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"Arctic Large parameter {name} contains non-finite values")

    def quantized_documents(self, texts: list[str]) -> np.ndarray:
        """Return pooled document vectors in the training storage format."""
        windows = [window for text in texts for window in teacher_windows_from_view(text)]
        window_vectors = self.embed_texts(
            windows,
            is_query=False,
            batch_size=INFERENCE_BATCH_SIZE,
            mrl=True,
            progress_bar=False,
        ).reshape(
            len(texts),
            WINDOWS_PER_DOCUMENT,
            TEACHER_EMBEDDING_DIMENSION,
        )
        if not np.isfinite(window_vectors).all():
            raise ValueError("Arctic Large returned non-finite window vectors")
        pooled = window_vectors.mean(axis=1)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
        quantized = fast_8bit_uniform_scalar_quantize(pooled, TEACHER_QUANTIZATION_LIMIT)
        if quantized.shape != (len(texts), TEACHER_EMBEDDING_DIMENSION):
            raise ValueError(f"Arctic Large returned an unexpected shape: {quantized.shape}")
        return quantized


def expected_metadata(manifest_sha256: str) -> dict[bytes, bytes]:
    """Return metadata that binds one vector file to the fixed inputs."""
    return {
        MANIFEST_METADATA_KEY: manifest_sha256.encode(),
        TEACHER_ID_METADATA_KEY: LARGE_TEACHER_ID.encode(),
        TEACHER_REVISION_METADATA_KEY: LARGE_TEACHER_REVISION.encode(),
        TEACHER_SCOPE_METADATA_KEY: b"evaluation-only",
        TEACHER_MAX_TOKENS_METADATA_KEY: str(MAX_TEACHER_TOKENS).encode(),
        TEACHER_WINDOWS_METADATA_KEY: str(WINDOWS_PER_DOCUMENT).encode(),
        TEACHER_DIMENSION_METADATA_KEY: str(TEACHER_EMBEDDING_DIMENSION).encode(),
        TEACHER_QUANTIZATION_METADATA_KEY: str(TEACHER_QUANTIZATION_LIMIT).encode(),
        TEACHER_ATTENTION_METADATA_KEY: ATTENTION_IMPLEMENTATION.encode(),
        TEACHER_POOLING_METADATA_KEY: POOLING_IMPLEMENTATION.encode(),
    }


def evaluation_table(url: str) -> pa.Table:
    """Load the fixed evaluation rows for one source."""
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(
        path,
        filesystem=filesystem,
        columns=["raw_sha256", "source", "source_category", "split", "eval_rank", "text"],
    )
    return table.filter(pc.equal(table["split"], "eval"))


def quantized_vectors(table: pa.Table) -> np.ndarray:
    """Return one quantized embedding column as a matrix."""
    return (
        table["embedding"]
        .combine_chunks()
        .values.to_numpy(zero_copy_only=False)
        .reshape(len(table), TEACHER_EMBEDDING_DIMENSION)
    )


def normalized_vectors(quantized: np.ndarray) -> np.ndarray:
    """Return normalized float vectors from the training storage format."""
    vectors = dequantize_8bit_uniform_scalar_quantized(quantized, TEACHER_QUANTIZATION_LIMIT)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    if not np.isfinite(vectors).all():
        raise ValueError("Dequantized teacher vectors contain non-finite values")
    return vectors


def large_output_url(manifest_output_url: str) -> str:
    """Return the large-teacher output paired with one manifest source."""
    return f"{LARGE_TEACHER_ROOT}/sources/{Path(manifest_output_url).name}"


def load_or_embed_source(
    teacher: ArcticLargeEmbedder,
    source_table: pa.Table,
    manifest_output_url: str,
    manifest_sha256: str,
) -> tuple[np.ndarray, bool, float]:
    """Load or create one aligned large-teacher source file."""
    output_url = large_output_url(manifest_output_url)
    filesystem, path = fsspec.core.url_to_fs(output_url)
    if filesystem.exists(path):
        output_table = pq.read_table(path, filesystem=filesystem)
        metadata = output_table.schema.metadata or {}
        if any(metadata.get(key) != value for key, value in expected_metadata(manifest_sha256).items()):
            raise ValueError(f"Existing Arctic Large output has different metadata: {output_url}")
        if source_table["raw_sha256"].to_pylist() != output_table["raw_sha256"].to_pylist():
            raise ValueError(f"Existing Arctic Large output is not aligned: {output_url}")
        return quantized_vectors(output_table), True, 0.0

    started = time.perf_counter()
    quantized = teacher.quantized_documents(source_table["text"].to_pylist())
    embedding_duration = time.perf_counter() - started
    embedding_array = pa.FixedSizeListArray.from_arrays(
        pa.array(quantized.ravel()),
        TEACHER_EMBEDDING_DIMENSION,
    )
    output_table = source_table.drop(["text"]).append_column("embedding", embedding_array)
    metadata = dict(output_table.schema.metadata or {})
    metadata.update(expected_metadata(manifest_sha256))
    output_table = output_table.replace_schema_metadata(metadata)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        pq.write_table(output_table, temporary_path, filesystem=filesystem, compression="zstd")
    return quantized, False, embedding_duration


def medium_vectors(manifest_output_url: str, expected_hashes: list[str]) -> np.ndarray:
    """Load aligned medium-teacher evaluation vectors."""
    url = teacher_output_url(manifest_output_url)
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(
        path,
        filesystem=filesystem,
        columns=["raw_sha256", "split", "embedding"],
    )
    table = table.filter(pc.equal(table["split"], "eval"))
    if table["raw_sha256"].to_pylist() != expected_hashes:
        raise ValueError(f"Arctic Medium output is not aligned: {url}")
    return normalized_vectors(quantized_vectors(table))


def comparison_rows(report: dict[str, Any]) -> str:
    """Render probe values for the three representations."""
    rows = []
    for name in ("luxical_one", "arctic_medium", "arctic_large"):
        probe = report[name]["probe"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{probe['macro_f1']:.5f}</td>"
            f"<td>{probe['category_macro_f1']['code']:.5f}</td>"
            f"<td>{probe['category_macro_f1']['multilingual']:.5f}</td>"
            f"<td>{probe['category_macro_f1']['standard']:.5f}</td>"
            "</tr>"
        )
    return "".join(rows)


def html_report(report: dict[str, Any]) -> str:
    """Render a standalone large-teacher report."""
    comparison = report["large_vs_luxical"]
    gate_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in comparison["gates"].items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Arctic Embed Large v2.0 evaluation</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 90rem; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #bbb; padding: .35rem .6rem; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f5f5; padding: 1rem; }}
</style></head><body>
<h1>Arctic Embed Large v2.0 evaluation</h1>
<p>All required gates: {'PASS' if comparison['all_required_gates_passed'] else 'FAIL'}</p>
<table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Source probe</h2>
<table><thead><tr><th>Representation</th><th>Overall</th><th>Code</th><th>Multilingual</th><th>Standard</th></tr></thead>
<tbody>{comparison_rows(report)}</tbody></table>
<details><summary>Complete JSON</summary><pre>{html.escape(json.dumps(report, indent=2, sort_keys=True))}</pre></details>
</body></html>"""


def write_report(report: dict[str, Any]) -> tuple[str, str]:
    """Write JSON and HTML reports atomically."""
    json_url = f"{EVALUATION_ROOT}/{OUTPUT_NAME}/report.json"
    html_url = f"{EVALUATION_ROOT}/{OUTPUT_NAME}/report.html"
    for url, payload in (
        (json_url, json.dumps(report, indent=2, sort_keys=True)),
        (html_url, html_report(report)),
    ):
        filesystem, path = fsspec.core.url_to_fs(url)
        with atomic_rename(path, fs=filesystem) as temporary_path:
            with filesystem.open(temporary_path, "w") as file:
                file.write(payload)
    return json_url, html_url


def summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return compact probe and collapse metrics."""
    return {
        "macro_f1": metrics["probe"]["macro_f1"],
        "worst_source_recall": metrics["probe"]["worst_source_recall"],
        "category_macro_f1": metrics["probe"]["category_macro_f1"],
        "finite_fraction": metrics["collapse"]["finite_fraction"],
        "exact_unique_fraction": metrics["collapse"]["exact_unique_fraction"],
        "unique_fraction_4dp": metrics["collapse"]["unique_fraction_4dp"],
        "cluster_distribution": metrics["collapse"]["cluster_distribution"],
    }


@threadpool_limits.wrap(limits=CPU_THREADS)
def main() -> None:
    """Embed the fixed holdout and compare all three representations."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    manifest = read_json(MANIFEST_URL)
    teacher = ArcticLargeEmbedder()
    control_vectors = normalized_vectors(
        teacher.quantized_documents(
            [
                "A short English document about data processing.",
                "def add(left: int, right: int) -> int:\n    return left + right",
                "これは日本語の短い文書です。",
                "word " * 128,
            ]
        )
    )
    if np.unique(control_vectors, axis=0).shape[0] != len(control_vectors):
        raise ValueError("Arctic Large returned duplicate control vectors")

    texts: list[str] = []
    labels: list[str] = []
    categories: list[str] = []
    probe_roles: list[str] = []
    large_batches = []
    medium_batches = []
    embedded_rows = 0
    reused_rows = 0
    new_embedding_duration = 0.0
    source_loop_started = time.perf_counter()
    for index, (source, source_result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Loading evaluation source %d/%d: %s", index, len(manifest["sources"]), source)
        source_table = evaluation_table(source_result["output_url"])
        hashes = source_table["raw_sha256"].to_pylist()
        large_quantized, reused, source_embedding_duration = load_or_embed_source(
            teacher,
            source_table,
            source_result["output_url"],
            manifest["sha256"],
        )
        if reused:
            reused_rows += len(source_table)
        else:
            embedded_rows += len(source_table)
            new_embedding_duration += source_embedding_duration
        large_batches.append(normalized_vectors(large_quantized))
        medium_batches.append(medium_vectors(source_result["output_url"], hashes))
        texts.extend(source_table["text"].to_pylist())
        labels.extend(source_table["source"].to_pylist())
        categories.extend(source_table["source_category"].to_pylist())
        probe_roles.extend(
            "probe_train" if rank < PROBE_TRAIN_ROWS_PER_SOURCE else "probe_eval"
            for rank in source_table["eval_rank"].to_pylist()
        )

    source_loop_duration = time.perf_counter() - source_loop_started
    if len(texts) != EXPECTED_EVALUATION_ROWS:
        raise ValueError(f"Loaded {len(texts)} evaluation rows; expected {EXPECTED_EVALUATION_ROWS}")
    labels_array = np.asarray(labels)
    categories_array = np.asarray(categories)
    probe_roles_array = np.asarray(probe_roles)
    large_vectors = np.concatenate(large_batches)
    medium_teacher_vectors = np.concatenate(medium_batches)
    left, right = pair_indices(labels_array)

    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    baseline = Embedder.load(baseline_path)
    baseline_metrics = model_metrics(
        baseline,
        texts,
        labels_array,
        probe_roles_array,
        categories_array,
        large_vectors,
        left,
        right,
    )
    baseline_metrics["large_teacher_fidelity"] = baseline_metrics.pop("arctic_fidelity")
    medium_metrics = vector_metrics(medium_teacher_vectors, labels_array, probe_roles_array, categories_array)
    large_metrics = vector_metrics(large_vectors, labels_array, probe_roles_array, categories_array)
    categories_by_source = source_categories(labels_array, categories_array)

    large_vs_luxical = teacher_comparison_report(large_metrics, baseline_metrics)
    add_source_details(large_vs_luxical, categories_by_source)
    large_vs_medium = representation_comparison(large_metrics, medium_metrics)
    del large_vs_medium["collapse"]
    report = {
        "evaluation": OUTPUT_NAME,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "evaluation_rows": len(texts),
        "predeclared_ood_sources": sorted(PREDECLARED_OOD_SOURCES),
        "teacher": {
            "id": LARGE_TEACHER_ID,
            "revision": LARGE_TEACHER_REVISION,
            "root": LARGE_TEACHER_ROOT,
            "embedding_dimension": TEACHER_EMBEDDING_DIMENSION,
            "quantization_limit": TEACHER_QUANTIZATION_LIMIT,
            "maximum_tokens_per_window": MAX_TEACHER_TOKENS,
            "windows_per_document": WINDOWS_PER_DOCUMENT,
            "inference_dtype": "float32",
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "pooling_implementation": POOLING_IMPLEMENTATION,
        },
        "embedding_run": {
            "source_loop_duration_seconds": source_loop_duration,
            "new_embedding_duration_seconds": new_embedding_duration,
            "embedded_rows": embedded_rows,
            "reused_rows": reused_rows,
            "new_documents_per_second": embedded_rows / new_embedding_duration if embedded_rows else None,
        },
        "thresholds": {
            "minimum_unique_fraction": MIN_UNIQUE_FRACTION,
            "maximum_source_cluster_share": CLUSTER_MAX_SOURCE_SHARE,
            "cluster_count": CLUSTER_COUNT,
            "cluster_seeds": list(CLUSTER_SEEDS),
            "minimum_effective_rank_ratio": MIN_EFFECTIVE_RANK_RATIO,
            "minimum_variance_ratio": MIN_VARIANCE_RATIO,
            "minimum_quality_delta": QUALITY_DELTA,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "luxical_one": baseline_metrics,
        "arctic_medium": medium_metrics,
        "arctic_large": large_metrics,
        "large_vs_luxical": large_vs_luxical,
        "large_vs_medium": large_vs_medium,
        "failure_summary": failure_summary(large_vs_luxical),
    }
    json_url, html_url = write_report(report)
    summary = {
        "json_url": json_url,
        "html_url": html_url,
        "evaluation_rows": len(texts),
        "all_required_gates_passed": large_vs_luxical["all_required_gates_passed"],
        "failed_gates": [name for name, passed in large_vs_luxical["gates"].items() if not passed],
        "failure_summary": report["failure_summary"],
        "embedding_run": report["embedding_run"],
        "luxical_one": summary_metrics(baseline_metrics),
        "arctic_medium": summary_metrics(medium_metrics),
        "arctic_large": summary_metrics(large_metrics),
        "large_vs_luxical": {
            "macro_f1_delta": large_vs_luxical["macro_f1_delta"],
            "worst_source_recall_delta": large_vs_luxical["worst_source_recall_delta"],
            "category_macro_f1_delta": large_vs_luxical["category_macro_f1_delta"],
            "probe_uncertainty": large_vs_luxical["probe_uncertainty"],
            "cluster_distribution_delta": large_vs_luxical["cluster_distribution_delta"],
        },
        "large_vs_medium": {
            "macro_f1_delta": large_vs_medium["macro_f1_delta"],
            "worst_source_recall_delta": large_vs_medium["worst_source_recall_delta"],
            "category_macro_f1_delta": large_vs_medium["category_macro_f1_delta"],
            "probe_uncertainty": large_vs_medium["probe_uncertainty"],
            "cluster_distribution_delta": large_vs_medium["cluster_distribution_delta"],
        },
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    serialized = json.dumps(summary, sort_keys=True)
    for index, start in enumerate(range(0, len(serialized), LOG_CHUNK_CHARACTERS)):
        chunk = serialized[start : start + LOG_CHUNK_CHARACTERS]
        logger.info("LUXICAL_ARCTIC_LARGE_TEACHER_CHUNK=%04d:%s", index, chunk)


if __name__ == "__main__":
    main()
