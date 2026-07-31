# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one Arctic-distilled Luxical ladder rung."""

import argparse
import hashlib
import html
import json
import logging
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from ladder_config import MANIFEST_ROOT, PREDECLARED_OOD_SOURCES, SEED, SourceCategory
from luxical.embedder import Embedder
from luxical.training import dequantize_8bit_uniform_scalar_quantized
from rigging.filesystem import atomic_rename
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, recall_score
from threadpoolctl import threadpool_limits

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-arctic-v1"
STUDENT_ROOT = f"{MANIFEST_ROOT}/students"
EVALUATION_ROOT = f"{MANIFEST_ROOT}/evaluation"
BASELINE_REPO = "DatologyAI/luxical-one"
BASELINE_FILE = "luxical_one_rc4.npz"
BASELINE_REVISION = "474cfeb959dd473b3d1cd61da630f566037e69e2"
PROBE_TRAIN_ROWS_PER_SOURCE = 256
CPU_THREADS = 8
EMBED_BATCH_SIZE = 4_096
SPEED_DOCUMENTS = 20_000
SPEED_WARMUP_DOCUMENTS = 1_024
SPEED_REPEATS = 5
CLUSTER_COUNT = 40
CLUSTER_MAX_SOURCE_SHARE = 0.90
MIN_UNIQUE_FRACTION = 0.99
MIN_EFFECTIVE_RANK_RATIO = 0.50
MIN_VARIANCE_RATIO = 0.50
QUALITY_DELTA = -0.02
SPEED_MINIMUM_RATIO = 0.70
SPEED_TARGET_RATIO = 0.85
PAIR_COUNT_WITHIN_SOURCE = 100_000
PAIR_COUNT_ACROSS_SOURCE = 100_000
TEACHER_QUANTIZATION_LIMIT = 0.3
TEACHER_EMBEDDING_DIMENSION = 256
RESULT_FILE = Path("/tmp/luxical-arctic-evaluation")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def teacher_output_url(manifest_output_url: str) -> str:
    """Return the teacher file paired with one manifest source file."""
    return f"{TEACHER_ROOT}/sources/{Path(manifest_output_url).name}"


def evaluation_mask(table: pa.Table) -> pa.Array | pa.ChunkedArray:
    """Select fixed probe-training and evaluation rows."""
    probe_train = pc.and_(
        pc.equal(table["split"], "train"),
        pc.less(table["train_rank"], PROBE_TRAIN_ROWS_PER_SOURCE),
    )
    return pc.or_(pc.equal(table["split"], "eval"), probe_train)


def fixed_evaluation_data(
    manifest: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned texts, labels, splits, categories, and teacher vectors."""
    texts = []
    labels = []
    splits = []
    categories = []
    teacher_batches = []
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Loading evaluation source %d/%d: %s", index, len(manifest["sources"]), source)
        manifest_filesystem, manifest_path = fsspec.core.url_to_fs(result["output_url"])
        source_table = pq.read_table(
            manifest_path,
            filesystem=manifest_filesystem,
            columns=["raw_sha256", "source", "source_category", "split", "train_rank", "text"],
        )
        source_table = source_table.filter(evaluation_mask(source_table))

        teacher_url = teacher_output_url(result["output_url"])
        teacher_filesystem, teacher_path = fsspec.core.url_to_fs(teacher_url)
        if not teacher_filesystem.exists(teacher_path):
            raise FileNotFoundError(f"Missing teacher output: {teacher_url}")
        teacher_table = pq.read_table(
            teacher_path,
            filesystem=teacher_filesystem,
            columns=["raw_sha256", "split", "train_rank", "embedding"],
        )
        teacher_table = teacher_table.filter(evaluation_mask(teacher_table))
        if source_table["raw_sha256"].to_pylist() != teacher_table["raw_sha256"].to_pylist():
            raise ValueError(f"Evaluation rows are not aligned for {source}")

        quantized = (
            teacher_table["embedding"]
            .combine_chunks()
            .values.to_numpy(zero_copy_only=False)
            .reshape(len(teacher_table), TEACHER_EMBEDDING_DIMENSION)
        )
        teacher = dequantize_8bit_uniform_scalar_quantized(
            quantized,
            TEACHER_QUANTIZATION_LIMIT,
        )
        teacher /= np.linalg.norm(teacher, axis=1, keepdims=True).clip(min=1e-12)
        teacher_batches.append(teacher)
        texts.extend(source_table["text"].to_pylist())
        labels.extend(source_table["source"].to_pylist())
        categories.extend(source_table["source_category"].to_pylist())
        splits.extend(source_table["split"].to_pylist())
    teacher_vectors = np.concatenate(teacher_batches)
    if not np.isfinite(teacher_vectors).all():
        raise ValueError("Evaluation teacher vectors contain non-finite values")
    return (
        texts,
        np.asarray(labels),
        np.asarray(splits),
        np.asarray(categories),
        teacher_vectors,
    )


def download_student(rung: str, directory: Path) -> Path:
    """Download one trained student model."""
    url = f"{STUDENT_ROOT}/{rung}/luxical-arctic.npz"
    training_report = read_json(f"{STUDENT_ROOT}/{rung}/training.json")
    filesystem, path = fsspec.core.url_to_fs(url)
    local_path = directory / f"luxical-arctic-{rung}.npz"
    filesystem.get(path, str(local_path))
    with local_path.open("rb") as model_file:
        actual_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    if actual_sha256 != training_report["model_sha256"]:
        raise ValueError(f"Student model digest is {actual_sha256}; expected {training_report['model_sha256']}")
    return local_path


def embed_on_cpu(model: Embedder, texts: list[str]) -> np.ndarray:
    """Embed fixed texts with a fixed CPU thread count."""
    torch.set_num_threads(CPU_THREADS)
    with threadpool_limits(limits=CPU_THREADS):
        vectors = model(texts, batch_size=EMBED_BATCH_SIZE)
    if not np.isfinite(vectors).all():
        raise ValueError("Model returned non-finite evaluation vectors")
    return vectors


def timed_embedding_rate(model: Embedder, texts: list[str]) -> tuple[float, float]:
    """Return elapsed time and throughput for one CPU embedding call."""
    started = time.perf_counter()
    embed_on_cpu(model, texts)
    duration = time.perf_counter() - started
    return duration, len(texts) / duration


def paired_speed_benchmark(
    baseline: Embedder,
    student: Embedder,
    texts: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return interleaved CPU throughput for the baseline and student."""
    if len(texts) < SPEED_DOCUMENTS:
        raise ValueError(f"Only {len(texts)} texts are available for the speed benchmark")
    speed_indices = np.linspace(0, len(texts) - 1, num=SPEED_DOCUMENTS, dtype=np.int64)
    speed_texts = [texts[index] for index in speed_indices]
    warmup_texts = speed_texts[:SPEED_WARMUP_DOCUMENTS]
    embed_on_cpu(baseline, warmup_texts)
    embed_on_cpu(student, warmup_texts)
    elapsed = {"baseline": [], "student": []}
    rates = {"baseline": [], "student": []}
    models = {"baseline": baseline, "student": student}
    for repeat in range(SPEED_REPEATS):
        order = ("baseline", "student") if repeat % 2 == 0 else ("student", "baseline")
        for name in order:
            duration, rate = timed_embedding_rate(models[name], speed_texts)
            elapsed[name].append(duration)
            rates[name].append(rate)

    def model_result(name: str) -> dict[str, Any]:
        return {
            "median_documents_per_second": float(np.median(rates[name])),
            "documents_per_second": rates[name],
            "elapsed_seconds": elapsed[name],
            "documents": len(speed_texts),
            "torch_threads": CPU_THREADS,
            "sampling": "evenly_spaced_across_fixed_evaluation_rows",
        }

    return model_result("baseline"), model_result("student")


def source_probe(
    vectors: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    """Fit and score a source-domain linear probe."""
    train_mask = splits == "train"
    eval_mask = splits == "eval"
    classifier = LogisticRegression(
        max_iter=300,
        random_state=SEED,
        solver="lbfgs",
    )
    classifier.fit(vectors[train_mask], labels[train_mask])
    predictions = classifier.predict(vectors[eval_mask])
    eval_labels = labels[eval_mask]
    eval_categories = categories[eval_mask]
    source_names = sorted(set(eval_labels))
    recalls = recall_score(
        eval_labels,
        predictions,
        labels=source_names,
        average=None,
        zero_division=0,
    )
    per_source_recall = dict(zip(source_names, map(float, recalls), strict=True))
    category_macro_f1 = {}
    for category in SourceCategory:
        mask = eval_categories == category.value
        if mask.any():
            category_sources = sorted(set(eval_labels[mask]))
            category_macro_f1[category.value] = float(
                f1_score(
                    eval_labels[mask],
                    predictions[mask],
                    labels=category_sources,
                    average="macro",
                    zero_division=0,
                )
            )
    return {
        "accuracy": float(accuracy_score(eval_labels, predictions)),
        "macro_f1": float(f1_score(eval_labels, predictions, average="macro", zero_division=0)),
        "worst_source_recall": min(per_source_recall.values()),
        "per_source_recall": per_source_recall,
        "category_macro_f1": category_macro_f1,
        "train_rows": int(train_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
        "source_count": len(source_names),
    }


def effective_rank(vectors: np.ndarray) -> tuple[float, float]:
    """Return total variance and effective covariance rank."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = np.square(singular_values) / max(1, len(vectors) - 1)
    total_variance = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total_variance, 1e-30)
    positive = probabilities > 0
    rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    return total_variance, rank


def cluster_distribution_metrics(
    clustering: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    """Measure global cluster balance and source information."""
    cluster_counts = np.bincount(clustering, minlength=CLUSTER_COUNT)
    nonempty_counts = cluster_counts[cluster_counts > 0]
    probabilities = nonempty_counts / nonempty_counts.sum()
    effective_cluster_count = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    category_metrics = {}
    for category in SourceCategory:
        mask = categories == category.value
        if not mask.any():
            continue
        counts = np.bincount(clustering[mask], minlength=CLUSTER_COUNT)
        probabilities = counts[counts > 0] / counts.sum()
        category_metrics[category.value] = {
            "rows": int(mask.sum()),
            "largest_cluster_share": float(counts.max() / counts.sum()),
            "effective_cluster_count": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
            "source_cluster_nmi": float(normalized_mutual_info_score(labels[mask], clustering[mask])),
            "distinct_dominant_source_clusters": len(
                {
                    int(np.bincount(clustering[labels == source], minlength=CLUSTER_COUNT).argmax())
                    for source in set(labels[mask])
                }
            ),
        }
    return {
        "cluster_counts": cluster_counts.tolist(),
        "nonempty_cluster_count": len(nonempty_counts),
        "largest_cluster_share": float(nonempty_counts.max() / nonempty_counts.sum()),
        "largest_to_smallest_nonempty_ratio": float(nonempty_counts.max() / nonempty_counts.min()),
        "effective_cluster_count": effective_cluster_count,
        "source_cluster_nmi": float(normalized_mutual_info_score(labels, clustering)),
        "categories": category_metrics,
    }


def collapse_metrics(
    vectors: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    """Measure unique vectors, source concentration, variance, and rank."""
    eval_vectors = vectors[splits == "eval"]
    eval_labels = labels[splits == "eval"]
    eval_categories = categories[splits == "eval"]
    rounded = np.round(eval_vectors, decimals=4)
    unique_fraction = float(np.unique(rounded, axis=0).shape[0] / len(rounded))
    exact_unique_fraction = float(np.unique(eval_vectors, axis=0).shape[0] / len(eval_vectors))
    clustering = MiniBatchKMeans(
        n_clusters=CLUSTER_COUNT,
        random_state=SEED,
        batch_size=4_096,
        n_init=10,
    ).fit_predict(eval_vectors)
    per_source = {}
    for source in sorted(set(eval_labels)):
        mask = eval_labels == source
        source_vectors = eval_vectors[mask]
        counts = Counter(clustering[mask])
        total_variance, rank = effective_rank(source_vectors)
        per_source[source] = {
            "rows": int(mask.sum()),
            "unique_fraction_4dp": float(
                np.unique(np.round(source_vectors, decimals=4), axis=0).shape[0] / len(source_vectors)
            ),
            "largest_cluster_share": max(counts.values()) / int(mask.sum()),
            "total_variance": total_variance,
            "effective_rank": rank,
        }
    return {
        "finite_fraction": float(np.isfinite(eval_vectors).all(axis=1).mean()),
        "exact_unique_fraction": exact_unique_fraction,
        "unique_fraction_4dp": unique_fraction,
        "cluster_count": CLUSTER_COUNT,
        "cluster_distribution": cluster_distribution_metrics(
            clustering,
            eval_labels,
            eval_categories,
        ),
        "per_source": per_source,
    }


def pair_indices(labels: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed within-source and across-source evaluation pairs."""
    eval_indices = np.flatnonzero(splits == "eval")
    by_source: dict[str, np.ndarray] = {
        source: eval_indices[labels[eval_indices] == source] for source in sorted(set(labels[eval_indices]))
    }
    rng = np.random.default_rng(SEED)
    sources = sorted(by_source)
    within_left = []
    within_right = []
    for _ in range(PAIR_COUNT_WITHIN_SOURCE):
        source = sources[int(rng.integers(len(sources)))]
        selected = rng.choice(by_source[source], size=2, replace=False)
        within_left.append(selected[0])
        within_right.append(selected[1])
    across_left = []
    across_right = []
    for _ in range(PAIR_COUNT_ACROSS_SOURCE):
        source_indices = rng.choice(len(sources), size=2, replace=False)
        left_source = sources[int(source_indices[0])]
        right_source = sources[int(source_indices[1])]
        across_left.append(rng.choice(by_source[left_source]))
        across_right.append(rng.choice(by_source[right_source]))
    return (
        np.asarray(within_left + across_left),
        np.asarray(within_right + across_right),
    )


def cosine_fidelity(
    vectors: np.ndarray,
    teacher_vectors: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    """Measure pairwise cosine-order fidelity to Arctic."""
    model_cosines = np.sum(vectors[left] * vectors[right], axis=1)
    teacher_cosines = np.sum(teacher_vectors[left] * teacher_vectors[right], axis=1)
    overall = spearmanr(model_cosines, teacher_cosines).statistic
    within = spearmanr(
        model_cosines[:PAIR_COUNT_WITHIN_SOURCE],
        teacher_cosines[:PAIR_COUNT_WITHIN_SOURCE],
    ).statistic
    across = spearmanr(
        model_cosines[PAIR_COUNT_WITHIN_SOURCE:],
        teacher_cosines[PAIR_COUNT_WITHIN_SOURCE:],
    ).statistic
    return {
        "spearman": float(overall),
        "within_source_spearman": float(within),
        "across_source_spearman": float(across),
        "pair_count": len(left),
    }


def model_metrics(
    model: Embedder,
    texts: list[str],
    labels: np.ndarray,
    splits: np.ndarray,
    categories: np.ndarray,
    teacher_vectors: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return all fixed metrics and model vectors."""
    vectors = embed_on_cpu(model, texts)
    return (
        {
            "probe": source_probe(vectors, labels, splits, categories),
            "collapse": collapse_metrics(vectors, labels, splits, categories),
            "arctic_fidelity": cosine_fidelity(vectors, teacher_vectors, left, right),
        },
        vectors,
    )


def collapse_comparison(
    student: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare per-source student collapse metrics with Luxical-One."""
    source_results = {}
    regular_failures = []
    ood_failures = []
    for source, metrics in student["per_source"].items():
        baseline_metrics = baseline["per_source"][source]
        rank_ratio = metrics["effective_rank"] / max(baseline_metrics["effective_rank"], 1e-30)
        variance_ratio = metrics["total_variance"] / max(baseline_metrics["total_variance"], 1e-30)
        passed = (
            metrics["largest_cluster_share"] <= CLUSTER_MAX_SOURCE_SHARE
            and metrics["unique_fraction_4dp"] >= MIN_UNIQUE_FRACTION
            and rank_ratio >= MIN_EFFECTIVE_RANK_RATIO
            and variance_ratio >= MIN_VARIANCE_RATIO
        )
        source_results[source] = metrics | {
            "effective_rank_ratio": rank_ratio,
            "variance_ratio": variance_ratio,
            "passed": passed,
            "ood_exception": source in PREDECLARED_OOD_SOURCES,
        }
        if not passed:
            if source in PREDECLARED_OOD_SOURCES:
                ood_failures.append(source)
            else:
                regular_failures.append(source)
    return {
        "source_results": source_results,
        "regular_failures": regular_failures,
        "ood_failures": ood_failures,
        "regular_source_gate_passed": not regular_failures,
        "overall_unique_gate_passed": student["unique_fraction_4dp"] >= MIN_UNIQUE_FRACTION,
        "finite_gate_passed": student["finite_fraction"] == 1.0,
    }


def comparison_report(student: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Return all pass gates for one student."""
    collapse = collapse_comparison(student["collapse"], baseline["collapse"])
    speed_ratio = student["speed"]["median_documents_per_second"] / baseline["speed"]["median_documents_per_second"]
    macro_f1_delta = student["probe"]["macro_f1"] - baseline["probe"]["macro_f1"]
    worst_recall_delta = student["probe"]["worst_source_recall"] - baseline["probe"]["worst_source_recall"]
    fidelity_delta = student["arctic_fidelity"]["spearman"] - baseline["arctic_fidelity"]["spearman"]
    student_distribution = student["collapse"]["cluster_distribution"]
    baseline_distribution = baseline["collapse"]["cluster_distribution"]
    required_categories = (
        SourceCategory.CODE.value,
        SourceCategory.MULTILINGUAL.value,
        SourceCategory.STANDARD.value,
    )
    category_macro_f1_delta = {
        category: student["probe"]["category_macro_f1"][category] - baseline["probe"]["category_macro_f1"][category]
        for category in required_categories
    }
    gates = {
        "finite": collapse["finite_gate_passed"],
        "unique": collapse["overall_unique_gate_passed"],
        "regular_source_collapse": collapse["regular_source_gate_passed"],
        "arctic_fidelity": fidelity_delta >= 0.0,
        "macro_f1": macro_f1_delta >= QUALITY_DELTA,
        "worst_source_recall": worst_recall_delta >= QUALITY_DELTA,
        "code_macro_f1": category_macro_f1_delta[SourceCategory.CODE.value] >= QUALITY_DELTA,
        "multilingual_macro_f1": category_macro_f1_delta[SourceCategory.MULTILINGUAL.value] >= QUALITY_DELTA,
        "standard_macro_f1": category_macro_f1_delta[SourceCategory.STANDARD.value] >= QUALITY_DELTA,
        "cpu_speed_minimum": speed_ratio >= SPEED_MINIMUM_RATIO,
    }
    return {
        "speed_ratio": speed_ratio,
        "speed_target_passed": speed_ratio >= SPEED_TARGET_RATIO,
        "macro_f1_delta": macro_f1_delta,
        "worst_source_recall_delta": worst_recall_delta,
        "arctic_fidelity_delta": fidelity_delta,
        "category_macro_f1_delta": category_macro_f1_delta,
        "cluster_distribution_delta": {
            "largest_cluster_share": (
                student_distribution["largest_cluster_share"] - baseline_distribution["largest_cluster_share"]
            ),
            "effective_cluster_count": (
                student_distribution["effective_cluster_count"] - baseline_distribution["effective_cluster_count"]
            ),
            "source_cluster_nmi": (
                student_distribution["source_cluster_nmi"] - baseline_distribution["source_cluster_nmi"]
            ),
        },
        "collapse": collapse,
        "gates": gates,
        "all_required_gates_passed": all(gates.values()),
    }


def html_report(report: dict[str, Any]) -> str:
    """Render a compact, standalone evaluation report."""
    comparison = report["comparison"]
    gate_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in comparison["gates"].items()
    )
    category_rows = "".join(
        f"<tr><td>{html.escape(category)}</td><td>{delta:.4f}</td></tr>"
        for category, delta in comparison["category_macro_f1_delta"].items()
    )
    failed_sources = comparison["collapse"]["regular_failures"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Luxical Arctic evaluation</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 90rem; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #bbb; padding: .35rem .6rem; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f5f5; padding: 1rem; }}
</style></head><body>
<h1>Luxical Arctic {html.escape(report["rung"])} evaluation</h1>
<p>All required gates: {'PASS' if comparison["all_required_gates_passed"] else 'FAIL'}</p>
<table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table>
<p>CPU speed ratio: {comparison["speed_ratio"]:.3f}</p>
<p>Macro-F1 delta: {comparison["macro_f1_delta"]:.4f}</p>
<p>Worst-source recall delta: {comparison["worst_source_recall_delta"]:.4f}</p>
<p>Arctic cosine Spearman delta: {comparison["arctic_fidelity_delta"]:.4f}</p>
<p>Largest global cluster share delta:
{comparison["cluster_distribution_delta"]["largest_cluster_share"]:.4f}</p>
<p>Effective cluster count delta:
{comparison["cluster_distribution_delta"]["effective_cluster_count"]:.4f}</p>
<p>Source-cluster NMI delta:
{comparison["cluster_distribution_delta"]["source_cluster_nmi"]:.4f}</p>
<h2>Category macro-F1 deltas</h2>
<table><thead><tr><th>Category</th><th>Student minus baseline</th></tr></thead>
<tbody>{category_rows}</tbody></table>
<p>Non-OOD collapse failures: {html.escape(", ".join(failed_sources) or "none")}</p>
<details><summary>Complete JSON</summary><pre>{html.escape(json.dumps(report, indent=2, sort_keys=True))}</pre></details>
</body></html>"""


def write_report(report: dict[str, Any], rung: str) -> tuple[str, str]:
    """Write JSON and HTML reports atomically."""
    json_url = f"{EVALUATION_ROOT}/{rung}/report.json"
    html_url = f"{EVALUATION_ROOT}/{rung}/report.html"
    for url, payload in (
        (json_url, json.dumps(report, indent=2, sort_keys=True)),
        (html_url, html_report(report)),
    ):
        filesystem, path = fsspec.core.url_to_fs(url)
        with atomic_rename(path, fs=filesystem) as temporary_path:
            with filesystem.open(temporary_path, "w") as file:
                file.write(payload)
    return json_url, html_url


def parse_args() -> argparse.Namespace:
    """Parse the ladder rung."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=("750k", "3m"), required=True)
    return parser.parse_args()


def main() -> None:
    """Evaluate one trained student against Luxical-One."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    arguments = parse_args()
    manifest = read_json(MANIFEST_URL)
    texts, labels, splits, categories, teacher_vectors = fixed_evaluation_data(manifest)
    left, right = pair_indices(labels, splits)
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        student_path = download_student(arguments.rung, Path(temporary_directory))
        baseline = Embedder.load(baseline_path)
        student = Embedder.load(student_path)
        baseline_metrics, _ = model_metrics(
            baseline,
            texts,
            labels,
            splits,
            categories,
            teacher_vectors,
            left,
            right,
        )
        student_metrics, _ = model_metrics(
            student,
            texts,
            labels,
            splits,
            categories,
            teacher_vectors,
            left,
            right,
        )
        baseline_speed, student_speed = paired_speed_benchmark(baseline, student, texts)
        baseline_metrics["speed"] = baseline_speed
        student_metrics["speed"] = student_speed
    comparison = comparison_report(student_metrics, baseline_metrics)
    report = {
        "rung": arguments.rung,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "predeclared_ood_sources": sorted(PREDECLARED_OOD_SOURCES),
        "present_ood_sources": sorted(set(manifest["sources"]) & PREDECLARED_OOD_SOURCES),
        "thresholds": {
            "minimum_unique_fraction": MIN_UNIQUE_FRACTION,
            "maximum_source_cluster_share": CLUSTER_MAX_SOURCE_SHARE,
            "minimum_effective_rank_ratio": MIN_EFFECTIVE_RANK_RATIO,
            "minimum_variance_ratio": MIN_VARIANCE_RATIO,
            "minimum_quality_delta": QUALITY_DELTA,
            "minimum_cpu_speed_ratio": SPEED_MINIMUM_RATIO,
            "target_cpu_speed_ratio": SPEED_TARGET_RATIO,
        },
        "baseline": baseline_metrics,
        "student": student_metrics,
        "comparison": comparison,
    }
    json_url, html_url = write_report(report, arguments.rung)
    regular_source_results = {
        source: metrics
        for source, metrics in comparison["collapse"]["source_results"].items()
        if not metrics["ood_exception"]
    }
    maximum_cluster_source, maximum_cluster_metrics = max(
        regular_source_results.items(),
        key=lambda item: item[1]["largest_cluster_share"],
    )
    summary = {
        "rung": arguments.rung,
        "json_url": json_url,
        "html_url": html_url,
        "all_required_gates_passed": comparison["all_required_gates_passed"],
        "failed_gates": [name for name, passed in comparison["gates"].items() if not passed],
        "regular_collapse_failures": comparison["collapse"]["regular_failures"],
        "ood_collapse_failures": comparison["collapse"]["ood_failures"],
        "maximum_regular_source_cluster_share": {
            "source": maximum_cluster_source,
            "share": maximum_cluster_metrics["largest_cluster_share"],
        },
        "minimum_regular_source_unique_fraction_4dp": min(
            metrics["unique_fraction_4dp"] for metrics in regular_source_results.values()
        ),
        "minimum_regular_source_effective_rank_ratio": min(
            metrics["effective_rank_ratio"] for metrics in regular_source_results.values()
        ),
        "minimum_regular_source_variance_ratio": min(
            metrics["variance_ratio"] for metrics in regular_source_results.values()
        ),
        "speed_ratio": comparison["speed_ratio"],
        "macro_f1_delta": comparison["macro_f1_delta"],
        "worst_source_recall_delta": comparison["worst_source_recall_delta"],
        "arctic_fidelity_delta": comparison["arctic_fidelity_delta"],
        "category_macro_f1_delta": comparison["category_macro_f1_delta"],
        "cluster_distribution_delta": comparison["cluster_distribution_delta"],
        "baseline": {
            "macro_f1": baseline_metrics["probe"]["macro_f1"],
            "worst_source_recall": baseline_metrics["probe"]["worst_source_recall"],
            "category_macro_f1": baseline_metrics["probe"]["category_macro_f1"],
            "arctic_fidelity_spearman": baseline_metrics["arctic_fidelity"]["spearman"],
            "cpu_documents_per_second": baseline_metrics["speed"]["median_documents_per_second"],
            "cluster_distribution": baseline_metrics["collapse"]["cluster_distribution"],
        },
        "student": {
            "macro_f1": student_metrics["probe"]["macro_f1"],
            "worst_source_recall": student_metrics["probe"]["worst_source_recall"],
            "category_macro_f1": student_metrics["probe"]["category_macro_f1"],
            "arctic_fidelity_spearman": student_metrics["arctic_fidelity"]["spearman"],
            "cpu_documents_per_second": student_metrics["speed"]["median_documents_per_second"],
            "finite_fraction": student_metrics["collapse"]["finite_fraction"],
            "exact_unique_fraction": student_metrics["collapse"]["exact_unique_fraction"],
            "unique_fraction_4dp": student_metrics["collapse"]["unique_fraction_4dp"],
            "cluster_distribution": student_metrics["collapse"]["cluster_distribution"],
        },
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_EVALUATION=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
