# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one fast-transformer student on the fixed Luxical gates."""

import argparse
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import equinox as eqx
import fsspec
import numpy as np
from evaluate_ladder import (
    BASELINE_FILE,
    BASELINE_REPO,
    BASELINE_REVISION,
    CLUSTER_COUNT,
    CLUSTER_MAX_SOURCE_SHARE,
    CLUSTER_SEEDS,
    MANIFEST_URL,
    MIN_EFFECTIVE_RANK_RATIO,
    MIN_UNIQUE_FRACTION,
    MIN_VARIANCE_RATIO,
    PREDECLARED_OOD_SOURCES,
    QUALITY_DELTA,
    SPEED_MINIMUM_RATIO,
    SPEED_TARGET_RATIO,
    comparison_report,
    fixed_evaluation_data,
    html_report,
    model_metrics,
    pair_indices,
    read_json,
)
from fast_student import LUXICAL_TOKENIZER_NAME, FastStudent
from huggingface_hub import hf_hub_download
from ladder_config import MANIFEST_ROOT, SEED
from luxical.embedder import Embedder
from rigging.filesystem import atomic_rename

TRAINING_ROOT = f"{MANIFEST_ROOT}/fast-student"
EVALUATION_ROOT = f"{MANIFEST_ROOT}/evaluation/fast-student"
SPEED_REPORT_URL = f"{MANIFEST_ROOT}/fast-student/speed/cpu-full-luxical-one-arrow.json"
RESULT_FILE = Path("/tmp/luxical-fast-student-evaluation")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def download(url: str, local_path: Path) -> None:
    """Download one private artifact."""
    filesystem, path = fsspec.core.url_to_fs(url)
    filesystem.get(path, str(local_path))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local file."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_student(config_name: str, rung: str, directory: Path) -> tuple[FastStudent, dict[str, Any]]:
    """Load and verify one trained fast student."""
    report = read_json(f"{TRAINING_ROOT}/{config_name}/{rung}/training.json")
    model_path = directory / "model.eqx"
    remap_path = directory / "raw-to-compact.npy"
    download(report["final_model_url"], model_path)
    download(report["raw_to_compact_url"], remap_path)
    if file_sha256(model_path) != report["final_model_sha256"]:
        raise ValueError("Downloaded model digest does not match the training report")
    if file_sha256(remap_path) != report["raw_to_compact_sha256"]:
        raise ValueError("Downloaded token remap digest does not match the training report")
    with remap_path.open("rb") as file:
        raw_to_compact = np.load(file)
    template = FastStudent.random(
        config_name,
        raw_to_compact,
        seed=SEED,
        tokenizer_name=LUXICAL_TOKENIZER_NAME,
    )
    model = eqx.tree_deserialise_leaves(model_path, template.model)
    return FastStudent(model, raw_to_compact, LUXICAL_TOKENIZER_NAME), report


def speed_metrics(report: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert the paired speed artifact to the fixed evaluation schema."""
    rates = report[f"{model}_rates"]
    durations = report[f"{model}_durations"]
    return {
        "median_documents_per_second": report[f"{model}_documents_per_second"],
        "documents_per_second": rates,
        "elapsed_seconds": durations,
        "documents": report["documents"],
        "sampling": "evenly_spaced_across_fixed_evaluation_rows",
    }


def write_report(report: dict[str, Any], config_name: str, rung: str) -> tuple[str, str]:
    """Write JSON and HTML reports atomically."""
    root = f"{EVALUATION_ROOT}/{config_name}/{rung}"
    json_url = f"{root}/report.json"
    html_url = f"{root}/report.html"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=("64k", "750k", "3m"), required=True)
    parser.add_argument("--config", choices=("full", "slim"), required=True)
    return parser.parse_args()


def main() -> None:
    """Run all fixed quality, collapse, fidelity, and speed gates."""
    arguments = parse_args()
    if arguments.config != "full":
        raise ValueError("A paired speed artifact is not available for the slim config")
    manifest = read_json(MANIFEST_URL)
    texts, labels, probe_roles, categories, teacher_vectors = fixed_evaluation_data(manifest)
    left, right = pair_indices(labels)
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        student, training_report = load_student(arguments.config, arguments.rung, Path(temporary_directory))
        baseline = Embedder.load(baseline_path)
        baseline_metrics = model_metrics(
            baseline,
            texts,
            labels,
            probe_roles,
            categories,
            teacher_vectors,
            left,
            right,
        )
        student_metrics = model_metrics(
            student,
            texts,
            labels,
            probe_roles,
            categories,
            teacher_vectors,
            left,
            right,
        )
    speed_report = read_json(SPEED_REPORT_URL)
    baseline_metrics["speed"] = speed_metrics(speed_report, "baseline")
    student_metrics["speed"] = speed_metrics(speed_report, "student")
    comparison = comparison_report(student_metrics, baseline_metrics)
    report = {
        "rung": f"fast-{arguments.config}-{arguments.rung}",
        "config_name": arguments.config,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "training_report_url": f"{TRAINING_ROOT}/{arguments.config}/{arguments.rung}/training.json",
        "training_report": training_report,
        "speed_report_url": SPEED_REPORT_URL,
        "predeclared_ood_sources": sorted(PREDECLARED_OOD_SOURCES),
        "thresholds": {
            "minimum_unique_fraction": MIN_UNIQUE_FRACTION,
            "maximum_source_cluster_share": CLUSTER_MAX_SOURCE_SHARE,
            "cluster_count": CLUSTER_COUNT,
            "cluster_seeds": list(CLUSTER_SEEDS),
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
    json_url, html_url = write_report(report, arguments.config, arguments.rung)
    summary = {
        "rung": arguments.rung,
        "config_name": arguments.config,
        "json_url": json_url,
        "html_url": html_url,
        "all_required_gates_passed": comparison["all_required_gates_passed"],
        "failed_gates": [name for name, passed in comparison["gates"].items() if not passed],
        "regular_collapse_failures": comparison["collapse"]["regular_failures"],
        "speed_ratio": comparison["speed_ratio"],
        "macro_f1_delta": comparison["macro_f1_delta"],
        "arctic_fidelity_delta": comparison["arctic_fidelity_delta"],
        "category_macro_f1_delta": comparison["category_macro_f1_delta"],
        "student_macro_f1": student_metrics["probe"]["macro_f1"],
        "student_arctic_fidelity": student_metrics["arctic_fidelity"]["spearman"],
        "student_finite_fraction": student_metrics["collapse"]["finite_fraction"],
        "student_unique_fraction_4dp": student_metrics["collapse"]["unique_fraction_4dp"],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("FAST_STUDENT_EVALUATION=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
