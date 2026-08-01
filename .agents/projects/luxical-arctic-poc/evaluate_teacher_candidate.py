# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed and evaluate one pinned alternative teacher on the fixed holdout."""

import argparse
import html
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as functional
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
from ladder_config import MANIFEST_ROOT, PREDECLARED_OOD_SOURCES, SEED, teacher_windows_from_view
from luxical.embedder import Embedder
from luxical.teacher_embedder import fast_8bit_uniform_scalar_quantize
from luxical.training import dequantize_8bit_uniform_scalar_quantized
from rigging.filesystem import atomic_rename
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

Pooling = Literal["cls", "last_token"]


@dataclass(frozen=True)
class Candidate:
    """Define one exact teacher checkpoint and its document encoding method."""

    name: str
    model_id: str
    revision: str
    prompt: str
    pooling: Pooling
    batch_size: int

    @property
    def output_name(self) -> str:
        """Return the stable evaluation directory name."""
        return f"teacher-{self.name}"

    @property
    def vector_root(self) -> str:
        """Return the source-vector storage root."""
        return f"{MANIFEST_ROOT}/{self.output_name}-eval-v1"


CANDIDATES = {
    "lfm2.5-embedding-350m": Candidate(
        name="lfm2.5-embedding-350m",
        model_id="LiquidAI/LFM2.5-Embedding-350M",
        revision="f35ae2c91d687658dbf1f2b449382f0b019b9808",
        prompt="document: ",
        pooling="cls",
        batch_size=96,
    ),
    "qwen3-embedding-0.6b": Candidate(
        name="qwen3-embedding-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        prompt="",
        pooling="last_token",
        batch_size=64,
    ),
}

RESULT_FILE_PREFIX = "/tmp/luxical-teacher-candidate"
MAX_TEACHER_TOKENS = 512
CANDIDATE_DIMENSION = 1_024
EXPECTED_EVALUATION_ROWS = 74_752
WINDOWS_PER_DOCUMENT = 3
INFERENCE_DTYPE = torch.float32
ATTENTION_IMPLEMENTATION = "eager"
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
TEACHER_PROMPT_METADATA_KEY = b"luxical_teacher_document_prompt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def noncausal_shortconv_forward(
    module: Lfm2ShortConv,
    hidden_states: torch.Tensor,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adapt the pinned bidirectional LFM convolution to the current interface."""
    del seq_idx
    return module.slow_forward(
        hidden_states,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
    )


class CandidateEmbedder:
    """Run one pinned alternative teacher on CUDA."""

    def __init__(self, candidate: Candidate) -> None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        self.candidate = candidate
        self.tokenizer = AutoTokenizer.from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            trust_remote_code=True,
            padding_side="left" if candidate.pooling == "last_token" else "right",
        )
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise TypeError(f"Expected a fast tokenizer, got {type(self.tokenizer).__name__}")
        self.model = AutoModel.from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            trust_remote_code=True,
            dtype=INFERENCE_DTYPE,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        ).to("cuda")
        if candidate.model_id.startswith("LiquidAI/LFM2.5-"):
            Lfm2ShortConv.forward = noncausal_shortconv_forward
        self.model.eval()
        hidden_size = int(self.model.config.hidden_size)
        if hidden_size != CANDIDATE_DIMENSION:
            raise ValueError(f"Teacher hidden size is {hidden_size}; expected {CANDIDATE_DIMENSION}")
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"Teacher parameter {name} contains non-finite values")

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return one vector for each tokenized window."""
        if self.candidate.pooling == "cls":
            return hidden[:, 0]
        if bool(attention_mask[:, -1].all()):
            return hidden[:, -1]
        row_indices = torch.arange(hidden.shape[0], device=hidden.device)
        token_indices = attention_mask.sum(dim=1) - 1
        return hidden[row_indices, token_indices]

    @torch.inference_mode()
    def embed_windows(self, texts: list[str]) -> np.ndarray:
        """Return normalized vectors for a list of document windows."""
        outputs = []
        for start in range(0, len(texts), self.candidate.batch_size):
            batch = [f"{self.candidate.prompt}{text}" for text in texts[start : start + self.candidate.batch_size]]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEACHER_TOKENS,
            )
            device_inputs = {name: value.to("cuda") for name, value in inputs.items()}
            model_output = self.model(**device_inputs, use_cache=False)
            vectors = self._pool(model_output.last_hidden_state, device_inputs["attention_mask"])
            vectors = functional.normalize(vectors.float(), p=2, dim=1)
            if not torch.isfinite(vectors).all():
                raise ValueError(f"Teacher returned non-finite vectors for batch {start}")
            outputs.append(vectors.cpu().numpy())
        return np.concatenate(outputs)

    def quantized_documents(self, texts: list[str]) -> np.ndarray:
        """Return pooled document vectors in the teacher storage format."""
        windows = [window for text in texts for window in teacher_windows_from_view(text)]
        window_vectors = self.embed_windows(windows).reshape(
            len(texts),
            WINDOWS_PER_DOCUMENT,
            CANDIDATE_DIMENSION,
        )
        pooled = window_vectors.mean(axis=1)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
        if not np.isfinite(pooled).all():
            raise ValueError("Teacher returned non-finite pooled vectors")
        quantized = fast_8bit_uniform_scalar_quantize(pooled, TEACHER_QUANTIZATION_LIMIT)
        if quantized.shape != (len(texts), CANDIDATE_DIMENSION):
            raise ValueError(f"Teacher returned an unexpected shape: {quantized.shape}")
        return quantized


def expected_metadata(candidate: Candidate, manifest_sha256: str) -> dict[bytes, bytes]:
    """Return metadata that binds one vector file to the fixed inputs."""
    return {
        MANIFEST_METADATA_KEY: manifest_sha256.encode(),
        TEACHER_ID_METADATA_KEY: candidate.model_id.encode(),
        TEACHER_REVISION_METADATA_KEY: candidate.revision.encode(),
        TEACHER_SCOPE_METADATA_KEY: b"evaluation-only",
        TEACHER_MAX_TOKENS_METADATA_KEY: str(MAX_TEACHER_TOKENS).encode(),
        TEACHER_WINDOWS_METADATA_KEY: str(WINDOWS_PER_DOCUMENT).encode(),
        TEACHER_DIMENSION_METADATA_KEY: str(CANDIDATE_DIMENSION).encode(),
        TEACHER_QUANTIZATION_METADATA_KEY: str(TEACHER_QUANTIZATION_LIMIT).encode(),
        TEACHER_ATTENTION_METADATA_KEY: ATTENTION_IMPLEMENTATION.encode(),
        TEACHER_POOLING_METADATA_KEY: candidate.pooling.encode(),
        TEACHER_PROMPT_METADATA_KEY: candidate.prompt.encode(),
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


def quantized_vectors(table: pa.Table, dimension: int) -> np.ndarray:
    """Return one quantized embedding column as a matrix."""
    return table["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(table), dimension)


def normalized_vectors(quantized: np.ndarray) -> np.ndarray:
    """Return normalized float vectors from the teacher storage format."""
    vectors = dequantize_8bit_uniform_scalar_quantized(quantized, TEACHER_QUANTIZATION_LIMIT)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    if not np.isfinite(vectors).all():
        raise ValueError("Dequantized teacher vectors contain non-finite values")
    return vectors


def candidate_output_url(candidate: Candidate, manifest_output_url: str) -> str:
    """Return the candidate output paired with one manifest source."""
    return f"{candidate.vector_root}/sources/{Path(manifest_output_url).name}"


def load_or_embed_source(
    embedder: CandidateEmbedder,
    source_table: pa.Table,
    manifest_output_url: str,
    manifest_sha256: str,
) -> tuple[np.ndarray, bool, float]:
    """Load or create one aligned candidate-teacher source file."""
    candidate = embedder.candidate
    output_url = candidate_output_url(candidate, manifest_output_url)
    filesystem, path = fsspec.core.url_to_fs(output_url)
    if filesystem.exists(path):
        output_table = pq.read_table(path, filesystem=filesystem)
        metadata = output_table.schema.metadata or {}
        if any(metadata.get(key) != value for key, value in expected_metadata(candidate, manifest_sha256).items()):
            raise ValueError(f"Existing teacher output has different metadata: {output_url}")
        if source_table["raw_sha256"].to_pylist() != output_table["raw_sha256"].to_pylist():
            raise ValueError(f"Existing teacher output is not aligned: {output_url}")
        return quantized_vectors(output_table, CANDIDATE_DIMENSION), True, 0.0

    started = time.perf_counter()
    quantized = embedder.quantized_documents(source_table["text"].to_pylist())
    embedding_duration = time.perf_counter() - started
    embedding_array = pa.FixedSizeListArray.from_arrays(pa.array(quantized.ravel()), CANDIDATE_DIMENSION)
    output_table = source_table.drop(["text"]).append_column("embedding", embedding_array)
    metadata = dict(output_table.schema.metadata or {})
    metadata.update(expected_metadata(candidate, manifest_sha256))
    output_table = output_table.replace_schema_metadata(metadata)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        pq.write_table(output_table, temporary_path, filesystem=filesystem, compression="zstd")
    return quantized, False, embedding_duration


def arctic_vectors(manifest_output_url: str, expected_hashes: list[str]) -> np.ndarray:
    """Load aligned Arctic Medium evaluation vectors."""
    url = teacher_output_url(manifest_output_url)
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(path, filesystem=filesystem, columns=["raw_sha256", "split", "embedding"])
    table = table.filter(pc.equal(table["split"], "eval"))
    if table["raw_sha256"].to_pylist() != expected_hashes:
        raise ValueError(f"Arctic Medium output is not aligned: {url}")
    return normalized_vectors(quantized_vectors(table, TEACHER_EMBEDDING_DIMENSION))


def regular_failures(comparison: dict[str, Any]) -> set[str]:
    """Return the regular source failures in one comparison."""
    return set(comparison["collapse"]["regular_failures"])


def failure_overlap(arctic_comparison: dict[str, Any], candidate_comparison: dict[str, Any]) -> dict[str, Any]:
    """Compare Arctic and candidate failure sets."""
    arctic = regular_failures(arctic_comparison)
    candidate = regular_failures(candidate_comparison)
    overlap = arctic & candidate
    union = arctic | candidate
    return {
        "arctic_failure_count": len(arctic),
        "candidate_failure_count": len(candidate),
        "overlap_count": len(overlap),
        "candidate_only_count": len(candidate - arctic),
        "arctic_only_count": len(arctic - candidate),
        "jaccard": len(overlap) / len(union) if union else 1.0,
        "overlap": sorted(overlap),
        "candidate_only": sorted(candidate - arctic),
        "arctic_only": sorted(arctic - candidate),
    }


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


def html_report(report: dict[str, Any]) -> str:
    """Render a standalone candidate-teacher report."""
    candidate = report["teacher"]["name"]
    comparison = report["candidate_vs_luxical"]
    gate_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in comparison["gates"].items()
    )
    probe_rows = []
    for name in ("luxical_one", "arctic_medium", "candidate"):
        probe = report[name]["probe"]
        probe_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{probe['macro_f1']:.5f}</td>"
            f"<td>{probe['category_macro_f1']['code']:.5f}</td>"
            f"<td>{probe['category_macro_f1']['multilingual']:.5f}</td>"
            f"<td>{probe['category_macro_f1']['standard']:.5f}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(candidate)} evaluation</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 90rem; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #bbb; padding: .35rem .6rem; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f5f5; padding: 1rem; }}
</style></head><body>
<h1>{html.escape(candidate)} evaluation</h1>
<p>All required gates: {'PASS' if comparison['all_required_gates_passed'] else 'FAIL'}</p>
<table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Source probe</h2>
<table><thead><tr><th>Representation</th><th>Overall</th><th>Code</th><th>Multilingual</th><th>Standard</th></tr></thead>
<tbody>{''.join(probe_rows)}</tbody></table>
<details><summary>Complete JSON</summary><pre>{html.escape(json.dumps(report, indent=2, sort_keys=True))}</pre></details>
</body></html>"""


def write_report(candidate: Candidate, report: dict[str, Any]) -> tuple[str, str]:
    """Write JSON and HTML reports atomically."""
    json_url = f"{EVALUATION_ROOT}/{candidate.output_name}/report.json"
    html_url = f"{EVALUATION_ROOT}/{candidate.output_name}/report.html"
    for url, payload in (
        (json_url, json.dumps(report, indent=2, sort_keys=True)),
        (html_url, html_report(report)),
    ):
        filesystem, path = fsspec.core.url_to_fs(url)
        with atomic_rename(path, fs=filesystem) as temporary_path:
            with filesystem.open(temporary_path, "w") as file:
                file.write(payload)
    return json_url, html_url


@threadpool_limits.wrap(limits=CPU_THREADS)
def evaluate(candidate: Candidate) -> None:
    """Embed the fixed holdout and compare the candidate with the references."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    manifest = read_json(MANIFEST_URL)
    embedder = CandidateEmbedder(candidate)
    control_vectors = normalized_vectors(
        embedder.quantized_documents(
            [
                "A short English document about data processing.",
                "def add(left: int, right: int) -> int:\n    return left + right",
                "これは日本語の短い文書です。",
                "word " * 128,
            ]
        )
    )
    if np.unique(control_vectors, axis=0).shape[0] != len(control_vectors):
        raise ValueError("Teacher returned duplicate control vectors")

    texts: list[str] = []
    labels: list[str] = []
    categories: list[str] = []
    probe_roles: list[str] = []
    candidate_batches = []
    arctic_batches = []
    embedded_rows = 0
    reused_rows = 0
    new_embedding_duration = 0.0
    source_loop_started = time.perf_counter()
    for index, (source, source_result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Loading evaluation source %d/%d: %s", index, len(manifest["sources"]), source)
        source_table = evaluation_table(source_result["output_url"])
        hashes = source_table["raw_sha256"].to_pylist()
        candidate_quantized, reused, source_embedding_duration = load_or_embed_source(
            embedder,
            source_table,
            source_result["output_url"],
            manifest["sha256"],
        )
        if reused:
            reused_rows += len(source_table)
        else:
            embedded_rows += len(source_table)
            new_embedding_duration += source_embedding_duration
        candidate_batches.append(normalized_vectors(candidate_quantized))
        arctic_batches.append(arctic_vectors(source_result["output_url"], hashes))
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
    candidate_vectors = np.concatenate(candidate_batches)
    arctic_teacher_vectors = np.concatenate(arctic_batches)
    left, right = pair_indices(labels_array)

    baseline_path = hf_hub_download(repo_id=BASELINE_REPO, filename=BASELINE_FILE, revision=BASELINE_REVISION)
    baseline = Embedder.load(baseline_path)
    baseline_metrics = model_metrics(
        baseline,
        texts,
        labels_array,
        probe_roles_array,
        categories_array,
        candidate_vectors,
        left,
        right,
    )
    baseline_metrics["candidate_teacher_fidelity"] = baseline_metrics.pop("arctic_fidelity")
    arctic_metrics = vector_metrics(arctic_teacher_vectors, labels_array, probe_roles_array, categories_array)
    candidate_metrics = vector_metrics(candidate_vectors, labels_array, probe_roles_array, categories_array)
    categories_by_source = source_categories(labels_array, categories_array)

    candidate_vs_luxical = teacher_comparison_report(candidate_metrics, baseline_metrics)
    add_source_details(candidate_vs_luxical, categories_by_source)
    arctic_vs_luxical = teacher_comparison_report(arctic_metrics, baseline_metrics)
    add_source_details(arctic_vs_luxical, categories_by_source)
    candidate_vs_arctic = representation_comparison(candidate_metrics, arctic_metrics)
    del candidate_vs_arctic["collapse"]
    overlap = failure_overlap(arctic_vs_luxical, candidate_vs_luxical)
    report = {
        "evaluation": candidate.output_name,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "evaluation_rows": len(texts),
        "predeclared_ood_sources": sorted(PREDECLARED_OOD_SOURCES),
        "teacher": {
            "name": candidate.name,
            "id": candidate.model_id,
            "revision": candidate.revision,
            "root": candidate.vector_root,
            "embedding_dimension": CANDIDATE_DIMENSION,
            "quantization_limit": TEACHER_QUANTIZATION_LIMIT,
            "maximum_tokens_per_window": MAX_TEACHER_TOKENS,
            "windows_per_document": WINDOWS_PER_DOCUMENT,
            "inference_dtype": str(INFERENCE_DTYPE).removeprefix("torch."),
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "pooling_implementation": candidate.pooling,
            "document_prompt": candidate.prompt,
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
        "arctic_medium": arctic_metrics,
        "candidate": candidate_metrics,
        "candidate_vs_luxical": candidate_vs_luxical,
        "arctic_vs_luxical": arctic_vs_luxical,
        "candidate_vs_arctic": candidate_vs_arctic,
        "failure_summary": failure_summary(candidate_vs_luxical),
        "failure_overlap_with_arctic": overlap,
    }
    json_url, html_url = write_report(candidate, report)
    summary = {
        "json_url": json_url,
        "html_url": html_url,
        "evaluation_rows": len(texts),
        "all_required_gates_passed": candidate_vs_luxical["all_required_gates_passed"],
        "failed_gates": [name for name, passed in candidate_vs_luxical["gates"].items() if not passed],
        "failure_summary": report["failure_summary"],
        "failure_overlap_with_arctic": overlap,
        "embedding_run": report["embedding_run"],
        "luxical_one": summary_metrics(baseline_metrics),
        "arctic_medium": summary_metrics(arctic_metrics),
        "candidate": summary_metrics(candidate_metrics),
        "candidate_vs_luxical": {
            "macro_f1_delta": candidate_vs_luxical["macro_f1_delta"],
            "worst_source_recall_delta": candidate_vs_luxical["worst_source_recall_delta"],
            "category_macro_f1_delta": candidate_vs_luxical["category_macro_f1_delta"],
            "probe_uncertainty": candidate_vs_luxical["probe_uncertainty"],
            "cluster_distribution_delta": candidate_vs_luxical["cluster_distribution_delta"],
        },
        "candidate_vs_arctic": {
            "macro_f1_delta": candidate_vs_arctic["macro_f1_delta"],
            "worst_source_recall_delta": candidate_vs_arctic["worst_source_recall_delta"],
            "category_macro_f1_delta": candidate_vs_arctic["category_macro_f1_delta"],
            "probe_uncertainty": candidate_vs_arctic["probe_uncertainty"],
            "cluster_distribution_delta": candidate_vs_arctic["cluster_distribution_delta"],
        },
    }
    result_path = Path(f"{RESULT_FILE_PREFIX}-{candidate.name}")
    result_path.write_text(json.dumps(summary, sort_keys=True))
    serialized = json.dumps(summary, sort_keys=True)
    for index, start in enumerate(range(0, len(serialized), LOG_CHUNK_CHARACTERS)):
        chunk = serialized[start : start + LOG_CHUNK_CHARACTERS]
        logger.info("LUXICAL_TEACHER_CANDIDATE_CHUNK=%04d:%s", index, chunk)


def parse_args() -> argparse.Namespace:
    """Parse the teacher candidate name."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    return parser.parse_args()


def main() -> None:
    """Evaluate the selected teacher candidate."""
    args = parse_args()
    evaluate(CANDIDATES[args.candidate])


if __name__ == "__main__":
    main()
