# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score every document in a pinned Uncheatable Eval manifest with Snowball.

A completed subset has both its document JSONL and a matching summary. Interrupted
subsets restart from their first document; completed subsets are verified and skipped.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Protocol

import fsspec
import jax
import jmp
import levanter.config
import levanter.tracker
import levanter.trainer
import numpy as np
from huggingface_hub import snapshot_download
from levanter.analysis.document_losses import Document, DocumentSourceConfig, iter_documents
from levanter.grug.sharding import compact_grug_mesh
from levanter.main.perplexity_gap import GapFinderModelConfig, load_model_runner
from levanter.models.snowball import SnowballConfig
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import barrier_sync_named
from levanter.utils.mesh import MeshConfig

from experiments.evaluation.native_snowball_losses import load_native_runner

logger = logging.getLogger(__name__)
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
SCORING_VERSION = 2


class ScoringBackend(StrEnum):
    HF_GPU = "hf_gpu"
    NATIVE_TPU = "native_tpu"


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: str
    backend: ScoringBackend
    stage: str
    step: int
    executor_info_path: str | None = None
    training_tokens: int | None = None
    parent: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    manifest_path: str
    source_revision: str
    documents: int
    subsets: int


@dataclass
class SweepConfig:
    checkpoint: CheckpointSpec
    datasets: list[DatasetSpec]
    output_path: str
    max_eval_length: int = 4096
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            tracker=NoopConfig(),
            per_device_eval_parallelism=1,
            mp=jmp.get_policy("params=bfloat16,compute=bfloat16,output=float32"),
            use_explicit_mesh_axes=True,
            mesh=MeshConfig(
                axes={"data": -1, "expert": 1, "model": 1},
                compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
            ),
        )
    )


@dataclass(frozen=True)
class Subset:
    name: str
    input_path: str
    expected_documents: int


class DocumentScorer(Protocol):
    eval_batch_size: int

    def score_token_totals(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class SweepDocumentLoss:
    doc_id: str
    corpus_id: str
    loss: float | None
    bits_per_byte: float | None
    total_nll: float
    scored_tokens: int
    num_bytes: int
    text_sha256: str


@dataclass
class SubsetTotals:
    documents: int = 0
    scored_documents: int = 0
    total_bytes: int = 0
    scored_bytes: int = 0
    scored_tokens: int = 0
    loss_sum: float = 0.0
    document_loss_sum: float = 0.0
    document_bpb_sum: float = 0.0

    def add(self, document: Document, loss_sum: float, token_count: int) -> SweepDocumentLoss:
        num_bytes = len(document.text.encode("utf-8"))
        self.documents += 1
        self.total_bytes += num_bytes
        if token_count == 0:
            return SweepDocumentLoss(
                document.doc_id,
                document.corpus_id,
                None,
                None,
                0.0,
                0,
                num_bytes,
                hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            )
        if token_count < 0 or num_bytes == 0 or not math.isfinite(loss_sum):
            raise ValueError(f"Invalid scoring totals for document {document.doc_id!r}")
        mean_loss = loss_sum / token_count
        bits_per_byte = loss_sum / (math.log(2) * num_bytes)
        self.scored_documents += 1
        self.scored_bytes += num_bytes
        self.scored_tokens += token_count
        self.loss_sum += loss_sum
        self.document_loss_sum += mean_loss
        self.document_bpb_sum += bits_per_byte
        return SweepDocumentLoss(
            document.doc_id,
            document.corpus_id,
            mean_loss,
            bits_per_byte,
            loss_sum,
            token_count,
            num_bytes,
            hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        )

    def summary(self) -> dict:
        return {
            "documents": self.documents,
            "scored_documents": self.scored_documents,
            "unscorable_documents": self.documents - self.scored_documents,
            "total_bytes": self.total_bytes,
            "scored_bytes": self.scored_bytes,
            "scored_tokens": self.scored_tokens,
            "loss_sum": self.loss_sum,
            "token_weighted_loss": self.loss_sum / self.scored_tokens if self.scored_tokens else None,
            "byte_weighted_bits_per_byte": (
                self.loss_sum / (math.log(2) * self.scored_bytes) if self.scored_bytes else None
            ),
            "mean_document_loss": self.document_loss_sum / self.scored_documents if self.scored_documents else None,
            "mean_document_bits_per_byte": (
                self.document_bpb_sum / self.scored_documents if self.scored_documents else None
            ),
        }


def write_json(path: str, value: dict) -> None:
    with fsspec.open(path, "wt", auto_mkdir=True) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: str) -> dict:
    with fsspec.open(path, "rt") as stream:
        return json.load(stream)


def completed_subset(subset: Subset, output_path: str, fingerprint: str) -> dict | None:
    summary_path = f"{output_path}/{subset.name}/summary.json"
    fs, path = fsspec.core.url_to_fs(summary_path)
    if not fs.exists(path):
        return None
    summary = read_json(summary_path)
    if summary["fingerprint"] != fingerprint or summary["subset"] != dataclasses.asdict(subset):
        raise ValueError(f"Completed subset {subset.name!r} belongs to a different manifest or scoring config")
    with fsspec.open(summary["document_losses_path"], "rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != summary["document_losses_sha256"]:
        raise ValueError(f"Completed subset {subset.name!r} output checksum does not match")
    return summary


def score_subset(scorer: DocumentScorer, subset: Subset, output_path: str, fingerprint: str) -> dict:
    # Finish rank-zero publication before any host decides whether to resume.
    barrier_sync_named(f"subset-start-{fingerprint}-{subset.name}")
    existing = completed_subset(subset, output_path, fingerprint)
    if existing is not None:
        logger.info("Skipping completed subset %s (%d documents)", subset.name, existing["documents"])
        return existing
    start = time.perf_counter()
    last_progress = start
    output_uri = f"{output_path}/{subset.name}/document-losses.jsonl"
    totals = SubsetTotals()
    output_hash = hashlib.sha256()
    source = DocumentSourceConfig(input_path=subset.input_path, corpus_id=subset.name)
    documents = iter(iter_documents(source))
    logger.info("Scoring subset %s: all %d documents from %s", subset.name, subset.expected_documents, subset.input_path)
    output = fsspec.open(output_uri, "wb", auto_mkdir=True) if jax.process_index() == 0 else nullcontext(None)
    with output as stream:
        while batch := list(islice(documents, scorer.eval_batch_size)):
            sums, counts = scorer.score_token_totals([doc.text for doc in batch])
            for document, loss_sum, token_count in zip(batch, sums, counts, strict=True):
                record = totals.add(document, float(loss_sum), int(token_count))
                encoded = (json.dumps(dataclasses.asdict(record), allow_nan=False) + "\n").encode("utf-8")
                if stream is not None:
                    stream.write(encoded)
                output_hash.update(encoded)
            now = time.perf_counter()
            if now - last_progress >= 60:
                progress = {
                    "fingerprint": fingerprint,
                    "subset": subset.name,
                    "elapsed_seconds": now - start,
                    **totals.summary(),
                }
                if jax.process_index() == 0:
                    write_json(f"{output_path}/progress.json", progress)
                logger.info("UNCHEATABLE_PROGRESS %s", json.dumps(progress))
                last_progress = now
    if totals.documents != subset.expected_documents:
        raise ValueError(
            f"Subset {subset.name} yielded {totals.documents} documents; manifest expects {subset.expected_documents}"
        )
    elapsed = time.perf_counter() - start
    summary = {
        "fingerprint": fingerprint,
        "subset": dataclasses.asdict(subset),
        "document_losses_path": output_uri,
        "document_losses_sha256": output_hash.hexdigest(),
        "elapsed_seconds": elapsed,
        "documents_per_second": totals.documents / elapsed,
        "tokens_per_second": totals.scored_tokens / elapsed,
        **totals.summary(),
    }
    if jax.process_index() == 0:
        write_json(f"{output_path}/{subset.name}/summary.json", summary)
    logger.info("UNCHEATABLE_SUBSET_COMPLETE %s", json.dumps(summary))
    return summary


def dataset_manifest(spec: DatasetSpec) -> dict:
    manifest = read_json(spec.manifest_path)
    subsets = manifest["subsets"]
    if manifest["source_revision"] != spec.source_revision:
        raise ValueError(f"Dataset {spec.name} does not match its pinned revision")
    if len(subsets) != spec.subsets or sum(row["expected_documents"] for row in subsets) != spec.documents:
        raise ValueError(f"Dataset {spec.name} does not match its expected counts")
    names = [row["name"] for row in subsets]
    if len(set(names)) != len(names):
        raise ValueError("Manifest subset names must be unique")
    for name in [spec.name, *names]:
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid dataset or subset name: {name!r}")
    return manifest


def validate_checkpoint_locality(checkpoint: CheckpointSpec, datasets: list[DatasetSpec], output_path: str) -> None:
    """Reject configurations that move checkpoint weights between execution regions."""
    if checkpoint.backend == ScoringBackend.NATIVE_TPU:
        prefix = "gs://marin-us-central2/"
        if checkpoint.executor_info_path is None:
            raise ValueError("Native checkpoints require their original executor metadata")
        paths = [checkpoint.path, checkpoint.executor_info_path, output_path, *(d.manifest_path for d in datasets)]
    else:
        prefix = "s3://marin-us-east-02a/"
        paths = [checkpoint.path, output_path, *(d.manifest_path for d in datasets)]
    if any(not path.startswith(prefix) for path in paths):
        raise ValueError(f"All checkpoint, manifest, and output paths must remain under {prefix}")


def score_dataset(scorer: DocumentScorer, dataset: DatasetSpec, manifest: dict, output_path: str, scoring: dict) -> dict:
    identity = {"manifest": manifest, "scoring": scoring}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    identity_path = f"{output_path}/manifest.json"
    fs, path = fsspec.core.url_to_fs(identity_path)
    if fs.exists(path) and read_json(identity_path) != {"fingerprint": fingerprint, **identity}:
        raise ValueError("Output path contains a different evaluation manifest")
    if jax.process_index() == 0:
        write_json(identity_path, {"fingerprint": fingerprint, **identity})
    summaries = [
        score_subset(scorer, Subset(row["name"], row["input_path"], row["expected_documents"]), output_path, fingerprint)
        for row in manifest["subsets"]
    ]
    result = {
        "fingerprint": fingerprint,
        "dataset": dataclasses.asdict(dataset),
        "manifest": manifest,
        "scoring": scoring,
        "subsets": summaries,
        "status": "complete",
    }
    if jax.process_index() == 0:
        write_json(f"{output_path}/summary.json", result)
        if "IRIS_OUTPUT_DIR" in os.environ:
            write_json(str(Path(os.environ["IRIS_OUTPUT_DIR"]) / f"{dataset.name}-summary.json"), result)
        logger.info("UNCHEATABLE_SWEEP_COMPLETE %s", json.dumps(result))
    return result


def main(config: SweepConfig):
    if config.max_eval_length < 2:
        raise ValueError("max_eval_length must be at least two")
    if not config.datasets or len({d.name for d in config.datasets}) != len(config.datasets):
        raise ValueError("Supply at least one dataset with unique names")
    validate_checkpoint_locality(config.checkpoint, config.datasets, config.output_path)
    manifests = [dataset_manifest(dataset) for dataset in config.datasets]
    allowed_prefix = (
        "gs://marin-us-central2/"
        if config.checkpoint.backend == ScoringBackend.NATIVE_TPU
        else "s3://marin-us-east-02a/"
    )
    if any(not row["input_path"].startswith(allowed_prefix) for m in manifests for row in m["subsets"]):
        raise ValueError("Dataset shard paths must be in the scoring region")
    tokenizer_path = snapshot_download(
        TOKENIZER,
        revision=TOKENIZER_REVISION,
        allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "chat_template*"],
    )
    levanter.trainer.initialize(config)
    try:
        if config.checkpoint.backend == ScoringBackend.NATIVE_TPU:
            if jax.default_backend() != "tpu":
                raise ValueError("GCS native checkpoints must run on TPU in us-central2")
            mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
            mesh_context = jax.set_mesh(mesh)
        else:
            if jax.default_backend() != "gpu" or jax.process_count() != 1 or jax.device_count() != 8:
                raise ValueError("HF GPU checkpoints require one host with eight GPUs in cw-us-east-02a")
            mesh = None
            mesh_context = config.trainer.use_device_mesh()
        with mesh_context:
            if config.checkpoint.backend == ScoringBackend.NATIVE_TPU:
                assert mesh is not None and config.checkpoint.executor_info_path is not None
                runner = load_native_runner(
                    checkpoint_path=config.checkpoint.path,
                    executor_info_path=config.checkpoint.executor_info_path,
                    tokenizer_path=tokenizer_path,
                    eval_batch_size=config.trainer.eval_batch_size,
                    max_eval_length=config.max_eval_length,
                    mp=config.trainer.mp,
                    mesh=mesh,
                )
            else:
                converter = SnowballConfig(
                    reference_checkpoint=config.checkpoint.path, tokenizer=tokenizer_path
                ).hf_checkpoint_converter()
                model_config = dataclasses.replace(
                    converter.config_from_hf_config(converter.default_hf_config),
                    tokenizer=tokenizer_path,
                    moe_implementation="sonic",
                    attention_implementation="gpu_fa4_cute",
                )
                runner = load_model_runner(
                    spec=GapFinderModelConfig(
                        checkpoint_path=config.checkpoint.path,
                        checkpoint_is_hf=True,
                        model=model_config,
                        tokenizer=tokenizer_path,
                    ),
                    trainer=config.trainer,
                    max_eval_length=config.max_eval_length,
                    compute_axis_mapping=config.trainer.compute_axis_mapping,
                    parameter_axis_mapping=config.trainer.parameter_axis_mapping,
                )
            scoring = {
                "version": SCORING_VERSION,
                "checkpoint": dataclasses.asdict(config.checkpoint),
                "tokenizer": TOKENIZER,
                "tokenizer_revision": TOKENIZER_REVISION,
                "max_eval_length": config.max_eval_length,
                "precision": str(config.trainer.mp),
                "device_count": jax.device_count(),
                "process_count": jax.process_count(),
                "eval_batch_size": runner.eval_batch_size,
            }
            results = [
                score_dataset(runner, dataset, manifest, f"{config.output_path}/{dataset.name}", scoring)
                for dataset, manifest in zip(config.datasets, manifests, strict=True)
            ]
            if jax.process_index() == 0:
                write_json(f"{config.output_path}/summary.json", {"status": "complete", "datasets": results})
    finally:
        levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
