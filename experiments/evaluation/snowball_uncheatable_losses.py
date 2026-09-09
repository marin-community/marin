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
from dataclasses import dataclass, field
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
from levanter.analysis.document_losses import Document, DocumentLoss, DocumentSourceConfig, iter_documents
from levanter.main.perplexity_gap import GapFinderModelConfig, load_model_runner
from levanter.models.snowball import SnowballConfig
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.testing.inference.snowball import SNOWBALL

logger = logging.getLogger(__name__)
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
SCORING_VERSION = 1
SOURCE_REVISION = "185c463882a8ae0f203e51ae5852d8cf4fe299cf"
EXPECTED_SUBSETS = 14
EXPECTED_DOCUMENTS = 11052


@dataclass
class SweepConfig:
    manifest_path: str
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

    def add(self, document: Document, loss_sum: float, token_count: int) -> DocumentLoss:
        num_bytes = len(document.text.encode("utf-8"))
        self.documents += 1
        self.total_bytes += num_bytes
        if token_count == 0:
            return DocumentLoss(document.doc_id, document.corpus_id, None, None)
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
        return DocumentLoss(document.doc_id, document.corpus_id, mean_loss, bits_per_byte)

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
    with fsspec.open(output_uri, "wb", auto_mkdir=True) as stream:
        while batch := list(islice(documents, scorer.eval_batch_size)):
            sums, counts = scorer.score_token_totals([doc.text for doc in batch])
            for document, loss_sum, token_count in zip(batch, sums, counts, strict=True):
                record = totals.add(document, float(loss_sum), int(token_count))
                encoded = (json.dumps(dataclasses.asdict(record), allow_nan=False) + "\n").encode("utf-8")
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
    write_json(f"{output_path}/{subset.name}/summary.json", summary)
    logger.info("UNCHEATABLE_SUBSET_COMPLETE %s", json.dumps(summary))
    return summary


def main(config: SweepConfig):
    if config.max_eval_length < 2:
        raise ValueError("max_eval_length must be at least two")
    manifest = read_json(config.manifest_path)
    subsets = [
        Subset(**{key: row[key] for key in ("name", "input_path", "expected_documents")}) for row in manifest["subsets"]
    ]
    if manifest["source_revision"] != SOURCE_REVISION:
        raise ValueError("Manifest does not describe the pinned Uncheatable Eval revision")
    if len(subsets) != EXPECTED_SUBSETS or sum(subset.expected_documents for subset in subsets) != EXPECTED_DOCUMENTS:
        raise ValueError("Manifest must contain all 14 subsets and 11,052 original documents")
    if len({subset.name for subset in subsets}) != len(subsets):
        raise ValueError("Manifest subset names must be unique")
    for subset in subsets:
        if not subset.name or Path(subset.name).name != subset.name or subset.name in (".", ".."):
            raise ValueError(f"Invalid subset name: {subset.name!r}")
    scoring = {
        "version": SCORING_VERSION,
        "checkpoint": SNOWBALL.export_uri,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "max_eval_length": config.max_eval_length,
        "moe_implementation": "sonic",
        "attention_implementation": "gpu_fa4_cute",
        "precision": str(config.trainer.mp),
        "mesh": dataclasses.asdict(config.trainer.mesh),
        "per_device_eval_parallelism": config.trainer.per_device_eval_parallelism,
        "device_count": 8,
    }
    identity = {"manifest": manifest, "scoring": scoring}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    identity_path = f"{config.output_path}/manifest.json"
    fs, path = fsspec.core.url_to_fs(identity_path)
    if fs.exists(path) and read_json(identity_path) != {"fingerprint": fingerprint, **identity}:
        raise ValueError("Output path contains a different evaluation manifest")
    write_json(identity_path, {"fingerprint": fingerprint, **identity})
    summaries = [completed_subset(subset, config.output_path, fingerprint) for subset in subsets]
    if any(summary is None for summary in summaries):
        tokenizer_path = snapshot_download(
            TOKENIZER,
            revision=TOKENIZER_REVISION,
            allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "chat_template*"],
        )
        converter = SnowballConfig(
            reference_checkpoint=SNOWBALL.export_uri, tokenizer=tokenizer_path
        ).hf_checkpoint_converter()
        model_config = dataclasses.replace(
            converter.config_from_hf_config(converter.default_hf_config),
            tokenizer=tokenizer_path,
            moe_implementation="sonic",
            attention_implementation="gpu_fa4_cute",
        )
        levanter.trainer.initialize(config)
        try:
            if jax.process_count() != 1 or jax.device_count() != 8:
                raise ValueError("This Snowball sweep requires one host with eight devices")
            with config.trainer.use_device_mesh():
                runner = load_model_runner(
                    spec=GapFinderModelConfig(
                        checkpoint_path=SNOWBALL.export_uri,
                        checkpoint_is_hf=True,
                        model=model_config,
                        tokenizer=tokenizer_path,
                    ),
                    trainer=config.trainer,
                    max_eval_length=config.max_eval_length,
                    compute_axis_mapping=config.trainer.compute_axis_mapping,
                    parameter_axis_mapping=config.trainer.parameter_axis_mapping,
                )
                summaries = [
                    previous if previous is not None else score_subset(runner, subset, config.output_path, fingerprint)
                    for subset, previous in zip(subsets, summaries, strict=True)
                ]
        finally:
            levanter.tracker.current_tracker().finish()
    result = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "scoring": scoring,
        "subsets": summaries,
        "status": "complete",
    }
    write_json(f"{config.output_path}/summary.json", result)
    if "IRIS_OUTPUT_DIR" in os.environ:
        write_json(str(Path(os.environ["IRIS_OUTPUT_DIR"]) / "summary.json"), result)
    logger.info("UNCHEATABLE_SWEEP_COMPLETE %s", json.dumps(result))


if __name__ == "__main__":
    levanter.config.main(main)()
