# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Score raw documents from JSONL, Parquet, or Hugging Face into JSONL."""

import dataclasses
import json
import logging
import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import islice
from typing import Iterable

import fsspec
import jax

import levanter.config
import levanter.tracker
import levanter.trainer
from levanter.analysis.document_losses import Document, DocumentLoss, DocumentSourceConfig, iter_documents
from levanter.main.perplexity_gap import GapFinderModelConfig, ModelLossRunner, load_model_runner, resolved_model_spec
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalDocumentsConfig:
    model: GapFinderModelConfig
    source: DocumentSourceConfig
    output_path: str
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(per_device_eval_parallelism=1, tracker=NoopConfig())
    )
    max_eval_length: int = 4096

    def __post_init__(self):
        if self.max_eval_length < 2:
            raise ValueError("max_eval_length must be at least 2")


def export_document_losses(runner: ModelLossRunner, documents: Iterable[Document], output_path: str) -> int:
    """Stream one record per original document; all hosts participate in scoring."""
    iterator = iter(documents)
    processed = 0
    with ExitStack() as stack:
        stream = None
        if jax.process_index() == 0:
            stream = stack.enter_context(fsspec.open(output_path, "wt", auto_mkdir=True))
        while batch := list(islice(iterator, runner.eval_batch_size)):
            sums, counts = runner.score_token_totals([doc.text for doc in batch])
            for document, loss_sum, token_count in zip(batch, sums, counts, strict=True):
                num_bytes = len(document.text.encode("utf-8"))
                record = DocumentLoss(
                    doc_id=document.doc_id,
                    corpus_id=document.corpus_id,
                    loss=float(loss_sum / token_count) if token_count else None,
                    bits_per_byte=float(loss_sum / (math.log(2) * num_bytes)) if token_count and num_bytes else None,
                )
                if stream is not None:
                    stream.write(json.dumps(dataclasses.asdict(record), allow_nan=False) + "\n")
            processed += len(batch)
            logger.info("Scored %d documents", processed)
    return processed


def main(config: EvalDocumentsConfig) -> None:
    levanter.trainer.initialize(config)
    try:
        with config.trainer.use_device_mesh():
            runner = load_model_runner(
                spec=resolved_model_spec(config.model),
                trainer=config.trainer,
                max_eval_length=config.max_eval_length,
                compute_axis_mapping=config.trainer.compute_axis_mapping,
                parameter_axis_mapping=config.trainer.parameter_axis_mapping,
            )
            count = export_document_losses(runner, iter_documents(config.source), config.output_path)
            logger.info("Wrote %d document losses to %s", count, config.output_path)
    finally:
        levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
