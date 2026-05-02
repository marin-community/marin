# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Sequence

import equinox as eqx
import jax
import jmp
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
import levanter.tracker
from levanter.analysis.model_perplexity import ModelScoreReportBuilder, ScoredDocument, write_model_score_files
from levanter.analysis.perplexity_gap import (
    GapReportBuilder,
    RawTextDocument,
    TokenizedChunk,
    TokenizedDocument,
    batch_chunks,
    chunk_tokenized_document,
    iter_raw_text_documents,
    tokenize_text_with_byte_spans,
    write_report_files,
)
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.data.text import DatasetComponent
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.tokenizers import MarinTokenizer, TokenizerBackend, load_tokenizer
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


@dataclass
class GapFinderModelConfig:
    checkpoint_path: str
    model: LmConfig | None = None
    checkpoint_is_hf: bool = False
    tokenizer: str | None = None
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF
    trust_remote_code: bool = False


@dataclass
class GapFinderConfig:
    model_a: GapFinderModelConfig
    model_b: GapFinderModelConfig
    datasets: dict[str, DatasetComponent] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    output_path: str = "perplexity-gap"
    max_eval_length: int = 4096
    max_docs_per_dataset: int | None = 256
    max_doc_bytes: int | None = 32_768


@dataclass
class ModelPerplexityConfig:
    model: GapFinderModelConfig
    datasets: dict[str, DatasetComponent] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    output_path: str = "model-perplexity"
    max_eval_length: int = 4096
    max_docs_per_dataset: int | None = 256
    max_doc_bytes: int | None = 32_768


@dataclass
class _ModelRunner:
    label: str
    model: LmHeadModel
    tokenizer: MarinTokenizer
    hf_tokenizer: Any
    eval_batch_size: int
    eval_length: int
    compute_losses: Any

    def score_texts(self, texts: list[str]) -> tuple[list[TokenizedDocument], list[np.ndarray]]:
        tokenized = [tokenize_text_with_byte_spans(self.tokenizer, self.hf_tokenizer, text) for text in texts]
        per_byte_losses = [np.zeros(doc.num_bytes, dtype=np.float64) for doc in tokenized]

        chunks: list[TokenizedChunk] = []
        for doc_index, doc in enumerate(tokenized):
            if doc.num_bytes <= 0 or len(doc.token_ids) <= 1:
                continue
            chunks.extend(chunk_tokenized_document(doc, self.eval_length, doc_index=doc_index))

        for chunk_batch in batch_chunks(chunks, batch_size=self.eval_batch_size, max_eval_length=self.eval_length):
            batch_losses = self._score_chunk_batch(chunk_batch)
            _check_finite_losses(self.label, batch_losses)
            for row, chunk in enumerate(chunk_batch):
                actual_len = len(chunk.token_ids)
                if actual_len <= 1:
                    continue
                starts = chunk.byte_starts[1:actual_len]
                ends = chunk.byte_ends[1:actual_len]
                losses = batch_losses[row, : actual_len - 1]
                out = per_byte_losses[chunk.doc_index]
                _accumulate_token_losses(out, starts, ends, losses)

        return tokenized, per_byte_losses

    def _score_chunk_batch(self, chunk_batch: list[TokenizedChunk]) -> np.ndarray:
        tokens = np.zeros((self.eval_batch_size, self.eval_length), dtype=np.int32)
        loss_weight = np.zeros((self.eval_batch_size, self.eval_length), dtype=np.float32)

        for row, chunk in enumerate(chunk_batch):
            length = len(chunk.token_ids)
            tokens[row, :length] = chunk.token_ids
            if length > 1:
                loss_weight[row, : length - 1] = 1.0

        batch = GrugLmExample(
            tokens=jax.device_put(tokens),
            loss_weight=jax.device_put(loss_weight),
            attn_mask=GrugAttentionMask.causal(),
        )
        losses = self.compute_losses(self.model, batch)
        return np.asarray(jax.device_get(losses), dtype=np.float64)


def score_main(config: ModelPerplexityConfig) -> None:
    levanter.initialize(config)
    if not config.datasets:
        raise ValueError("Model perplexity scoring requires at least one dataset.")

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping
    model_spec = _resolved_model_spec(config.model)

    with config.trainer.use_device_mesh():
        runner = _load_model_runner(
            spec=model_spec,
            trainer=config.trainer,
            max_eval_length=config.max_eval_length,
            compute_axis_mapping=compute_axis_mapping,
            parameter_axis_mapping=parameter_axis_mapping,
        )

        report = ModelScoreReportBuilder(model_name=runner.label)
        scored_documents: list[ScoredDocument] = []

        docs_processed = 0
        current_dataset: str | None = None
        for docs in _document_batches(
            iter_raw_text_documents(
                config.datasets,
                max_docs_per_dataset=config.max_docs_per_dataset,
                max_doc_bytes=config.max_doc_bytes,
            ),
            batch_size=config.trainer.eval_batch_size,
        ):
            batch_dataset = docs[0].dataset_name
            if batch_dataset != current_dataset:
                current_dataset = batch_dataset
                logger.info("Starting dataset %s", current_dataset)
            texts = [doc.text for doc in docs]
            tokenized_docs, per_byte_losses = runner.score_texts(texts)
            for doc, tokenized, losses in zip(docs, tokenized_docs, per_byte_losses, strict=True):
                report.add_document(document=doc, per_byte_loss=losses)
                scored_documents.append(ScoredDocument(document=doc, per_byte_loss=losses, tokenized=tokenized))
            docs_processed += len(docs)
            if docs_processed % 32 == 0:
                logger.info("Processed %s documents for model perplexity scores", docs_processed)

        token_id_to_text = _token_id_to_text(scored_documents, runner.hf_tokenizer)
        summary = report.build_summary()
        write_model_score_files(
            config.output_path,
            summary,
            scored_documents,
            vocab_size=len(runner.tokenizer),
            token_id_to_text=token_id_to_text,
        )
        levanter.tracker.log(_model_score_scalars(summary), step=0)
        _log_model_score_artifact(
            summary,
            scored_documents,
            vocab_size=len(runner.tokenizer),
            token_id_to_text=token_id_to_text,
        )

    levanter.tracker.current_tracker().finish()


def main(config: GapFinderConfig) -> None:
    levanter.initialize(config)
    if not config.datasets:
        raise ValueError("Gap finder requires at least one dataset.")

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    model_a_spec = _resolved_model_spec(config.model_a)
    model_b_spec = _resolved_model_spec(config.model_b)

    with config.trainer.use_device_mesh():
        runner_a = _load_model_runner(
            spec=model_a_spec,
            trainer=config.trainer,
            max_eval_length=config.max_eval_length,
            compute_axis_mapping=compute_axis_mapping,
            parameter_axis_mapping=parameter_axis_mapping,
        )
        runner_b = _load_model_runner(
            spec=model_b_spec,
            trainer=config.trainer,
            max_eval_length=config.max_eval_length,
            compute_axis_mapping=compute_axis_mapping,
            parameter_axis_mapping=parameter_axis_mapping,
        )

        report = GapReportBuilder(
            model_a_name=runner_a.label,
            model_b_name=runner_b.label,
            output_path=config.output_path,
        )

        docs_processed = 0
        current_dataset: str | None = None
        for docs in _document_batches(
            iter_raw_text_documents(
                config.datasets,
                max_docs_per_dataset=config.max_docs_per_dataset,
                max_doc_bytes=config.max_doc_bytes,
            ),
            batch_size=config.trainer.eval_batch_size,
        ):
            batch_dataset = docs[0].dataset_name
            if batch_dataset != current_dataset:
                current_dataset = batch_dataset
                logger.info("Starting dataset %s", current_dataset)
            texts = [doc.text for doc in docs]
            tokenized_a, per_byte_a = runner_a.score_texts(texts)
            tokenized_b, per_byte_b = runner_b.score_texts(texts)
            for doc, doc_a, losses_a, doc_b, losses_b in zip(
                docs,
                tokenized_a,
                per_byte_a,
                tokenized_b,
                per_byte_b,
                strict=True,
            ):
                report.add_document(
                    document=doc,
                    per_byte_loss_a=losses_a,
                    per_byte_loss_b=losses_b,
                    tokenized_a=doc_a,
                    tokenized_b=doc_b,
                )
            docs_processed += len(docs)
            if docs_processed % 32 == 0:
                logger.info("Processed %s documents for perplexity-gap report", docs_processed)

        summary = report.write()
        levanter.tracker.log(_summary_scalars(summary), step=0)
        _log_report_artifact(summary)

    levanter.tracker.current_tracker().finish()


def _resolved_model_spec(spec: GapFinderModelConfig) -> GapFinderModelConfig:
    model = spec.model
    if model is None:
        if not spec.checkpoint_is_hf:
            raise ValueError("Native checkpoints require an explicit model config.")
        model = HFCheckpointConverter.from_hf(
            spec.checkpoint_path,
            trust_remote_code=spec.trust_remote_code,
        ).config_from_hf_checkpoint(spec.checkpoint_path)

    tokenizer = spec.tokenizer
    if tokenizer is None:
        if not spec.checkpoint_is_hf:
            raise ValueError("Native checkpoints require an explicit tokenizer.")
        tokenizer = spec.checkpoint_path

    return dataclasses.replace(spec, model=model, tokenizer=tokenizer)


def _load_model_runner(
    *,
    spec: GapFinderModelConfig,
    trainer: TrainerConfig,
    max_eval_length: int,
    compute_axis_mapping: Any,
    parameter_axis_mapping: Any,
) -> _ModelRunner:
    assert spec.model is not None
    assert spec.tokenizer is not None

    tokenizer = load_tokenizer(spec.tokenizer, backend=spec.tokenizer_backend)
    hf_tokenizer = tokenizer.as_hf_tokenizer()
    if not getattr(hf_tokenizer, "is_fast", False):
        raise ValueError(f"Tokenizer {spec.tokenizer!r} does not expose a fast tokenizer with offset mappings.")

    key = jax.random.PRNGKey(0)
    vocab_size = len(tokenizer)
    eval_length = min(max_eval_length, spec.model.max_Pos.size)
    EvalBatch = trainer.EvalBatch
    Pos = Axis("position", eval_length)
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
    if vocab_size != Vocab.size:
        logger.info("Rounding vocab size from %s to %s for partitioning", vocab_size, Vocab.size)

    mp: jmp.Policy = trainer.mp

    @hax.named_jit(axis_resources=compute_axis_mapping)
    def compute_losses(model: LmHeadModel, batch: GrugLmExample):
        model = inference_mode(model, True)
        model = mp.cast_to_compute(model)
        named_batch = named_lm_example_from_grug(batch, Pos=Pos, batch_axis=EvalBatch)
        return model.compute_next_token_loss(named_batch, reduction=None, reduction_axis=()).array

    if spec.checkpoint_is_hf:
        model_config = spec.model
        if not isinstance(model_config, HFCompatConfig):
            raise ValueError(f"Model config {type(model_config).__name__} cannot load HF checkpoints.")
        converter = model_config.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=spec.checkpoint_path, tokenizer=tokenizer)
        model = converter.load_pretrained(
            model_config.model_type,
            ref=spec.checkpoint_path,
            axis_mapping=parameter_axis_mapping,
            dtype=trainer.mp.compute_dtype,  # type: ignore[arg-type]
        )
    else:
        with use_cpu_device():
            model = eqx.filter_eval_shape(spec.model.build, Vocab, key=key)
            checkpoint_path = latest_checkpoint_path(spec.checkpoint_path)
            model = load_checkpoint(model, checkpoint_path, subpath="model")
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

    label = _model_label(spec)
    return _ModelRunner(
        label=label,
        model=model,
        tokenizer=tokenizer,
        hf_tokenizer=hf_tokenizer,
        eval_batch_size=trainer.eval_batch_size,
        eval_length=eval_length,
        compute_losses=compute_losses,
    )


def _document_batches(documents: Any, *, batch_size: int) -> Any:
    batch: list[RawTextDocument] = []
    for document in documents:
        batch.append(document)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _model_label(spec: GapFinderModelConfig) -> str:
    if spec.checkpoint_is_hf:
        return spec.checkpoint_path
    return os.path.basename(spec.checkpoint_path.rstrip("/")) or spec.checkpoint_path


def _token_id_to_text(scored_documents: Sequence[ScoredDocument], hf_tokenizer: Any) -> dict[int, str]:
    token_ids = sorted({int(token_id) for doc in scored_documents for token_id in doc.tokenized.token_ids.tolist()})
    if not token_ids:
        return {}

    token_texts = hf_tokenizer.convert_ids_to_tokens(token_ids)
    if not isinstance(token_texts, list):
        token_texts = list(token_texts)

    mapping: dict[int, str] = {}
    for token_id, token_text in zip(token_ids, token_texts, strict=True):
        if token_text is None:
            continue
        mapping[token_id] = str(token_text)
    return mapping


def _accumulate_token_losses(
    out: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    losses: np.ndarray,
) -> None:
    valid = (starts >= 0) & (ends > starts)
    if not np.any(valid):
        return

    valid_starts = starts[valid].astype(np.int64, copy=False)
    valid_ends = ends[valid].astype(np.int64, copy=False)
    widths = valid_ends - valid_starts
    weights = losses[valid] / widths

    diff = np.zeros(len(out) + 1, dtype=np.float64)
    np.add.at(diff, valid_starts, weights)
    np.add.at(diff, valid_ends, -weights)
    out += np.cumsum(diff[:-1])


def _summary_scalars(summary: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for row in summary["datasets"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/datasets/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
        scalars[f"gap/datasets/{row['name']}/model_a_bpb"] = float(row["model_a_bpb"])
        scalars[f"gap/datasets/{row['name']}/model_b_bpb"] = float(row["model_b_bpb"])
    for row in summary["dataset_groups"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/groups/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
    for row in summary["pattern_buckets"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/patterns/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
    return scalars


def _model_score_scalars(summary: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for row in summary["datasets"]:
        if row["bpb"] is None:
            continue
        scalars[f"score/datasets/{row['name']}/bpb"] = float(row["bpb"])
    for row in summary["dataset_groups"]:
        if row["bpb"] is None:
            continue
        scalars[f"score/groups/{row['name']}/bpb"] = float(row["bpb"])
    for row in summary["pattern_buckets"]:
        if row["bpb"] is None:
            continue
        scalars[f"score/patterns/{row['name']}/bpb"] = float(row["bpb"])
    return scalars


def _check_finite_losses(label: str, losses: np.ndarray) -> None:
    if np.isfinite(losses).all():
        return

    raise ValueError(
        f"Non-finite losses while scoring {label}. "
        "This usually means the checkpoint and tokenizer are incompatible."
    )


def _log_report_artifact(summary: dict[str, Any]) -> None:
    if jax.process_index() != 0:
        return

    with tempfile.TemporaryDirectory(prefix="perplexity-gap-report-") as tmpdir:
        write_report_files(tmpdir, summary)
        levanter.tracker.current_tracker().log_artifact(
            tmpdir,
            name="perplexity_gap_report",
            type="perplexity_gap_report",
        )


def _log_model_score_artifact(
    summary: dict[str, Any],
    scored_documents: Sequence[ScoredDocument],
    *,
    vocab_size: int | None = None,
    token_id_to_text: dict[int, str] | None = None,
) -> None:
    if jax.process_index() != 0:
        return

    with tempfile.TemporaryDirectory(prefix="model-perplexity-scores-") as tmpdir:
        write_model_score_files(
            tmpdir,
            summary,
            scored_documents,
            vocab_size=vocab_size,
            token_id_to_text=token_id_to_text,
        )
        levanter.tracker.current_tracker().log_artifact(
            tmpdir,
            name="model_perplexity_scores",
            type="model_perplexity_scores",
        )


if __name__ == "__main__":
    levanter.config.main(main)()
