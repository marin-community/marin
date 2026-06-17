# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from types import SimpleNamespace

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from haliax.partitioning import ResourceAxis

from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.examples import (
    GrugLmExample,
    LabeledLmExample,
    LossLabelSpan,
    loss_labels_from_spans,
    named_lm_example_from_grug,
)
from levanter.eval import (
    EvalResult,
    LabeledEvaluator,
    LabeledLossFnOutput,
    LossFnOutput,
    LossLabelSpec,
    TaggedEvaluator,
    cb_tagged_evaluate,
)
from levanter.models.lm_model import LmExample
from levanter.tracker import current_tracker
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.utils.tree_utils import inference_mode

from .test_lm_model_loss import ToyLmConfig, ToyLmHeadModel
from .test_utils import use_test_mesh


class _FixedByteTokenizer:
    all_special_ids = []

    def get_vocab(self):
        return {"a": 0, "longtoken": 1}

    def convert_ids_to_tokens(self, idx):
        return ["a", "longtoken"][idx]

    def encode(self, text, *, add_special_tokens=False):
        del text, add_special_tokens
        return [0]

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self.convert_ids_to_tokens(i) for i in ids)


class _ZeroByteTokenizer:
    all_special_ids = []

    def get_vocab(self):
        return {".": 0, "<zero-byte>": 1, "abc": 2}

    def convert_ids_to_tokens(self, idx):
        return {0: ".", 1: "<zero-byte>", 2: "abc"}[idx]

    def encode(self, text, *, add_special_tokens=False):
        assert text == "."
        del add_special_tokens
        return [0]

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        token_text = {0: ".", 1: "", 2: "abc"}
        return "".join(token_text[token_id] for token_id in ids)


def test_tagged_evaluator_logs_historical_bpb_and_source_document_bpb():
    EvalBatch = Axis("batch", len(jax.devices()))
    examples = [
        *(GrugLmExample.causal(jnp.asarray([0], dtype=jnp.int32)) for _ in range(EvalBatch.size)),
        *(GrugLmExample.causal(jnp.asarray([1], dtype=jnp.int32)) for _ in range(EvalBatch.size)),
    ]
    dataset = ListAsyncDataset(examples)

    def loss_fn(_model, batch: GrugLmExample) -> LossFnOutput:
        return (
            jnp.ones_like(batch.tokens, dtype=jnp.float32),
            jnp.ones_like(batch.tokens, dtype=jnp.float32),
            batch.tokens,
        )

    with use_test_mesh(tensor_parallelism=1) as mesh:
        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(dataset, ["tiny"])],
            loss_fn=loss_fn,
            tokenizer=_FixedByteTokenizer(),
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(jnp.zeros((), dtype=jnp.float32))

    expected_bpb = np.mean(
        [
            np.log2(np.e) / len("a".encode("utf-8")),
            np.log2(np.e) / len("longtoken".encode("utf-8")),
        ]
    )
    expected_source_document_bpb = 2.0 * np.log2(np.e) / (len("a".encode("utf-8")) + len("longtoken".encode("utf-8")))
    assert result.micro_bpb == pytest.approx(expected_bpb, rel=2e-5)
    assert result.source_document_bpb == pytest.approx(expected_source_document_bpb, rel=2e-5)


def test_tagged_evaluator_source_document_bpb_includes_zero_byte_token_loss():
    EvalBatch = Axis("batch", len(jax.devices()))
    examples = [
        *(
            GrugLmExample(tokens=jnp.array([1, 1], dtype=jnp.int32), loss_weight=jnp.ones((2,), dtype=jnp.float32))
            for _ in range(EvalBatch.size)
        ),
        *(
            GrugLmExample(tokens=jnp.array([2, 2], dtype=jnp.int32), loss_weight=jnp.ones((2,), dtype=jnp.float32))
            for _ in range(EvalBatch.size)
        ),
    ]

    def loss_fn(_model, batch: GrugLmExample) -> LossFnOutput:
        losses = jnp.ones_like(batch.tokens, dtype=jnp.float32)
        return losses, batch.loss_weight, batch.tokens

    with use_test_mesh(tensor_parallelism=1) as mesh:
        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(ListAsyncDataset(examples), ["root/mix"])],
            loss_fn=loss_fn,
            tokenizer=_ZeroByteTokenizer(),
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(None)

    expected_source_document_bpb = 4.0 * EvalBatch.size * np.log2(np.e) / (6.0 * EvalBatch.size)
    assert result.source_document_bpb == pytest.approx(expected_source_document_bpb, rel=2e-5)
    assert result.tag_source_document_bpb["root/mix"] == pytest.approx(expected_source_document_bpb, rel=2e-5)
    assert result.tag_source_document_bpb["root"] == pytest.approx(expected_source_document_bpb, rel=2e-5)


def test_tagged_evaluator_accepts_grug_lm_examples():
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    Vocab = Axis("vocab", 32)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    EvalBatch = Axis("batch", len(jax.devices()))

    examples = []
    for i in range(EvalBatch.size):
        tokens = jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + i
        tokens = jnp.mod(tokens, Vocab.size)
        examples.append(GrugLmExample.causal(tokens))

    dataset = ListAsyncDataset(examples)

    with use_test_mesh(tensor_parallelism=1) as mesh:

        def loss_fn(model, batch) -> LossFnOutput:
            model = inference_mode(model, True)
            if isinstance(batch, LmExample):
                named_batch = batch
            else:
                named_batch = named_lm_example_from_grug(batch, Pos=model.Pos, batch_axis=EvalBatch)
            with hax.axis_mapping({EvalBatch.name: ResourceAxis.DATA}):
                per_pos_loss = model.compute_next_token_loss(named_batch, reduction=None, reduction_axis=()).array
            return per_pos_loss, named_batch.loss_weight.array, jnp.roll(named_batch.tokens.array, -1, axis=-1)

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(dataset, ["grug"])],
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(model)

    assert np.isfinite(result.micro_avg_loss)
    assert "grug" in result.tag_micro_losses


def test_tagged_evaluator_accepts_mixed_lm_example_types():
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    Vocab = Axis("vocab", 32)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    batch_size = max(2, len(jax.devices()) * 2)
    EvalBatch = Axis("batch", batch_size)
    Pos = Axis("position", cfg.max_seq_len)

    named_examples = []
    grug_examples = []
    for i in range(batch_size):
        named_tokens = hax.named(jnp.mod(jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + i, Vocab.size), Pos)
        named_examples.append(LmExample.causal(named_tokens))

    for i in range(batch_size):
        grug_tokens = jnp.mod(jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + (batch_size + i), Vocab.size)
        grug_examples.append(GrugLmExample.causal(grug_tokens))

    named_dataset = ListAsyncDataset(named_examples)
    grug_dataset = ListAsyncDataset(grug_examples)

    with use_test_mesh(tensor_parallelism=1) as mesh:

        def loss_fn(model, batch) -> LossFnOutput:
            model = inference_mode(model, True)
            if isinstance(batch, LmExample):
                named_batch = batch
            else:
                named_batch = named_lm_example_from_grug(batch, Pos=model.Pos, batch_axis=EvalBatch)
            with hax.axis_mapping({EvalBatch.name: ResourceAxis.DATA}):
                per_pos_loss = model.compute_next_token_loss(named_batch, reduction=None, reduction_axis=()).array
            return per_pos_loss, named_batch.loss_weight.array, jnp.roll(named_batch.tokens.array, -1, axis=-1)

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(named_dataset, ["named"]), (grug_dataset, ["grug"])],
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(model)

    assert np.isfinite(result.micro_avg_loss)
    assert "named" in result.tag_micro_losses
    assert "grug" in result.tag_micro_losses


def test_tagged_evaluator_accepts_grug_loss_protocol():
    seq_len = 8
    vocab_size = 32
    EvalBatch = Axis("batch", len(jax.devices()))

    examples = []
    for i in range(EvalBatch.size):
        tokens = jnp.mod(jnp.arange(seq_len, dtype=jnp.int32) + i, vocab_size)
        examples.append(GrugLmExample.causal(tokens))

    dataset = ListAsyncDataset(examples)

    def fake_grug_loss_fn(
        _params,
        token_ids,
        _loss_weight,
        *,
        mask=None,
        reduction="mean",
        logsumexp_weight=None,
        loss_dtype=jnp.float32,
    ):
        del mask, logsumexp_weight
        per_token = jnp.ones_like(token_ids, dtype=loss_dtype)
        if reduction == "none":
            return per_token
        if reduction == "sum":
            return jnp.sum(per_token)
        return jnp.mean(per_token)

    with use_test_mesh(tensor_parallelism=1) as mesh:

        def loss_fn(_model, batch: GrugLmExample) -> LossFnOutput:
            per_pos_loss = fake_grug_loss_fn(
                _model,
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="none",
                logsumexp_weight=None,
            )
            return per_pos_loss, batch.loss_weight, jnp.roll(batch.tokens, -1, axis=-1)

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(dataset, ["grug"])],
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(jnp.zeros((), dtype=jnp.float32))

    assert np.isfinite(result.micro_avg_loss)
    assert "grug" in result.tag_micro_losses


def test_labeled_evaluator_aggregates_exclusive_loss_labels():
    EvalBatch = Axis("batch", len(jax.devices()))

    examples = []
    for _ in range(EvalBatch.size):
        examples.append(
            LabeledLmExample(
                tokens=jnp.array([1, 2, 4, 8], dtype=jnp.int32),
                loss_labels=loss_labels_from_spans(
                    4,
                    [
                        LossLabelSpan(start=0, end=1, label=1),
                        LossLabelSpan(start=1, end=2, label=2),
                        LossLabelSpan(start=2, end=3, label=3),
                    ],
                ),
            )
        )

    label_spec = LossLabelSpec(
        id_to_name={
            0: "dont_score",
            1: "assistant_text",
            2: "assistant_tool_call",
            3: "tool_observation",
        },
        aggregates={
            "assistant": [1, 2],
            "assistant_text": [1],
            "tool_observation": [3],
        },
    )

    def loss_fn(_model, batch: LabeledLmExample) -> LabeledLossFnOutput:
        return batch.tokens.astype(jnp.float32), batch.loss_labels, jnp.roll(batch.tokens, -1, axis=-1)

    with use_test_mesh(tensor_parallelism=1) as mesh:
        evaluator = LabeledEvaluator(
            EvalBatch=EvalBatch,
            eval_set=ListAsyncDataset(examples),
            label_spec=label_spec,
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(None)

    np.testing.assert_allclose(result.label_losses["assistant"], 1.5)
    np.testing.assert_allclose(result.label_losses["assistant_text"], 1.0)
    np.testing.assert_allclose(result.label_losses["tool_observation"], 4.0)
    np.testing.assert_allclose(result.label_token_counts["assistant"], EvalBatch.size * 2)
    assert "dont_score" not in result.label_losses


def test_labeled_lm_evaluator_accepts_labeled_lm_examples():
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    Vocab = Axis("vocab", 32)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    EvalBatch = Axis("batch", len(jax.devices()))
    examples = []
    for i in range(EvalBatch.size):
        tokens = jnp.mod(jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + i, Vocab.size)
        loss_labels = jnp.ones_like(tokens)
        loss_labels = loss_labels.at[-1].set(0)
        examples.append(LabeledLmExample(tokens=tokens, loss_labels=loss_labels))

    label_spec = LossLabelSpec(id_to_name={0: "dont_score", 1: "assistant"})

    with use_test_mesh(tensor_parallelism=1) as mesh:
        evaluator = LabeledEvaluator.for_labeled_examples(
            EvalBatch=EvalBatch,
            eval_set=ListAsyncDataset(examples),
            label_spec=label_spec,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(model)

    assert np.isfinite(result.label_losses["assistant"])
    np.testing.assert_allclose(result.label_token_counts["assistant"], EvalBatch.size * (cfg.max_seq_len - 1))


def test_loss_label_spec_validates_aggregates():
    label_spec = LossLabelSpec(id_to_name={1: "assistant"})
    assert label_spec.aggregate_names == ("assistant",)

    with np.testing.assert_raises_regex(ValueError, "unknown label id"):
        LossLabelSpec(
            id_to_name={0: "dont_score", 1: "assistant"},
            aggregates={"assistant": [2]},
        )

    with np.testing.assert_raises_regex(ValueError, "dont_score_label"):
        LossLabelSpec(
            id_to_name={0: "dont_score", 1: "assistant"},
            aggregates={"assistant": [0, 1]},
        )


def test_loss_labels_from_spans_rejects_overlapping_spans():
    with np.testing.assert_raises_regex(ValueError, "overlaps"):
        loss_labels_from_spans(
            4,
            [
                LossLabelSpan(start=0, end=2, label=1),
                LossLabelSpan(start=1, end=3, label=2),
            ],
        )


def test_cb_tagged_evaluate_dedupes_force_and_logs_ema(caplog):
    class _FakeEvaluator:
        tokenizer = None
        dataset = SimpleNamespace(tag_to_index={"base": 0})

        def evaluate(self, _model):
            return EvalResult(
                micro_avg_loss=1.0,
                macro_avg_loss=1.0,
                tag_macro_losses={},
                tag_micro_losses={},
                total_eval_loading_time=0.0,
            )

    logger_name = "test.cb_tagged_evaluate"
    tracker = JsonLoggerConfig(logger_name=logger_name).init(run_id=None)
    caplog.set_level(logging.INFO, logger=logger_name)

    callback = cb_tagged_evaluate(_FakeEvaluator(), prefix="eval", eval_current=True, eval_ema=True)

    with current_tracker(tracker):
        step0 = SimpleNamespace(step=0, model=object(), eval_model=object())
        callback(step0)
        callback(step0, force=True)
        step1 = SimpleNamespace(step=1, model=object(), eval_model=object())
        callback(step1)

    log_events = []
    for record in caplog.records:
        if record.name != logger_name:
            continue
        payload = json.loads(record.message)
        if payload.get("event") == "log":
            log_events.append(payload)

    assert len(log_events) == 4
    assert [event["step"] for event in log_events] == [0, 0, 1, 1]
    assert "eval/loss" in log_events[0]["metrics"]
    assert "eval/ema/loss" in log_events[1]["metrics"]
    assert "eval/loss" in log_events[2]["metrics"]
    assert "eval/ema/loss" in log_events[3]["metrics"]
