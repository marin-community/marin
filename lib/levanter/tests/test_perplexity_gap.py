# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import tempfile
from typing import Any

import jax
import numpy as np
import pytest

import haliax

import levanter.analysis.perplexity_gap as gap_analysis
from levanter.analysis.perplexity_gap import (
    GapReportBuilder,
    RawTextDocument,
    TokenizedChunk,
    TokenizedDocument,
    _truncate_text_to_byte_limit,
    batch_chunks,
    chunk_tokenized_document,
    render_report_markdown,
    tokenize_text_with_byte_spans,
    worst_document_rows,
    write_report_files,
)
from levanter.checkpoint import save_checkpoint
from levanter.data.text import DatasetComponent, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.distributed import DistributedConfig
from levanter.main.perplexity_gap import (
    GapFinderConfig,
    GapFinderModelConfig,
    _accumulate_token_losses,
    _check_finite_losses,
    _log_report_artifact,
    main,
)
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.tracker import current_tracker
from levanter.tracker.tracker import DictTracker
from levanter.tokenizers import load_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig


def test_tokenize_text_with_byte_spans_covers_utf8_bytes():
    tokenizer = load_tokenizer("gpt2")
    hf_tokenizer = tokenizer.as_hf_tokenizer()
    text = "hello  \nnaive café"

    tokenized = tokenize_text_with_byte_spans(tokenizer, hf_tokenizer, text)

    spans = [
        (start, end)
        for start, end in zip(tokenized.byte_starts, tokenized.byte_ends, strict=True)
        if start >= 0 and end > start
    ]
    assert tokenized.num_bytes == len(text.encode("utf-8"))
    assert spans
    assert spans[0][0] == 0
    assert spans[-1][1] == tokenized.num_bytes
    assert sum(end - start for start, end in spans) == tokenized.num_bytes


def test_gap_report_builder_tracks_whitespace_bucket():
    report = GapReportBuilder(model_a_name="a", model_b_name="b", output_path="/tmp/report")
    document = RawTextDocument(
        dataset_name="paloma/example",
        tags=("paloma/example",),
        shard_name="docs",
        row_index=0,
        text="a  b",
    )

    report.add_document(
        document=document,
        per_byte_loss_a=jax.device_get(jax.numpy.asarray([0.1, 0.2, 0.2, 0.1], dtype=jax.numpy.float32)),
        per_byte_loss_b=jax.device_get(jax.numpy.asarray([0.0, 0.0, 0.0, 0.0], dtype=jax.numpy.float32)),
    )

    summary = report.build_summary()
    bucket_names = {row["name"] for row in summary["pattern_buckets"]}
    group_names = {row["name"] for row in summary["dataset_groups"]}

    assert "whitespace/multi_space" in bucket_names
    assert "paloma" in group_names


def test_gap_report_builder_records_per_model_literal_boundaries():
    report = GapReportBuilder(model_a_name="a", model_b_name="b", output_path="/tmp/report")
    document = RawTextDocument(
        dataset_name="paloma/example",
        tags=("paloma/example",),
        shard_name="docs",
        row_index=0,
        text="abc",
    )
    tokenized_a = TokenizedDocument(
        token_ids=np.asarray([1], dtype=np.int32),
        byte_starts=np.asarray([0], dtype=np.int32),
        byte_ends=np.asarray([3], dtype=np.int32),
        num_bytes=3,
    )
    tokenized_b = TokenizedDocument(
        token_ids=np.asarray([1, 2], dtype=np.int32),
        byte_starts=np.asarray([0, 1], dtype=np.int32),
        byte_ends=np.asarray([1, 3], dtype=np.int32),
        num_bytes=3,
    )

    report.add_document(
        document=document,
        per_byte_loss_a=np.asarray([0.1, 0.1, 0.1], dtype=np.float64),
        per_byte_loss_b=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        tokenized_a=tokenized_a,
        tokenized_b=tokenized_b,
    )

    summary = report.build_summary()
    literal_row = summary["top_literals"]["model_a_worse"][0]

    assert literal_row["name"] == "abc"
    assert literal_row["model_a_token_boundaries"] == "|abc|"
    assert literal_row["model_b_token_boundaries"] == "|a|bc|"
    assert literal_row["example_dataset"] == "paloma/example"


def test_gap_report_builder_renders_literal_boundaries_only_for_reported_literals(monkeypatch):
    report = GapReportBuilder(model_a_name="a", model_b_name="b", output_path="/tmp/report", top_k_literals=1)
    calls: list[str] = []

    def fake_render_token_boundaries(**kwargs):
        calls.append(kwargs["segment_text"])
        return f"|{kwargs['segment_text']}|"

    monkeypatch.setattr(gap_analysis, "render_token_boundaries", fake_render_token_boundaries)

    weaker_document = RawTextDocument(
        dataset_name="paloma/example",
        tags=("paloma/example",),
        shard_name="docs",
        row_index=0,
        text="aaa",
    )
    stronger_document = RawTextDocument(
        dataset_name="paloma/example",
        tags=("paloma/example",),
        shard_name="docs",
        row_index=1,
        text="bbb",
    )
    tokenized_weaker = TokenizedDocument(
        token_ids=np.asarray([1], dtype=np.int32),
        byte_starts=np.asarray([0], dtype=np.int32),
        byte_ends=np.asarray([3], dtype=np.int32),
        num_bytes=3,
    )
    tokenized_stronger = TokenizedDocument(
        token_ids=np.asarray([2], dtype=np.int32),
        byte_starts=np.asarray([0], dtype=np.int32),
        byte_ends=np.asarray([3], dtype=np.int32),
        num_bytes=3,
    )

    report.add_document(
        document=weaker_document,
        per_byte_loss_a=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        per_byte_loss_b=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        tokenized_a=tokenized_weaker,
        tokenized_b=tokenized_weaker,
    )
    report.add_document(
        document=stronger_document,
        per_byte_loss_a=np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
        per_byte_loss_b=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        tokenized_a=tokenized_stronger,
        tokenized_b=tokenized_stronger,
    )

    assert calls == []

    summary = report.build_summary()
    literal_rows = summary["top_literals"]["model_a_worse"]

    assert [row["name"] for row in literal_rows] == ["bbb"]
    assert literal_rows[0]["model_a_token_boundaries"] == "|bbb|"
    assert literal_rows[0]["model_b_token_boundaries"] == "|bbb|"
    assert calls == ["bbb", "bbb"]


def test_gap_report_builder_previews_worst_region():
    report = GapReportBuilder(model_a_name="a", model_b_name="b", output_path="/tmp/report")
    prefix = "safe " * 40
    target = "needle"
    suffix = " tail" * 40
    text = prefix + target + suffix
    document = RawTextDocument(
        dataset_name="paloma/example",
        tags=("paloma/example",),
        shard_name="docs",
        row_index=0,
        text=text,
    )
    loss_a = np.zeros(len(text), dtype=np.float64)
    loss_b = np.zeros(len(text), dtype=np.float64)
    target_start = text.index(target)
    target_end = target_start + len(target)
    loss_a[target_start:target_end] = 1.0

    report.add_document(document=document, per_byte_loss_a=loss_a, per_byte_loss_b=loss_b)

    summary = report.build_summary()
    doc_row = summary["top_documents"]["model_a_worse"][0]
    segment_row = next(row for row in summary["top_segments"]["model_a_worse"] if row["text"] == target)

    assert target in doc_row["preview"]
    assert doc_row["preview"].startswith("\u2026")
    assert doc_row["model_a_bpb"] > doc_row["model_b_bpb"]
    assert doc_row["worst_bucket"] == "text/word"
    assert doc_row["worst_text"] == target
    assert doc_row["worst_gap_bpb"] > 0.0
    assert target in segment_row["doc_preview"]
    assert segment_row["doc_preview"].startswith("\u2026")

    worst_rows = worst_document_rows(summary)
    assert worst_rows[0]["direction"] == "model_a_worse"
    assert worst_rows[0]["rank"] == 1
    assert worst_rows[0]["worst_text"] == target


def test_write_report_files_and_log_artifact_bundle():
    summary: dict[str, Any] = {
        "model_a": "a",
        "model_b": "b",
        "datasets": [],
        "dataset_groups": [],
        "pattern_buckets": [],
        "top_documents": {
            "model_a_worse": [{"dataset": "tiny/raw", "row_index": 3, "gap_bpb": 0.5}],
            "model_b_worse": [],
        },
        "top_segments": {"model_a_worse": [], "model_b_worse": []},
        "top_literals": {"model_a_worse": [], "model_b_worse": []},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path, report_path, worst_documents_path = write_report_files(tmpdir, summary)
        assert os.path.exists(summary_path)
        assert os.path.exists(report_path)
        assert os.path.exists(worst_documents_path)
        assert report_path.endswith("report.md")
        assert worst_documents_path.endswith("worst_documents.jsonl")
        with open(worst_documents_path) as f:
            worst_rows = [json.loads(line) for line in f]
        assert worst_rows == [
            {"dataset": "tiny/raw", "direction": "model_a_worse", "gap_bpb": 0.5, "rank": 1, "row_index": 3}
        ]

    tracker = DictTracker()
    captured: dict[str, Any] = {}

    def capture_artifact(artifact_path, *, name=None, type=None):
        captured["name"] = name
        captured["type"] = type
        captured["files"] = sorted(os.listdir(artifact_path))
        with open(os.path.join(artifact_path, "summary.json")) as f:
            captured["summary"] = json.load(f)
        with open(os.path.join(artifact_path, "worst_documents.jsonl")) as f:
            captured["worst_documents"] = [json.loads(line) for line in f]

    tracker.log_artifact = capture_artifact  # type: ignore[method-assign]

    with current_tracker(tracker):
        _log_report_artifact(summary)

    assert captured["name"] == "perplexity_gap_report"
    assert captured["type"] == "perplexity_gap_report"
    assert captured["files"] == ["report.md", "summary.json", "worst_documents.jsonl"]
    assert captured["summary"]["model_a"] == "a"
    assert captured["worst_documents"][0]["direction"] == "model_a_worse"


def test_render_report_markdown_escapes_table_boundaries():
    summary: dict[str, Any] = {
        "model_a": "a|model",
        "model_b": "b",
        "datasets": [{"name": "data|set", "documents": 1, "gap_bpb": 1.25}],
        "dataset_groups": [],
        "pattern_buckets": [],
        "top_documents": {"model_a_worse": [], "model_b_worse": []},
        "top_segments": {"model_a_worse": [], "model_b_worse": []},
        "top_literals": {
            "model_a_worse": [
                {
                    "name": "abc",
                    "model_a_token_boundaries": "|ab|c|",
                    "model_b_token_boundaries": "|a|bc|",
                }
            ],
            "model_b_worse": [],
        },
    }

    markdown = render_report_markdown(summary)

    assert "# Perplexity Gap Report" in markdown
    assert "**Model A:** a\\|model" in markdown
    assert "data\\|set" in markdown
    assert "\\|ab\\|c\\|" in markdown


def test_check_finite_losses_raises_on_non_finite_values():
    with pytest.raises(ValueError, match="checkpoint and tokenizer are incompatible"):
        _check_finite_losses("bad-model", np.asarray([[float("nan")]], dtype=np.float64))


def test_truncate_text_to_byte_limit_respects_utf8_boundaries():
    text = "café🙂z"

    assert _truncate_text_to_byte_limit(text, 3) == "caf"
    assert _truncate_text_to_byte_limit(text, 5) == "café"
    assert _truncate_text_to_byte_limit(text, 9) == "café🙂"
    assert _truncate_text_to_byte_limit(text, 10) == text


def test_accumulate_token_losses_matches_naive_interval_scatter():
    out = np.zeros(7, dtype=np.float64)
    starts = np.asarray([0, 2, -1, 4], dtype=np.int32)
    ends = np.asarray([2, 5, -1, 7], dtype=np.int32)
    losses = np.asarray([0.6, 0.9, 10.0, 1.5], dtype=np.float64)

    _accumulate_token_losses(out, starts, ends, losses)

    expected = np.zeros(7, dtype=np.float64)
    for loss, start, end in zip(losses, starts, ends, strict=True):
        if start < 0 or end <= start:
            continue
        expected[start:end] += float(loss) / (end - start)

    assert np.allclose(out, expected)


def test_chunk_tokenized_document_overlaps_boundary_context():
    document = TokenizedDocument(
        token_ids=np.arange(7, dtype=np.int32),
        byte_starts=np.arange(7, dtype=np.int32),
        byte_ends=np.arange(1, 8, dtype=np.int32),
        num_bytes=7,
    )

    chunks = chunk_tokenized_document(document, max_eval_length=4, doc_index=2)

    assert [chunk.token_ids.tolist() for chunk in chunks] == [[0, 1, 2, 3], [3, 4, 5, 6]]
    assert [chunk.doc_index for chunk in chunks] == [2, 2]
    target_byte_starts = np.concatenate([chunk.byte_starts[1:] for chunk in chunks])
    assert target_byte_starts.tolist() == [1, 2, 3, 4, 5, 6]


def test_batch_chunks_rejects_oversized_chunks():
    chunk = TokenizedChunk(
        doc_index=0,
        token_ids=np.arange(4, dtype=np.int32),
        byte_starts=np.arange(4, dtype=np.int32),
        byte_ends=np.arange(1, 5, dtype=np.int32),
    )

    with pytest.raises(ValueError, match="exceeds max_eval_length"):
        list(batch_chunks([chunk], batch_size=1, max_eval_length=3))


@pytest.mark.entry
def test_perplexity_gap_main_same_model_zero_gap():
    model_config = LlamaConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        hidden_dim=32,
        max_seq_len=64,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "validation.jsonl")
        with open(data_path, "w") as f:
            f.write(json.dumps({"text": "hello  world\n"}) + "\n")
            f.write(json.dumps({"text": "tabs\tand café\n"}) + "\n")

        tokenizer = load_tokenizer("gpt2")
        vocab = haliax.Axis("vocab", len(tokenizer))
        model = LlamaLMHeadModel.init(vocab, model_config, key=jax.random.PRNGKey(0))
        ckpt_path = os.path.join(tmpdir, "ckpt")
        save_checkpoint({"model": model}, 0, ckpt_path)

        datasets = {
            "tiny/raw": DatasetComponent(
                source=UrlDatasetSourceConfig(
                    validation_urls=[f"file://{data_path}"],
                    format=TextLmDatasetFormat(),
                ),
                format=TextLmDatasetFormat(),
            )
        }

        config = GapFinderConfig(
            model_a=GapFinderModelConfig(
                checkpoint_path=ckpt_path,
                model=model_config,
                checkpoint_is_hf=False,
                tokenizer="gpt2",
            ),
            model_b=GapFinderModelConfig(
                checkpoint_path=ckpt_path,
                model=model_config,
                checkpoint_is_hf=False,
                tokenizer="gpt2",
            ),
            datasets=datasets,
            trainer=TrainerConfig(
                per_device_eval_parallelism=len(jax.devices()),
                tracker=NoopConfig(),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
            output_path=os.path.join(tmpdir, "gap"),
            max_eval_length=32,
            max_docs_per_dataset=2,
        )

        main(config)

        with open(os.path.join(tmpdir, "gap", "summary.json")) as f:
            summary = json.load(f)

        assert summary["datasets"]
        assert math.isclose(summary["datasets"][0]["gap_bpb"], 0.0, abs_tol=1e-7)
