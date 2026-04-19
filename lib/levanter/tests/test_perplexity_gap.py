# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import tempfile

import jax
import numpy as np
import pytest

import haliax

from levanter.analysis.perplexity_gap import (
    GapReportBuilder,
    RawTextDocument,
    TokenizedDocument,
    _truncate_text_to_byte_limit,
    tokenize_text_with_byte_spans,
)
from levanter.checkpoint import save_checkpoint
from levanter.data.text import DatasetComponent, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.distributed import DistributedConfig
from levanter.main.perplexity_gap import (
    GapFinderConfig,
    GapFinderModelConfig,
    _accumulate_token_losses,
    main,
)
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
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
