# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.analysis.document_losses import DocumentSourceConfig
from levanter.analysis.perplexity_gap import tokenize_text_with_byte_spans
from levanter.checkpoint import save_checkpoint
from levanter.distributed import DistributedConfig
from levanter.main.eval_documents import EvalDocumentsConfig, main
from levanter.main.perplexity_gap import GapFinderModelConfig
from levanter.testing.toy_lm import ToyLmConfig, ToyLmHeadModel
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig


def test_eval_documents_exports_whole_document_loss_and_utf8_bits_per_byte(
    tmp_path, local_gpt2_tokenizer, local_gpt2_tokenizer_path
):
    tokenizer = local_gpt2_tokenizer
    model_config = ToyLmConfig(max_seq_len=8, embed_dim=16)
    model = ToyLmHeadModel.init(hax.Axis("vocab", len(tokenizer)), model_config, key=jax.random.PRNGKey(0))
    model = dataclasses.replace(model, aux_loss=jnp.array(7.0))
    checkpoint = str(tmp_path / "checkpoint")
    save_checkpoint({"model": model}, 0, checkpoint)
    texts = ["hello world " * 5, "café 東京", ""]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "".join(json.dumps({"id": str(index), "text": text}) + "\n" for index, text in enumerate(texts))
    )
    output_path = tmp_path / "losses.jsonl"
    config = EvalDocumentsConfig(
        model=GapFinderModelConfig(
            checkpoint_path=checkpoint, model=model_config, tokenizer=local_gpt2_tokenizer_path
        ),
        source=DocumentSourceConfig(input_path=str(input_path), corpus_id="test-corpus"),
        output_path=str(output_path),
        max_eval_length=8,
        trainer=TrainerConfig(
            per_device_eval_parallelism=2,
            tracker=NoopConfig(),
            require_accelerator=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
        ),
    )
    main(config)
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 3
    for index, (text, record) in enumerate(zip(texts, records, strict=True)):
        assert set(record) == {"doc_id", "corpus_id", "loss", "bits_per_byte"}
        assert record["doc_id"] == str(index)
        assert record["corpus_id"] == "test-corpus"
        if not text:
            assert record["loss"] is None
            assert record["bits_per_byte"] is None
            continue
        tokens = tokenize_text_with_byte_spans(tokenizer, tokenizer.as_hf_tokenizer(), text).token_ids
        # ToyLm is context-independent, so the unsplit NumPy projection is an
        # independent oracle for the chunked, padded JAX scoring path.
        logits = np.asarray(model.embed_weight.array)[tokens[:-1]] @ np.asarray(model.lm_head.array)
        losses = np.logaddexp.reduce(logits, axis=-1) - logits[np.arange(len(tokens) - 1), tokens[1:]]
        np.testing.assert_allclose(record["loss"], losses.mean(), rtol=1e-5)
        np.testing.assert_allclose(
            record["bits_per_byte"], losses.sum() / (math.log(2) * len(text.encode("utf-8"))), rtol=1e-5
        )
