# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise standalone document evaluation on the frozen Snowball 67B-A2B export.

Run on one H100x8 node in cw-us-east-02a, colocated with the checkpoint::

    uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
      --target-cluster cw-us-east-02a --gpu H100x8 --cpu 32 --memory 512g --disk 256g \
      --extra gpu --sync-package marin-core --sync-package marin-levanter \
      --timeout 3600 --job-name snowball-document-losses \
      -- python -m experiments.evaluation.snowball_document_losses
"""

import dataclasses
import json
import logging
import math
import os
import uuid
from pathlib import Path

import fsspec
import jmp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from levanter.analysis.document_losses import DocumentSourceConfig
from levanter.analysis.perplexity_gap import tokenize_text_with_byte_spans
from levanter.main.eval_documents import EvalDocumentsConfig
from levanter.main.eval_documents import main as evaluate_documents
from levanter.main.perplexity_gap import GapFinderModelConfig
from levanter.models.snowball import SnowballConfig
from levanter.tokenizers import load_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.testing.inference.snowball import SNOWBALL, read_prompt_fixture, read_representative_goldens

logger = logging.getLogger(__name__)
MAX_EVAL_LENGTH = 256


def main():
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = read_prompt_fixture(read_representative_goldens())
    tokenizer_path = snapshot_download(
        fixture.tokenizer,
        revision=fixture.tokenizer_revision,
        allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "chat_template*"],
    )
    tokenizer = load_tokenizer(tokenizer_path)
    hf_tokenizer = tokenizer.as_hf_tokenizer()
    cases = sorted(fixture.cases, key=lambda case: case.id)[:8]
    documents = [
        {
            "doc_id": case.id,
            "corpus_id": "snowball/frozen-prompts",
            "text": hf_tokenizer.decode(case.prompt_token_ids[:96], skip_special_tokens=True),
        }
        for case in cases
    ]
    documents.extend(
        [
            {
                "doc_id": "long-document",
                "corpus_id": "snowball/smoke",
                "text": "The snowy mountains reflect sunlight across the valley. " * 100 + "雪山",
            },
            {**documents[0], "doc_id": "duplicate-content"},
            {"doc_id": "empty-document", "corpus_id": "snowball/smoke", "text": ""},
        ]
    )
    prefix = f"s3://marin-us-east-02a/tmp/ttl=30d/document-losses-8975/{uuid.uuid4().hex}"
    jsonl_uri, parquet_uri = f"{prefix}/inputs/input.jsonl", f"{prefix}/inputs/input.parquet"
    output_uri = f"{prefix}/document-losses.jsonl"
    with fsspec.open(jsonl_uri, "wt") as stream:
        for doc in documents[:6]:
            stream.write(json.dumps(doc) + "\n")
    with fsspec.open(parquet_uri, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(documents[6:]), stream)

    converter = SnowballConfig(
        reference_checkpoint=SNOWBALL.export_uri, tokenizer=tokenizer_path
    ).hf_checkpoint_converter()
    model_config = dataclasses.replace(
        converter.config_from_hf_config(converter.default_hf_config),
        tokenizer=tokenizer_path,
        moe_implementation="sonic",
    )
    config = EvalDocumentsConfig(
        model=GapFinderModelConfig(
            checkpoint_path=SNOWBALL.export_uri,
            checkpoint_is_hf=True,
            model=model_config,
            tokenizer=tokenizer_path,
        ),
        source=DocumentSourceConfig(input_path=f"{prefix}/inputs", doc_id_field="doc_id"),
        output_path=output_uri,
        max_eval_length=MAX_EVAL_LENGTH,
        trainer=TrainerConfig(
            tracker=NoopConfig(),
            per_device_eval_parallelism=1,
            mp=jmp.get_policy("params=bfloat16,compute=bfloat16,output=float32"),
            use_explicit_mesh_axes=True,
            mesh=MeshConfig(
                axes={"data": -1, "expert": 1, "model": 1},
                compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
            ),
        ),
    )
    evaluate_documents(config)
    with fsspec.open(output_uri, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    by_id = {row["doc_id"]: row for row in rows}
    assert len(rows) == len(documents)
    assert set(by_id) == {doc["doc_id"] for doc in documents}
    token_counts = {}
    for document in documents:
        row = by_id[document["doc_id"]]
        assert set(row) == {"doc_id", "corpus_id", "loss", "bits_per_byte"}
        assert row["corpus_id"] == document["corpus_id"]
        if not document["text"]:
            assert row["loss"] is None and row["bits_per_byte"] is None
            continue
        tokenized = tokenize_text_with_byte_spans(tokenizer, hf_tokenizer, document["text"])
        count = len(tokenized.token_ids) - 1
        token_counts[document["doc_id"]] = count
        assert math.isfinite(row["loss"]) and row["loss"] >= 0
        expected_bpb = row["loss"] * count / (math.log(2) * len(document["text"].encode("utf-8")))
        np.testing.assert_allclose(row["bits_per_byte"], expected_bpb, rtol=1e-5, atol=1e-5)
    assert token_counts["long-document"] > MAX_EVAL_LENGTH
    np.testing.assert_allclose(by_id["duplicate-content"]["loss"], by_id[documents[0]["doc_id"]]["loss"], rtol=1e-5)
    summary = {
        "checkpoint": SNOWBALL.export_uri,
        "inputs": [jsonl_uri, parquet_uri],
        "output": output_uri,
        "tokenizer": fixture.tokenizer,
        "tokenizer_revision": fixture.tokenizer_revision,
        "max_eval_length": MAX_EVAL_LENGTH,
        "documents": len(rows),
        "token_counts": token_counts,
        "results": rows,
        "validation": "passed",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "document-losses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    logger.info("SNOWBALL_DOCUMENT_LOSSES_VALIDATED %s", json.dumps(summary))


if __name__ == "__main__":
    main()
