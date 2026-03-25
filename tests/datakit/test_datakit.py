# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the datakit pipeline: download → normalize → tokenize, wired as StepSpecs."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from levanter.store.cache import CacheLedger, TreeCache

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import content_hash_id, normalize_step
from marin.datakit.tokenize import tokenize_step
from marin.execution.step_runner import StepRunner


@pytest.mark.slow
def test_download_normalize_tokenize(tmp_path):
    """Download → normalize → tokenize as a StepSpec DAG via StepRunner."""

    dl = download_hf_step(
        "datakit/download",
        hf_dataset_id="wikitext",
        revision="main",
        hf_urls_glob=["wikitext-2-v1/test-*.parquet"],
        override_output_path=str(tmp_path / "raw"),
    )

    norm = normalize_step(
        "datakit/normalize",
        input_path=dl.output_path,
        deps=[dl],
        override_output_path=str(tmp_path / "normalized"),
    )

    tok = tokenize_step(
        "datakit/tokenize",
        input_path=norm.output_path,
        tokenizer="gpt2",
        deps=[norm],
        override_output_path=str(tmp_path / "tokenized"),
    )

    StepRunner().run([dl, norm, tok])

    # -- Verify download output --
    raw_files = [f for f in Path(dl.output_path).rglob("*") if f.is_file() and not f.name.startswith(".")]
    assert len(raw_files) >= 1

    # -- Verify normalize output --
    parquet_files = sorted(Path(norm.output_path).glob("*.parquet"))
    assert len(parquet_files) >= 1

    all_records = []
    for pf in parquet_files:
        records = pq.read_table(str(pf)).to_pylist()
        all_records.extend(records)
        ids = [r["id"] for r in records]
        assert ids == sorted(ids), f"Partition {pf.name} not sorted by id"

    assert len(all_records) > 0
    for record in all_records:
        assert record["id"] == content_hash_id(record["text"])

    # -- Verify tokenize output --
    train_dir = Path(tok.output_path) / "train"
    ledger = CacheLedger.load(str(train_dir))
    assert ledger.is_finished
    assert ledger.total_num_rows > 0

    exemplar = {"input_ids": np.array([0], dtype=np.int32)}
    cache = TreeCache.load(str(train_dir), exemplar=exemplar)
    assert len(cache) == ledger.total_num_rows
    assert len(cache[0]["input_ids"]) > 0
