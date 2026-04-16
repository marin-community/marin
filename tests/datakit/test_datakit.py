# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the datakit pipeline: download → tokenize, wired as StepSpecs."""

from pathlib import Path

import numpy as np
import pytest
from levanter.store.cache import CacheLedger, TreeCache

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize


@pytest.mark.slow
def test_download_and_tokenize(tmp_path):
    """Download → tokenize as a StepSpec DAG via StepRunner."""

    dl = download_hf_step(
        "datakit/download",
        hf_dataset_id="wikitext",
        revision="main",
        hf_urls_glob=["wikitext-2-v1/test-*.parquet"],
        override_output_path=str(tmp_path / "raw"),
    )

    tok = StepSpec(
        name="datakit/tokenize",
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[dl.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer="gpt2",
                allow_test_in_train=True,
            )
        ),
        deps=[dl],
        hash_attrs={"tokenizer": "gpt2"},
        override_output_path=str(tmp_path / "tokenized"),
    )

    StepRunner().run([dl, tok])

    # -- Verify download output --
    raw_files = [f for f in Path(dl.output_path).rglob("*") if f.is_file() and not f.name.startswith(".")]
    assert len(raw_files) >= 1

    # -- Verify tokenize output --
    train_dir = Path(tok.output_path) / "train"
    ledger = CacheLedger.load(str(train_dir))
    assert ledger.is_finished
    assert ledger.total_num_rows > 0

    exemplar = {"input_ids": np.array([0], dtype=np.int32)}
    cache = TreeCache.load(str(train_dir), exemplar=exemplar)
    assert len(cache) == ledger.total_num_rows
    assert len(cache[0]["input_ids"]) > 0
