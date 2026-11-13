# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from experiments.post_training.mochi_one_rack.data import AGENTIC_IDS, NONAGENTIC_IDS, write_context_trial_data


def test_context_data_reaches_beyond_the_original_ledger_and_preserves_the_verifier(tmp_path: Path) -> None:
    retained = tmp_path / "retained"
    retained.mkdir()
    tasks = [
        {
            "id": task_id,
            "context": {"events": [{"role": "user", "content": "What is six times seven?"}]},
            "tags": ["original"],
            "source": {"dataset": "original", "revision": "fixed", "row": str(index)},
            "verifier": {"kind": "exact", "answer": "42"},
        }
        for index, task_id in enumerate((*AGENTIC_IDS, *NONAGENTIC_IDS))
    ]
    source = retained / "train.parquet"
    pq.write_table(pa.table({"task_spec": [json.dumps(task) for task in tasks]}), source)
    original = source.read_bytes()
    tokenizer_root = tmp_path / "tokenizer"
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(tokenizer_root)
    (tokenizer_root / "config.json").write_text(json.dumps({"max_position_embeddings": 131072}))
    output = tmp_path / "context"
    write_context_trial_data(
        retained_data=str(retained),
        retained_data_sha256=hashlib.sha256(original).hexdigest(),
        tokenizer_source=str(tokenizer_root),
        output_path=str(output),
        context_cap=65536,
        updates=2,
        max_new_tokens=1024,
    )
    actual = [json.loads(row["task_spec"]) for row in pq.read_table(output / "train.parquet").to_pylist()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["train_parquet_sha256"] == hashlib.sha256((output / "train.parquet").read_bytes()).hexdigest()
    assert source.read_bytes() == original
    assert actual[0] == tasks[0]
    assert actual[2] == tasks[1]
    for probe in (actual[1], actual[3]):
        assert probe["verifier"] == tasks[2]["verifier"]
        text = probe["context"]["events"][-1]["content"]
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        assert 65536 * 0.95 < tokens <= 65536 - 1024
        assert text.endswith("What is six times seven?")
        assert "controlled-length-probe" in probe["tags"]
