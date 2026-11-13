# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import draccus
import jax
import numpy as np
import pytest
from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import SerializedChatLmDatasetFormat, TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.store.cache import SerialCacheWriter
from levanter.trainer import TrainerConfig
from levanter.utils.thread_utils import blocking_wait
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.long_loop.archive import seal_archive, verify_archive
from experiments.post_training.long_loop.sft_accounting import sft_accounting_inputs, write_sft_accounting


@pytest.mark.parametrize("pack", [False, True])
def test_committed_native_samples_and_population_reports_survive_in_archive(tmp_path, pack):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")),
        unk_token="t0",
        eos_token="t9",
        bos_token="t8",
    )
    tokenizer.save_pretrained(tokenizer_path)
    documents = [
        np.asarray([1, 7, 3, 4] + [50 + i] * size + [9], dtype=np.int32) for i, size in enumerate([2, 6, 19, 25, 9])
    ]
    chat = tmp_path / "chat"
    replay = tmp_path / "replay"
    with SerialCacheWriter(str(chat), {"input_ids": np.empty(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": tokens} for tokens in documents])
    with SerialCacheWriter(str(replay), {"input_ids": np.empty(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": np.arange(64, dtype=np.int32)}])
    data = LmDataConfig(
        tokenizer=str(tokenizer_path),
        cache_dir=str(tmp_path / "unused"),
        auto_build_caches=False,
        components={
            "sft/chat": DatasetComponent(
                cache_dir=str(chat),
                flat_cache=True,
                pack=pack,
                format=SerializedChatLmDatasetFormat(message_start_id=1, assistant_prefix=(1, 7, 3, 4)),
            ),
            "pretrain/text": DatasetComponent(
                cache_dir=str(replay), flat_cache=True, pack=False, format=TextLmDatasetFormat()
            ),
        },
        train_weights=[(0, {"sft/chat": 0.75, "pretrain/text": 0.25}), (4, {"sft/chat": 0.5, "pretrain/text": 0.5})],
        num_validation_sequences={"sft/chat": 1},
        mixture_block_size=8,
        shuffle=True,
        permutation_type="feistel",
        block_cross_document_attention=pack,
    )
    config = TrainLmConfig(
        data=data, train_seq_len=16, data_seed=31, trainer=TrainerConfig(seed=17, train_batch_size=2, num_train_steps=8)
    )
    native = tmp_path / "trained"
    checkpoint = native / "checkpoints/step-8"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text('{"step":8,"is_temporary":false}')
    (native / "selected.json").write_text(json.dumps({"step": 8, "checkpoint": str(checkpoint), "format": "SFTSpec"}))
    (native / "consumed-config.json").write_text(
        json.dumps({"format": "SFTSpec", "train_config": draccus.encode(config)})
    )
    pins = {}
    for name, content in {
        "truncation-census.json": '{"unique_sft_caches":1}',
        "population-padding.json": '{"pad_fraction":0.25}',
        "decision.md": "User-approved context/packing; retain historical sources.",
    }.items():
        path = tmp_path / name
        path.write_text(content)
        pins[name] = {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    output = tmp_path / "accounting"
    write_sft_accounting(
        output=str(output),
        trained=str(native),
        settings={
            "pinned_evidence": pins,
            "measured_step_seconds": 2.0,
            "timing_source": "tiny CPU fixture",
            "pack": pack,
        },
    )
    rows = [json.loads(line) for line in (output / "realized-samples.jsonl").read_text().splitlines()]
    # Compare the manifest against the real NamedLmDataset boundary consumed by train_lm.
    actual = data.train_set(Axis("position", 16), config.trainer.batch_schedule, key=jax.random.PRNGKey(31))
    examples = blocking_wait(actual.get_batch(list(range(16))))
    for row, example in zip(rows, examples, strict=True):
        tokens = np.asarray(example.tokens.array)
        assert row["retained_tokens_sha256"] == hashlib.sha256(tokens[: row["real_tokens"]].tobytes()).hexdigest()
        assert row["loss_tokens"] == int(np.count_nonzero(np.asarray(example.loss_weight.array)))
        if row["is_sft"]:
            originals = [documents[doc["row"]] for doc in row["documents"]]
            assert row["original_tokens"] == sum(map(len, originals))
            np.testing.assert_array_equal(tokens[: row["real_tokens"]], np.concatenate([doc[:16] for doc in originals]))
    report = json.loads((output / "realized-truncation-padding.json").read_text())
    assert report["overall"]["sequences"] == 16
    assert report["overall"]["positions"] == 256
    assert report["overall"]["left_truncated_traces"] > 0
    assert report["overall"]["pad_tokens"] > 0
    assert report["overall"]["real_tokens"] + report["overall"]["pad_tokens"] == 256
    inputs, receipt = sft_accounting_inputs(str(output))
    archive = tmp_path / "archive"
    seal_archive(str(archive), inputs, {"sft_data_accounting": receipt})
    manifest = verify_archive(str(archive))
    destinations = {item["destination"]: item["sha256"] for item in manifest["objects"]}
    for name in [
        "truncation-census.json",
        "population-padding.json",
        "realized-truncation-padding.json",
        "realized-samples.jsonl",
        "decision.md",
    ]:
        assert (
            destinations[str(archive / "objects/sft-data" / name)]
            == hashlib.sha256((output / name).read_bytes()).hexdigest()
        )
    (output / "realized-truncation-padding.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        sft_accounting_inputs(str(output))
