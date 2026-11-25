# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import draccus
import jax
import numpy as np
import pytest
from haliax import Axis
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig
from levanter.data.text.formats import SerializedChatLmDatasetFormat, TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.store.cache import SerialCacheWriter
from levanter.trainer import TrainerConfig
from levanter.utils.thread_utils import blocking_wait
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.long_loop import sft_accounting
from experiments.post_training.long_loop.archive import seal_archive, verify_archive
from experiments.post_training.long_loop.sft_accounting import (
    sequence_statistics,
    sft_accounting_inputs,
    write_sft_accounting,
)


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("concatenated", [False, True])
def test_committed_native_samples_and_population_reports_survive_in_archive(tmp_path, monkeypatch, pack, concatenated):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")),
        unk_token="t0",
        eos_token="t9",
        bos_token="t8",
    )
    tokenizer.save_pretrained(tokenizer_path)
    # Unpacked transcripts must fit the sequence length: over-long documents raise
    # instead of being silently truncated, so the unpacked axis uses fitting sizes.
    sizes = [2, 6, 19, 25, 9] if pack else [2, 6, 9, 10, 9]
    documents = [np.asarray([1, 7, 3, 4] + [50 + i] * size + [9], dtype=np.int32) for i, size in enumerate(sizes)]
    chat = tmp_path / "chat"
    replay = tmp_path / "replay"
    with SerialCacheWriter(str(chat), {"input_ids": np.empty(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": tokens} for tokens in documents])
    with SerialCacheWriter(str(replay), {"input_ids": np.empty(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": np.arange(64 if pack else 16, dtype=np.int32)}])
    chat_component = DatasetComponent(
        cache_dir=str(chat),
        flat_cache=True,
        pack=pack,
        format=SerializedChatLmDatasetFormat(message_start_id=1, assistant_prefix=(1, 7, 3, 4)),
    )
    if concatenated:
        second = tmp_path / "chat-second"
        with SerialCacheWriter(str(second), {"input_ids": np.empty(0, dtype=np.int32)}) as writer:
            writer.write_batch([{"input_ids": tokens} for tokens in reversed(documents)])
        chat_component = ConcatDatasetComponent(
            children={"first": chat_component, "second": replace(chat_component, cache_dir=str(second))}
        )
    data = LmDataConfig(
        tokenizer=str(tokenizer_path),
        cache_dir=str(tmp_path / "unused"),
        auto_build_caches=False,
        components={
            "sft/chat": chat_component,
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
        data=data, train_seq_len=16, data_seed=31, trainer=TrainerConfig(seed=17, train_batch_size=40, num_train_steps=8)
    )
    native = tmp_path / "trained"
    checkpoint = native / "checkpoints/step-7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text('{"step":7,"is_temporary":false}')
    (native / "selected.json").write_text(
        json.dumps({"step": 7, "completed_updates": 8, "checkpoint": str(checkpoint), "format": "SFTSpec"})
    )
    consumed = draccus.encode(config)
    # Data replay must work in a fresh CPU process without the producer's optimizer plugins.
    consumed["optimizer"] = {"type": "producer-only-optimizer"}
    (native / "consumed-config.json").write_text(json.dumps({"format": "SFTSpec", "train_config": consumed}))
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
    monkeypatch.setattr(sft_accounting, "ACCOUNTING_CHECKPOINT_SAMPLES", 37)
    if not pack and not concatenated:
        replay_rows = sft_accounting.sampled_rows

        async def interrupted_rows(*args, **kwargs):
            async for row in replay_rows(*args, **kwargs):
                if row["global_index"] == 40:
                    raise RuntimeError("accounting worker interrupted")
                yield row

        monkeypatch.setattr(sft_accounting, "sampled_rows", interrupted_rows)
        with pytest.raises(RuntimeError, match="worker interrupted"):
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
        assert (output / "progress/samples-000000037.jsonl.pin.json").is_file()
        assert not (output / "receipt.json").exists()
        monkeypatch.setattr(sft_accounting, "sampled_rows", replay_rows)
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
    examples = blocking_wait(actual.get_batch(list(range(320))))
    for row, example in zip(rows, examples, strict=True):
        tokens = np.asarray(example.tokens.array)
        assert row["retained_tokens_sha256"] == hashlib.sha256(tokens[: row["real_tokens"]].tobytes()).hexdigest()
        assert row["loss_tokens"] == int(np.count_nonzero(np.asarray(example.loss_weight.array)))
        if row["is_sft"]:
            child_documents = list(reversed(documents)) if row["child"] == "second" else documents
            originals = [child_documents[doc["row"]] for doc in row["documents"]]
            assert row["original_tokens"] == sum(map(len, originals))
            np.testing.assert_array_equal(tokens[: row["real_tokens"]], np.concatenate([doc[:16] for doc in originals]))
    report = json.loads((output / "realized-truncation-padding.json").read_text())
    assert report["overall"]["sequences"] == 320
    assert report["overall"]["positions"] == 5120
    # Packed runs truncate over-long documents; unpacked ones raise instead, so only
    # the packed axis records left-truncated traces.
    if pack:
        assert report["overall"]["left_truncated_traces"] > 0
    else:
        assert report["overall"]["left_truncated_traces"] == 0
    assert report["overall"]["pad_tokens"] > 0
    assert report["overall"]["real_tokens"] + report["overall"]["pad_tokens"] == 5120
    replay = [row for row in rows if not row["is_sft"]]
    assert report["overall"]["replay"]["loss_tokens"] == sum(row["loss_tokens"] for row in replay)
    assert report["overall"]["replay"]["fraction_of_positions"] == len(replay) / len(rows)
    prefix = output / "progress/samples-000000037.jsonl"
    assert prefix.read_bytes() == b"".join(
        (output / "realized-samples.jsonl").read_bytes().splitlines(keepends=True)[:37]
    )
    resumed = tmp_path / "resumed"
    resume_pin = json.loads(prefix.with_suffix(".jsonl.pin.json").read_bytes())
    write_sft_accounting(
        output=str(resumed),
        trained=str(native),
        settings={
            "pinned_evidence": pins,
            "measured_step_seconds": 2.0,
            "timing_source": "tiny CPU fixture",
            "pack": pack,
            "resume_samples": resume_pin,
        },
    )
    assert (resumed / "realized-samples.jsonl").read_bytes() == (output / "realized-samples.jsonl").read_bytes()
    resumed_report = json.loads((resumed / "realized-truncation-padding.json").read_text())
    assert resumed_report == {**report, "resumed_prefix": resume_pin}
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
    modern = tmp_path / "accounting-with-separate-population"
    modern.mkdir()
    population = {name: pins[name] for name in ("truncation-census.json", "population-padding.json", "decision.md")}
    producer_files = [item for item in receipt["files"] if item["name"] not in population]
    for item in producer_files:
        (modern / item["name"]).write_bytes((output / item["name"]).read_bytes())
    (modern / "receipt.json").write_text(json.dumps({**receipt, "files": producer_files}))
    with pytest.raises(ValueError, match="lacks population"):
        sft_accounting_inputs(str(modern))
    modern_inputs, modern_receipt = sft_accounting_inputs(str(modern), population)
    assert modern_receipt["supplemental_population_evidence"] == population
    seal_archive(str(tmp_path / "modern-archive"), modern_inputs, {"sft_data_accounting": modern_receipt})
    decision = tmp_path / "decision.md"
    decision.write_text("changed after pinning")
    with pytest.raises(ValueError, match="Supplemental accounting evidence changed"):
        sft_accounting_inputs(str(modern), population)
    (output / "realized-truncation-padding.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        sft_accounting_inputs(str(output))


@pytest.mark.parametrize("cap,packed_sequences", [(64, 5), (256, 2)])
def test_sample_repacking_comparison_respects_the_configured_segment_cap(cap, packed_sequences):
    rows = [
        {
            "real_tokens": 1,
            "original_tokens": 1,
            "loss_tokens": 1,
            "is_sft": True,
            "documents": [{"original_tokens": 1, "real_tokens": 1}],
        }
        for _ in range(257)
    ]
    statistics = sequence_statistics(rows, 4096, 8, 1.0, max_segments=cap)
    assert statistics["greedy_packed_retained_sequences"] == packed_sequences
    assert statistics["greedy_packed_update_equivalent"] == packed_sequences / 8
    assert statistics["real_tokens"] == 257
