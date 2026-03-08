# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import cloudpickle
import json
import logging
import uuid
from io import StringIO

import numpy as np
from PIL import Image
from fray.cluster import ResourceConfig

from experiments.jpeg_tokenizer.base.data import (
    InMemoryTokenSequenceDataset,
    TokenSequenceRecord,
    TokenStoreMetadata,
    TokenStoreSplitInfo,
    as_causal_lm_dataset,
    build_passthrough_lm_data_config,
    build_passthrough_lm_data_config_from_store,
    open_token_matrix_dataset,
    read_token_store_manifest,
    read_token_store_metadata,
    write_token_store,
)
from experiments.jpeg_tokenizer.base.eval import compute_reconstruction_metrics, compute_token_sequence_stats
from experiments.jpeg_tokenizer.base.model import JpegLmConfig
from experiments.jpeg_tokenizer.base.train import JpegEvalConfig, JpegRunConfig, JpegTrainerConfig, run_jpeg_tokenizer
from experiments.jpeg_tokenizer.base.jpeg_codecs import (
    canonicalize_image,
    encode_dct_coeffs,
    encode_jpeg_bytes,
    encode_jpeg_symbols,
    reconstruct_luma_from_coeff_tokens,
    stable_byte_checksum,
)
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig, RayConfig
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig


def test_in_memory_token_dataset_and_passthrough_config_round_trip():
    sequences = [
        np.arange(8, dtype=np.int32),
        np.arange(8, 16, dtype=np.int32),
    ]
    train_tokens = InMemoryTokenSequenceDataset(sequences)
    validation_tokens = InMemoryTokenSequenceDataset(sequences[:1])

    train_dataset = as_causal_lm_dataset(train_tokens, seq_len=8)
    validation_dataset = as_causal_lm_dataset(validation_tokens, seq_len=8)
    data_config = build_passthrough_lm_data_config(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        vocab_size=32,
    )

    assert data_config.tokenizer == "passthrough"
    validation_sets = data_config.validation_grug_sets(seq_len=8)
    assert set(validation_sets.keys()) == {"jpeg_tokens"}
    first_example = asyncio.run(validation_sets["jpeg_tokens"].get_batch([0]))
    assert first_example[0].tokens.shape == (8,)


def test_phase0_helpers_produce_stable_byte_tokens_and_stats():
    token_sequences = [
        encode_jpeg_bytes(b"\x01\x02\x03"),
        encode_jpeg_bytes(b"\x02\x03\x04\x05"),
    ]

    stats = compute_token_sequence_stats(token_sequences)

    assert stats.num_examples == 2
    assert stats.min_length == 3
    assert stats.max_length == 4
    assert stats.unique_tokens == 5
    assert stable_byte_checksum(b"jpeg") == stable_byte_checksum(b"jpeg")


def test_canonicalization_and_reference_tokenizers_are_deterministic():
    pixels = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    image = Image.fromarray(pixels, mode="L").convert("RGB")

    canonical = canonicalize_image(image)
    repeated = canonicalize_image(image)

    assert canonical.jpeg_bytes == repeated.jpeg_bytes
    assert canonical.checksum == repeated.checksum
    assert canonical.luma_plane.shape == (256, 256)

    coeff_tokens = encode_dct_coeffs(canonical)
    symbol_tokens = encode_jpeg_symbols(canonical)

    assert coeff_tokens.ndim == 1
    assert symbol_tokens.ndim == 1
    assert coeff_tokens.min() >= 0
    assert symbol_tokens.min() >= 0


def test_coeff_reconstruction_metrics_are_well_formed():
    pixels = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    canonical = canonicalize_image(image)
    coeff_tokens = encode_dct_coeffs(canonical)
    reconstructed = reconstruct_luma_from_coeff_tokens(coeff_tokens, canonical.luma_plane.shape)

    metrics = compute_reconstruction_metrics(canonical.luma_plane, reconstructed)

    assert reconstructed.shape == canonical.luma_plane.shape
    assert metrics.mse >= 0.0
    assert metrics.psnr > 0.0
    assert 0.0 <= metrics.ssim <= 1.0
    assert 0.0 <= metrics.dssim <= 0.5


def test_file_backed_token_store_round_trip(tmp_path):
    store_dir = tmp_path / "coeff_store"
    metadata = TokenStoreMetadata(
        dataset="dummy",
        dataset_config="v0",
        image_column="image",
        vocab_size=4095,
        seq_len=4,
        canonical_config={"resolution": 256},
        tokenizer_config={"zigzag_coefficients": 4},
        splits={
            "train": TokenStoreSplitInfo(
                num_examples=2,
                seq_len=4,
                tokens_path="train_tokens.npy",
                manifest_path="train_manifest.jsonl",
            ),
            "validation": TokenStoreSplitInfo(
                num_examples=1,
                seq_len=4,
                tokens_path="validation_tokens.npy",
                manifest_path="validation_manifest.jsonl",
            ),
        },
    )
    split_tokens = {
        "train": np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
        "validation": np.asarray([[9, 10, 11, 12]], dtype=np.int32),
    }
    split_records = {
        "train": [
            TokenSequenceRecord(example_id="train:0", split="train", num_tokens=4, checksum="a", source_index=0),
            TokenSequenceRecord(example_id="train:1", split="train", num_tokens=4, checksum="b", source_index=1),
        ],
        "validation": [
            TokenSequenceRecord(
                example_id="validation:0",
                split="validation",
                num_tokens=4,
                checksum="c",
                source_index=0,
            )
        ],
    }

    write_token_store(store_dir, metadata=metadata, split_tokens=split_tokens, split_records=split_records)

    loaded_metadata = read_token_store_metadata(store_dir)
    assert loaded_metadata.seq_len == 4
    assert loaded_metadata.vocab_size == 4095
    assert len(read_token_store_manifest(store_dir, "train")) == 2

    train_dataset = open_token_matrix_dataset(store_dir, "train")
    batch = asyncio.run(train_dataset.get_batch([1, 0]))
    assert batch[0].tolist() == [5, 6, 7, 8]
    assert batch[1].tolist() == [1, 2, 3, 4]

    data_config = build_passthrough_lm_data_config_from_store(store_dir=store_dir)
    validation_sets = data_config.validation_grug_sets(seq_len=4)
    first_example = asyncio.run(validation_sets["jpeg_tokens"].get_batch([0]))
    assert first_example[0].tokens.tolist() == [9, 10, 11, 12]


def test_file_backed_token_store_runs_one_training_step(tmp_path):
    store_dir = tmp_path / "coeff_store"
    seq_len = 8
    vocab_size = 32
    metadata = TokenStoreMetadata(
        dataset="dummy",
        dataset_config="v0",
        image_column="image",
        vocab_size=vocab_size,
        seq_len=seq_len,
        canonical_config={"resolution": 256},
        tokenizer_config={"zigzag_coefficients": 4},
        splits={
            "train": TokenStoreSplitInfo(
                num_examples=4,
                seq_len=seq_len,
                tokens_path="train_tokens.npy",
                manifest_path="train_manifest.jsonl",
            ),
            "validation": TokenStoreSplitInfo(
                num_examples=1,
                seq_len=seq_len,
                tokens_path="validation_tokens.npy",
                manifest_path="validation_manifest.jsonl",
            ),
        },
    )
    split_tokens = {
        "train": np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
                [3, 4, 5, 6, 7, 8, 9, 10],
                [4, 5, 6, 7, 8, 9, 10, 11],
            ],
            dtype=np.int32,
        ),
        "validation": np.asarray([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=np.int32),
    }
    split_records = {
        "train": [
            TokenSequenceRecord(
                example_id=f"train:{i}", split="train", num_tokens=seq_len, checksum=str(i), source_index=i
            )
            for i in range(4)
        ],
        "validation": [
            TokenSequenceRecord(
                example_id="validation:0", split="validation", num_tokens=seq_len, checksum="v", source_index=0
            )
        ],
    }
    write_token_store(store_dir, metadata=metadata, split_tokens=split_tokens, split_records=split_records)

    logger_name = f"test_jpeg_store_run_{uuid.uuid4().hex}"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        trainer_config = TrainerConfig(
            id="test-jpeg-store-run",
            num_train_steps=1,
            train_batch_size=1,
            tracker=JsonLoggerConfig(logger_name=logger_name),
            require_accelerator=False,
            use_explicit_mesh_axes=True,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            ray=RayConfig(auto_start_cluster=False),
            log_dir=tmp_path / "logs",
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints")),
        )
        run_config = JpegRunConfig(
            model=JpegLmConfig(
                vocab_size=vocab_size,
                hidden_dim=32,
                intermediate_dim=64,
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=seq_len,
            ),
            token_store_path=str(store_dir),
            resources=ResourceConfig.with_cpu(),
            optimizer=AdamConfig(learning_rate=1e-3),
            trainer=JpegTrainerConfig(trainer=trainer_config, log_every=1),
            eval=JpegEvalConfig(
                eval_batch_size=1,
                steps_per_eval=1,
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            ),
        )
        assert cloudpickle.dumps(run_config)
        run_jpeg_tokenizer(run_config)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    finish_records = [record for record in records if record.get("event") == "finish"]
    assert len(finish_records) == 1
    summary = finish_records[0]["summary"]
    assert "train/loss" in summary
    assert "eval/loss" in summary
