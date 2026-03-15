# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import cloudpickle
import json
import logging
import uuid
from dataclasses import replace
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
from experiments.jpeg_tokenizer.base.eval import (
    causal_loss_mask_from_lengths,
    coefficient_prefix_loss_mask,
    compute_reconstruction_metrics,
    compute_token_sequence_stats,
)
from experiments.jpeg_tokenizer.base.model import JpegLmConfig
from experiments.jpeg_tokenizer.base.train import JpegEvalConfig, JpegRunConfig, JpegTrainerConfig, run_jpeg_tokenizer
from experiments.jpeg_tokenizer.base.jpeg_codecs import (
    V0_AC_DENSE_CONFIG,
    ByteWindowTokenizerConfig,
    CoefficientTokenSource,
    CoefficientTokenizerConfig,
    HuffmanEventTokenizerConfig,
    WholeImageByteTokenizerConfig,
    ac_dense_vocab_size,
    V0_SYMBOL_CONFIG,
    byte_window_vocab_size,
    canonicalize_image,
    encode_dct_coeffs,
    encode_jpeg_ac_dense_absolute_dc_tokens,
    encode_jpeg_ac_dense_tokens,
    encode_jpeg_bytes,
    encode_jpeg_huffman_events,
    encode_jpeg_scan_bytes,
    encode_jpeg_symbols,
    extract_jpeg_scan_payload,
    huffman_event_vocab_size,
    pad_whole_image_byte_tokens,
    pad_whole_image_huffman_event_tokens,
    pad_whole_image_symbol_tokens,
    whole_image_byte_length,
    whole_image_byte_vocab_size,
    whole_image_huffman_event_length,
    whole_image_huffman_event_vocab_size,
    whole_image_symbol_length,
    whole_image_symbol_vocab_size,
    quantized_luma_blocks,
    reconstruct_luma_from_coeff_tokens,
    stable_byte_checksum,
    symbol_vocab_size,
    window_byte_tokens,
)
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig, RayConfig
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from scripts.jpeg_tokenizer.evaluate_coefficient_sweep import (
    _ablate_context_batch,
    _coefficient_prefix_context_keep_mask,
    _pad_batch,
)
from scripts.jpeg_tokenizer.evaluate_representation_head2head import parse_run_spec as parse_representation_run_spec


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


def test_byte_window_tokenizer_appends_eos_and_pads_tail():
    windows = window_byte_tokens(
        np.asarray([1, 2, 3, 4, 5], dtype=np.int32),
        config=ByteWindowTokenizerConfig(window_size=4, stride=4, pad_token_id=256, append_eos=True),
    )

    assert [window.tolist() for window in windows] == [[1, 2, 3, 4], [5, 256, 256, 256]]
    assert byte_window_vocab_size() == 257


def test_whole_image_byte_tokenizer_appends_eos_and_masks_pad_tail():
    config = WholeImageByteTokenizerConfig(eos_token_id=256, pad_token_id=257, append_eos=True)
    padded = pad_whole_image_byte_tokens(np.asarray([1, 2, 3], dtype=np.int32), seq_len=6, config=config)

    assert padded.tolist() == [1, 2, 3, 256, 257, 257]
    assert whole_image_byte_length(np.asarray([1, 2, 3], dtype=np.int32), config=config) == 4
    assert whole_image_byte_vocab_size(config) == 258


def test_whole_image_symbol_tokenizer_appends_eos_and_masks_pad_tail():
    padded = pad_whole_image_symbol_tokens(np.asarray([1, 2, 3], dtype=np.int32), seq_len=6)

    assert padded.tolist() == [1, 2, 3, 36833, 36834, 36834]
    assert whole_image_symbol_length(np.asarray([1, 2, 3], dtype=np.int32)) == 4
    assert whole_image_symbol_vocab_size() == 36835


def test_whole_image_huffman_event_tokenizer_appends_eos_and_masks_pad_tail():
    config = HuffmanEventTokenizerConfig()
    base_vocab = huffman_event_vocab_size(config)
    padded = pad_whole_image_huffman_event_tokens(np.asarray([1, 2, 3], dtype=np.int32), seq_len=6, config=config)

    assert padded.tolist() == [1, 2, 3, base_vocab, base_vocab + 1, base_vocab + 1]
    assert whole_image_huffman_event_length(np.asarray([1, 2, 3], dtype=np.int32)) == 4
    assert whole_image_huffman_event_vocab_size(config) == base_vocab + 2


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


def test_scan_payload_bytes_are_deterministic_and_shorter_than_full_jpeg():
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    canonical = canonicalize_image(image)

    full_bytes = encode_jpeg_bytes(canonical)
    scan_bytes = encode_jpeg_scan_bytes(canonical)

    assert extract_jpeg_scan_payload(canonical) == extract_jpeg_scan_payload(canonical.jpeg_bytes)
    assert scan_bytes.ndim == 1
    assert 0 < len(scan_bytes) < len(full_bytes)


def test_libjpeg_symbol_tokens_are_deterministic_and_within_vocab():
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    canonical = canonicalize_image(image)
    symbol_config = replace(V0_SYMBOL_CONFIG, source=CoefficientTokenSource.LIBJPEG)
    first_tokens = encode_jpeg_symbols(canonical, config=symbol_config)
    second_tokens = encode_jpeg_symbols(canonical, config=symbol_config)

    assert first_tokens.ndim == 1
    assert np.array_equal(first_tokens, second_tokens)
    assert first_tokens.min() >= 0
    assert first_tokens.max() < symbol_vocab_size(symbol_config)


def test_libjpeg_huffman_event_tokens_are_deterministic_and_within_vocab():
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    canonical = canonicalize_image(image)
    event_config = HuffmanEventTokenizerConfig(source=CoefficientTokenSource.LIBJPEG)
    first_tokens = encode_jpeg_huffman_events(canonical, config=event_config)
    second_tokens = encode_jpeg_huffman_events(canonical, config=event_config)

    assert first_tokens.ndim == 1
    assert np.array_equal(first_tokens, second_tokens)
    assert first_tokens.min() >= 0
    assert first_tokens.max() < huffman_event_vocab_size(event_config)


def test_libjpeg_ac_dense_tokens_are_deterministic_and_within_vocab():
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    canonical = canonicalize_image(image)
    ac_dense_config = replace(V0_AC_DENSE_CONFIG, source=CoefficientTokenSource.LIBJPEG)
    first_tokens = encode_jpeg_ac_dense_tokens(canonical, config=ac_dense_config)
    second_tokens = encode_jpeg_ac_dense_tokens(canonical, config=ac_dense_config)

    assert first_tokens.ndim == 1
    assert np.array_equal(first_tokens, second_tokens)
    assert first_tokens.min() >= 0
    assert first_tokens.max() < ac_dense_vocab_size(ac_dense_config)
    assert first_tokens.shape == (1024 * 64,)


def test_libjpeg_ac_dense_absolute_dc_tokens_are_deterministic_and_within_vocab():
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    canonical = canonicalize_image(image)
    ac_dense_config = replace(V0_AC_DENSE_CONFIG, source=CoefficientTokenSource.LIBJPEG)
    first_tokens = encode_jpeg_ac_dense_absolute_dc_tokens(canonical, config=ac_dense_config)
    second_tokens = encode_jpeg_ac_dense_absolute_dc_tokens(canonical, config=ac_dense_config)

    assert first_tokens.ndim == 1
    assert np.array_equal(first_tokens, second_tokens)
    assert first_tokens.min() >= 0
    assert first_tokens.max() < ac_dense_vocab_size(ac_dense_config)
    assert first_tokens.shape == (1024 * 64,)


def test_libjpeg_quantized_blocks_are_deterministic_and_match_expected_token_count():
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    canonical = canonicalize_image(image)
    libjpeg_config = CoefficientTokenizerConfig(zigzag_coefficients=4, source=CoefficientTokenSource.LIBJPEG)

    first_blocks = quantized_luma_blocks(canonical, config=libjpeg_config)
    second_blocks = quantized_luma_blocks(canonical, config=libjpeg_config)
    coeff_tokens = encode_dct_coeffs(canonical, config=libjpeg_config)

    assert first_blocks.shape == (1024, 8, 8)
    assert np.array_equal(first_blocks, second_blocks)
    assert coeff_tokens.shape == (1024 * 4,)
    assert coeff_tokens.min() >= 0


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


def test_coefficient_prefix_loss_mask_tracks_target_prefixes_across_blocks():
    mask = coefficient_prefix_loss_mask(8, tokens_per_block=4, prefix_tokens_per_block=2)
    assert mask.tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    full_mask = coefficient_prefix_loss_mask(8, tokens_per_block=4, prefix_tokens_per_block=4)
    assert full_mask.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_causal_loss_mask_from_lengths_masks_pad_tail_per_example():
    mask = causal_loss_mask_from_lengths([4, 2], seq_len=6)

    assert mask.tolist() == [
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_sequence_eval_tail_batch_padding_repeats_last_example():
    batch = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    padded, actual_batch_size = _pad_batch(batch, 4)

    assert actual_batch_size == 2
    assert padded.tolist() == [[1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6]]


def test_sequence_eval_context_prefix_only_ablation_masks_tail_coefficients():
    keep_mask = _coefficient_prefix_context_keep_mask(8, tokens_per_block=4, prefix_tokens_per_block=2)
    assert keep_mask.tolist() == [True, True, False, False, True, True, False, False]

    batch = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
    ablated = _ablate_context_batch(batch, keep_mask=keep_mask, replacement_token_id=2047)
    assert ablated.tolist() == [[1, 2, 2047, 2047, 5, 6, 2047, 2047]]


def test_representation_eval_run_spec_parses_optional_normalization():
    spec = parse_representation_run_spec(
        "name=symbols,checkpoint=gs://bucket/checkpoint,store=gs://bucket/store,sliding_window=4096,unit_name=block,unit_count=1024"
    )

    assert spec.name == "symbols"
    assert spec.sliding_window == 4096
    assert spec.normalization_unit_name == "block"
    assert spec.normalization_unit_count == 1024


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


def test_file_backed_token_store_uses_loss_mask_ignore_id_from_metadata(tmp_path):
    store_dir = tmp_path / "byte_store"
    metadata = TokenStoreMetadata(
        dataset="dummy",
        dataset_config="v0",
        image_column="image",
        vocab_size=258,
        seq_len=6,
        canonical_config={"resolution": 256},
        tokenizer_config={"eos_token_id": 256, "pad_token_id": 257, "loss_mask_ignore_id": 257},
        splits={
            "train": TokenStoreSplitInfo(
                num_examples=1,
                seq_len=6,
                tokens_path="train_tokens.npy",
                manifest_path="train_manifest.jsonl",
            ),
            "validation": TokenStoreSplitInfo(
                num_examples=1,
                seq_len=6,
                tokens_path="validation_tokens.npy",
                manifest_path="validation_manifest.jsonl",
            ),
        },
    )
    split_tokens = {
        "train": np.asarray([[1, 2, 3, 256, 257, 257]], dtype=np.uint16),
        "validation": np.asarray([[4, 5, 6, 256, 257, 257]], dtype=np.uint16),
    }
    split_records = {
        "train": [TokenSequenceRecord(example_id="train:0", split="train", num_tokens=4, checksum="a", source_index=0)],
        "validation": [
            TokenSequenceRecord(
                example_id="validation:0",
                split="validation",
                num_tokens=4,
                checksum="b",
                source_index=0,
            )
        ],
    }

    write_token_store(store_dir, metadata=metadata, split_tokens=split_tokens, split_records=split_records)

    data_config = build_passthrough_lm_data_config_from_store(store_dir=store_dir)
    validation_sets = data_config.validation_grug_sets(seq_len=6)
    first_example = asyncio.run(validation_sets["jpeg_tokens"].get_batch([0]))[0]

    assert first_example.tokens.tolist() == [4, 5, 6, 256, 257, 257]
    assert np.asarray(first_example.loss_weight).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


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
