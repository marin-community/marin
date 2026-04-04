# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import math


EXPECTED_SUBSET_SIZES = {
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": 226313,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": 44800,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": 89300,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": 5690,
}


def test_qwen3_32b_tokenizer_vocab_size_is_cached():
    data_configs = importlib.import_module("marin.processing.tokenize.data_configs")

    assert data_configs._KNOWN_VOCAB_SIZES["Qwen/Qwen3-32B"] == 151936


def test_qwen3_32b_hf_matches_official_model_shape():
    qwen3 = importlib.import_module("experiments.qwen3")

    cfg = qwen3.qwen3_32b_hf

    assert qwen3.qwen3_32b_tokenizer == "Qwen/Qwen3-32B"
    assert cfg.reference_checkpoint == "Qwen/Qwen3-32B"
    assert cfg.max_seq_len == 40960
    assert cfg.hidden_dim == 5120
    assert cfg.intermediate_dim == 25600
    assert cfg.num_heads == 64
    assert cfg.num_kv_heads == 8
    assert cfg.num_layers == 64
    assert cfg.head_dim == 128
    assert cfg.tie_word_embeddings is False
    assert cfg.layer_norm_epsilon == 1e-6
    assert cfg.initializer_range == 0.02


def test_exp4307_matches_full_corpus_recipe():
    module = importlib.import_module("experiments.exp4307_sft_nemotron_terminal_corpus_qwen3_32b")

    assert module.SUBSET_SIZES == EXPECTED_SUBSET_SIZES
    assert module.DATASETS == {name: name for name in EXPECTED_SUBSET_SIZES}
    assert module.EFFECTIVE_EXAMPLES == sum(EXPECTED_SUBSET_SIZES.values())
    assert module.TARGET_EPOCHS == 2
    assert module.TRAIN_BATCH_SIZE == 128
    assert module.NUM_TRAIN_STEPS == math.ceil(2 * module.EFFECTIVE_EXAMPLES / module.TRAIN_BATCH_SIZE)
    assert module.RESOURCES.device.variant == "v5p-256"
    assert module.sft_config.initialize_from_hf == "Qwen/Qwen3-32B"
    assert module.sft_config.max_seq_len == 32768
    assert module.qwen3_32b_32k.max_seq_len == 32768
