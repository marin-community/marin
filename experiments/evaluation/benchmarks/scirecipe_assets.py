# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned offline weights and tokenizers for SciRecipe text metrics (1.52 GB)."""

from taskcompendium.importers.evalchemy.scorers.scirecipe import MINILM_PATH, ROBERTA_PATH

from experiments.post_training.task_curation.environment import Download

ROBERTA_SOURCE = "https://huggingface.co/FacebookAI/roberta-large/resolve/722cf37b1afa9454edce342e7895e588b6ff1d59"
MINILM_SOURCE = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

MODEL_FILES = {
    f"{ROBERTA_PATH}/config.json": Download(
        f"{ROBERTA_SOURCE}/config.json",
        "82ba49810e6441735e696033ac2512ee09da555507b2917ac3865b202d592cc3",
    ),
    f"{ROBERTA_PATH}/merges.txt": Download(
        f"{ROBERTA_SOURCE}/merges.txt",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
    f"{ROBERTA_PATH}/model.safetensors": Download(
        f"{ROBERTA_SOURCE}/model.safetensors",
        "047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38",
    ),
    f"{ROBERTA_PATH}/tokenizer.json": Download(
        f"{ROBERTA_SOURCE}/tokenizer.json",
        "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    ),
    f"{ROBERTA_PATH}/tokenizer_config.json": Download(
        f"{ROBERTA_SOURCE}/tokenizer_config.json",
        "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
    ),
    f"{ROBERTA_PATH}/vocab.json": Download(
        f"{ROBERTA_SOURCE}/vocab.json",
        "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
    ),
    f"{MINILM_PATH}/1_Pooling/config.json": Download(
        f"{MINILM_SOURCE}/1_Pooling/config.json",
        "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    ),
    f"{MINILM_PATH}/config.json": Download(
        f"{MINILM_SOURCE}/config.json",
        "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    ),
    f"{MINILM_PATH}/config_sentence_transformers.json": Download(
        f"{MINILM_SOURCE}/config_sentence_transformers.json",
        "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
    ),
    f"{MINILM_PATH}/model.safetensors": Download(
        f"{MINILM_SOURCE}/model.safetensors",
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    ),
    f"{MINILM_PATH}/modules.json": Download(
        f"{MINILM_SOURCE}/modules.json",
        "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    ),
    f"{MINILM_PATH}/sentence_bert_config.json": Download(
        f"{MINILM_SOURCE}/sentence_bert_config.json",
        "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    ),
    f"{MINILM_PATH}/special_tokens_map.json": Download(
        f"{MINILM_SOURCE}/special_tokens_map.json",
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    ),
    f"{MINILM_PATH}/tokenizer.json": Download(
        f"{MINILM_SOURCE}/tokenizer.json",
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    ),
    f"{MINILM_PATH}/tokenizer_config.json": Download(
        f"{MINILM_SOURCE}/tokenizer_config.json",
        "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    ),
    f"{MINILM_PATH}/vocab.txt": Download(
        f"{MINILM_SOURCE}/vocab.txt",
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
}
