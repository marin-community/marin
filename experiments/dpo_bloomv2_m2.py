# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# STAGING DRAFT — target path on dpo-lora-clean: experiments/dpo_bloomv2_m2.py

"""
Run DPO on the bloomv2_m2 preferences (bloomv2 base + 10-point pilot tension pairs).

The preference data combines:
  - original bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite preferences
    (bloomv2 base — unchanged)
  - 40 train + ~20 val tension-corner preference pairs from the 10-point pilot
    (see .agents/logbooks/claude_stress_testing.md Experiments 14-16)

Data lives on GCS at:
  train: gs://marin-us-central1/preference/bloomv2_m2/train/
  val:   gs://marin-us-central1/preference/bloomv2_m2/val_deduped/

Record schema matches bloomv2 (chat-format chosen/rejected + hash/prompt/statement_id/question_id).
"""

from levanter.data.text import PreferenceChatLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from marin.processing.tokenize import lm_data_config

GCS_PREFIX = "gs://marin-us-central1/preference/bloomv2_m2"

tokenized_train = default_tokenize(
    name="bloomv2_m2_train_prefs_marin_tokenizer",
    dataset=f"{GCS_PREFIX}/train/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_eval = default_tokenize(
    name="bloomv2_m2_val_deduped_prefs_marin_tokenizer",
    dataset=f"{GCS_PREFIX}/val_deduped/shard-00000.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={"bloomv2_m2_val": tokenized_eval},
)
