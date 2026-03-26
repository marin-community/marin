# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test: verify that hf_generation_eos_token_ids writes generation_config.json
in DPO checkpoint saves. Uses marin-8b-instruct on v5p-8 with 2 training steps.

After the run completes, check the HF checkpoint output for generation_config.json:
  gcloud storage cat gs://marin-us-east5/checkpoints/dpo/test_generation_config_smoke-<hash>/hf/step-2/generation_config.json

Expected content: {"eos_token_id": [128001, 128009], "bos_token_id": 128000}
"""

from experiments.defaults import default_dpo
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config

# Use pre-tokenized bloom speceval v2 data already on us-east5
TRAIN_CACHE = "gs://marin-us-east5/tokenized/bloom_speceval_v2_train_prefs_marin_tokenizer-12920b"
VAL_CACHE = "gs://marin-us-east5/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8"

tokenized_preferences = lm_data_config(
    training_set=TRAIN_CACHE,
    validation_sets={"bloom_speceval_v2_val": VAL_CACHE},
)

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=8,
    num_train_steps=2,
    learning_rate=5e-7,
    lr_schedule="cosine",
    warmup=0,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    reference_model_path="marin-community/marin-8b-instruct",
    reference_is_hf=True,
    train_seq_len=512,
    max_seq_len=512,
    beta=0.1,
    validation_split_fraction=None,
    steps_per_eval=2,
    steps_per_checkpoint=2,
    steps_per_hf_export=2,
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    seed=0,
)

training_step = default_dpo(
    name="dpo/test_generation_config_smoke",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=dpo_config,
    tags=["dpo", "smoke-test", "generation-config"],
)


if __name__ == "__main__":
    executor_main(steps=[training_step])
