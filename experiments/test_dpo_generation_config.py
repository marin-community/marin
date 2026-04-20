# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test: verify that hf_generation_eos_token_ids writes generation_config.json
in DPO checkpoint saves. Uses marin-8b-instruct on v5p-8 with 2 training steps.

After the run completes, check the HF checkpoint output for generation_config.json:
  gcloud storage cat gs://marin-us-east5/checkpoints/dpo/test_generation_config_smoke-<hash>/hf/step-2/generation_config.json

Expected content: {"eos_token_id": [128001, 128009], "bos_token_id": 128000}
"""

from levanter.data.text import PreferenceChatLmDatasetFormat

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import llama3_chat_stop_token_ids, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.preference_datasets import get_preference_dataset
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

preference_dataset = get_preference_dataset(DATASET_NAME, splits=["train_prefs", "test_prefs"])

tokenized_train = default_tokenize(
    name="ultrafeedback_binarized_train_prefs_marin_tokenizer",
    dataset=preference_dataset / "train_prefs/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_val = default_tokenize(
    name="ultrafeedback_binarized_test_prefs_marin_tokenizer",
    dataset=preference_dataset / "test_prefs/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={"ultrafeedback_test_prefs": tokenized_val},
)

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-8", ram="400g"),
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
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    steps_per_eval=2,
    steps_per_checkpoint=2,
    steps_per_hf_export=2,
    hf_generation_eos_token_ids=llama3_chat_stop_token_ids(marin_tokenizer),
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
    executor_main(
        steps=[
            preference_dataset,
            tokenized_train,
            tokenized_val,
            training_step,
        ]
    )
