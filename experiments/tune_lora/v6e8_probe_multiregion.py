# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe: LoRA DPO on v6e-8, region-agnostic via mirror://. Short (10 steps).

Uses mirrored() on the source preference data so the executor can resolve
tokenization steps in any region. The mirror filesystem copies data on-demand.
"""

from levanter.adaptation import LoraAdaptationConfig
from levanter.data.text import PreferenceChatLmDatasetFormat
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import mirrored
from marin.processing.tokenize import lm_data_config

# Source data paths wrapped with mirrored() so the executor is region-agnostic.
TRAIN_DATA = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/train/*.jsonl.gz",
    budget_gb=1,
)
VAL_DATA = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz",
    budget_gb=1,
)

tokenized_train = default_tokenize(
    name="bloom_speceval_v2_train_prefs_marin_tokenizer",
    dataset=TRAIN_DATA,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_eval = default_tokenize(
    name="bloom_speceval_v2_val_deduped_prefs_marin_tokenizer",
    dataset=VAL_DATA,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={"bloom_speceval_v2_val": tokenized_eval},
)


def make_v6e8_probe(regions: list[str] | None = None, name_suffix: str = "", per_device: int = 4):
    config = SimpleDPOConfig(
        resources=ResourceConfig.with_tpu("v6e-8", regions=regions) if regions else ResourceConfig.with_tpu("v6e-8"),
        per_device_parallelism=per_device,
        per_device_eval_parallelism=per_device,
        train_batch_size=64,
        num_train_steps=10,
        steps_per_eval=5,
        learning_rate=5e-6,
        lr_schedule="cosine",
        warmup=0.1,
        wandb_project="dpo",
        tokenizer=marin_tokenizer,
        model_name_or_path="marin-community/marin-8b-instruct",
        adapter=LoraAdaptationConfig(
            r=64,
            alpha=64,
            dropout=0.0,
            zero_init_b=True,
            target_modules=None,
        ),
        reference=AdapterBaseReferenceConfig(),
        train_seq_len=4096,
        max_seq_len=4096,
        beta=0.1,
        validation_split_fraction=None,
        reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
        steps_per_checkpoint=200,
        steps_per_hf_export=200,
        hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
        seed=0,
    )
    return default_dpo(
        name=f"dpo/tune_lora/v6e8_probe{name_suffix}",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=config,
        tags=["dpo", "lora-dpo", "bloom", "speceval-v2", "llama3", "marin-instruct", "v6e-probe"],
    )
