# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full (non-LoRA) DPO on Bloom SpecEval v2 preferences.

Policy + separate reference model (both Llama 8B, bf16 compute / fp32 master).
Targets v6e-32 (4 hosts x 8 chips, 32 GB HBM/chip) and v6e-64 (8 hosts x 8 chips)
as a fallback if v6e-32 OOMs.

Napkin math per chip at batch=64, seq=4096, FSDP across N chips:
  policy bf16         16 GB / N
  reference bf16      16 GB / N
  grads fp32          32 GB / N
  Adam m fp32         32 GB / N
  Adam v fp32         32 GB / N
  DCN replication     ~8 GB (multi-host)
  activations (gc)    ~4-5 GB/example/chip

v6e-32: static ~12 GB/chip, pd=2 → ~10 GB activations → ~22 GB total (fits in 32 GB).
v6e-64: static ~10 GB/chip, pd=1 → ~5 GB activations → ~15 GB total (comfortable).
"""

from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import SeparateReferenceConfig

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from levanter.data.text import PreferenceChatLmDatasetFormat
from marin.execution.executor import mirrored
from marin.processing.tokenize import lm_data_config

# Full bloom speceval v2 preference set (GPT-4.1 chosen vs Mixtral opposite rejected).
TRAIN_DATA = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/train/*.jsonl.gz",
    budget_gb=5,
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

# v6e is available in these regions for the marin cluster pool.
V6E_REGIONS = ["europe-west4", "us-east5", "us-east1"]

# Reference model: marin-8b-instruct (same as policy init).
REF_PATH = "marin-community/marin-8b-instruct"


def make_full_dpo_step(
    tpu_type: str,
    per_device: int,
    regions: list[str] | None = None,
    name_suffix: str = "",
    learning_rate: float = 5e-7,
    num_train_steps: int | None = None,
    num_epochs: float = 1.0,
    train_batch_size: int = 64,
    steps_per_eval: int | None = None,
    steps_per_hf_export: int = 500,
    steps_per_checkpoint: int | None = None,
):
    """Full (non-LoRA) DPO on Bloom SpecEval v2.

    per_device sets train_batch per chip. Kept explicit so we can trade memory for
    throughput when escalating v6e-32 → v6e-64.
    """
    config = SimpleDPOConfig(
        resources=ResourceConfig.with_tpu(tpu_type, regions=regions or V6E_REGIONS),
        per_device_parallelism=per_device,
        per_device_eval_parallelism=per_device,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        num_epochs=num_epochs,
        steps_per_eval=steps_per_eval,
        learning_rate=learning_rate,
        lr_schedule="cosine",
        warmup=0.1,
        wandb_project="dpo",
        tokenizer=marin_tokenizer,
        model_name_or_path="marin-community/marin-8b-instruct",
        reference=SeparateReferenceConfig(),
        reference_model_path=REF_PATH,
        reference_is_hf=True,
        train_seq_len=4096,
        max_seq_len=4096,
        beta=0.1,
        validation_split_fraction=None,
        reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
        steps_per_checkpoint=steps_per_checkpoint,
        steps_per_hf_export=steps_per_hf_export,
        hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
        seed=0,
    )

    tpu_short = tpu_type.replace("-", "")
    return default_dpo(
        name=f"dpo/full_dpo/bloom_speceval_v2_{tpu_short}_pd{per_device}_lr{learning_rate:g}{name_suffix}",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=config,
        tags=[
            "dpo",
            "full-dpo",
            "bloom",
            "speceval-v2",
            "llama3",
            "marin-instruct",
            tpu_type,
            f"pd{per_device}",
        ],
    )
