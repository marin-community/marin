# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-statement DPO experiments: common data setup and config builder.

Trains LoRA DPO on a single Bloom statement (support_mental_health) with
dual validation: per-statement val + full 46-statement val for regression
tracking. See .agents/logbooks/continual-alignment-per-statement-dpo.md
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
from marin.execution.executor import executor_main, mirrored
from marin.processing.tokenize import lm_data_config

# --- Data paths (mirrored for region-agnostic access) ---

STMT_TRAIN = mirrored(
    "preference/bloom_v2_singleton/support_mental_health/train/shard-00000.jsonl.gz",
    budget_gb=1,
)
STMT_VAL = mirrored(
    "preference/bloom_v2_singleton/support_mental_health/val/shard-00000.jsonl.gz",
    budget_gb=1,
)
FULL_VAL = mirrored(
    "preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz",
    budget_gb=1,
)

# --- Tokenize steps ---

tokenized_train = default_tokenize(
    name="bloom_v2_stmt_support_mental_health_train_marin_tokenizer",
    dataset=STMT_TRAIN,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_stmt_val = default_tokenize(
    name="bloom_v2_stmt_support_mental_health_val_marin_tokenizer",
    dataset=STMT_VAL,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_full_val = default_tokenize(
    name="bloom_speceval_v2_val_deduped_prefs_marin_tokenizer",
    dataset=FULL_VAL,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

# --- Dual validation: per-statement + full 46-statement ---

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={
        "stmt_val": tokenized_stmt_val,
        "full_val": tokenized_full_val,
    },
)

# --- TPU-specific configs ---

REGIONS: dict[str, list[str]] = {
    "v5p-8": ["us-central1", "us-east5"],
    "v6e-8": ["europe-west4", "us-east5", "us-east1"],
}
PER_DEVICE: dict[str, int] = {"v5p-8": -1, "v6e-8": 4}
PER_DEVICE_EVAL: dict[str, int] = {"v5p-8": 16, "v6e-8": 4}


def make_exp1a_step(lr: float, steps: int, tpu: str = "v6e-8"):
    regions = REGIONS[tpu]
    ram = "400g" if tpu.startswith("v5p") else None
    resources = ResourceConfig.with_tpu(tpu, ram=ram, regions=regions)

    config = SimpleDPOConfig(
        resources=resources,
        per_device_parallelism=PER_DEVICE[tpu],
        per_device_eval_parallelism=PER_DEVICE_EVAL[tpu],
        train_batch_size=64,
        num_train_steps=steps,
        steps_per_eval=max(steps // 3, 1),
        learning_rate=lr,
        lr_schedule="cosine",
        warmup=0.1,
        wandb_project="dpo",
        tokenizer=marin_tokenizer,
        model_name_or_path="marin-community/marin-8b-instruct",
        adapter=LoraAdaptationConfig(
            r=16,
            alpha=32,
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
        steps_per_checkpoint=steps,
        steps_per_hf_export=steps,
        hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
        seed=0,
    )

    lr_str = f"{lr:g}".replace("-", "m").replace(".", "p")
    tpu_short = tpu.replace("-", "")
    slug = f"stmt_dpo/exp1a/smh_lr{lr_str}_s{steps}_{tpu_short}"

    return default_dpo(
        name=f"dpo/{slug}",
        tokenized=tokenized_preferences,
        model_config=llama_8b,
        dpo_config=config,
        tags=[
            "dpo",
            "lora-dpo",
            "bloom",
            "per-stmt",
            "support-mental-health",
            "exp1a",
            f"lr{lr:g}",
            f"s{steps}",
            tpu,
        ],
    )


def run_exp1a(lr: float, steps: int, tpu: str = "v6e-8"):
    step = make_exp1a_step(lr, steps, tpu)
    executor_main(
        steps=[
            tokenized_train,
            tokenized_stmt_val,
            tokenized_full_val,
            step,
        ]
    )


# === Phase 2: Exps 1b, 2a, 2b ===

# --- 3-statement data paths ---

THREE_STMT_TRAIN = mirrored(
    "preference/bloom_v2_singleton/support_mental_health+do_not_encourage_self_harm+avoid_overstepping/train/*.jsonl.gz",
    budget_gb=1,
)
THREE_STMT_VAL = mirrored(
    "preference/bloom_v2_singleton/support_mental_health+do_not_encourage_self_harm+avoid_overstepping/val/shard-00000.jsonl.gz",
    budget_gb=1,
)

tokenized_3stmt_train = default_tokenize(
    name="bloom_v2_3stmt_smh_dnesh_ao_train_marin_tokenizer",
    dataset=THREE_STMT_TRAIN,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_3stmt_val = default_tokenize(
    name="bloom_v2_3stmt_smh_dnesh_ao_val_marin_tokenizer",
    dataset=THREE_STMT_VAL,
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

# 1-statement dual val (reuse existing)
tokenized_1stmt_prefs = tokenized_preferences

# 3-statement dual val
tokenized_3stmt_prefs = lm_data_config(
    training_set=tokenized_3stmt_train,
    validation_sets={
        "stmt_val": tokenized_3stmt_val,
        "full_val": tokenized_full_val,
    },
)

# DPO base checkpoint for continual alignment (exps 2a, 2b)
# In us-central2 (copied from us-central1). Sub-jobs need MARIN_MIRROR_BUDGET_GB=50
# to allow the ~32 GB cross-region model read when the TPU is outside us-central2.
DPO_BASE_CHECKPOINT = "gs://marin-us-central2/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699"


def _make_dpo_step(
    exp: str,
    lr: float,
    steps: int,
    tpu: str,
    tokenized_prefs,
    model_name_or_path: str = "marin-community/marin-8b-instruct",
    initialize_from_hf: bool | None = None,
    initialize_from_checkpoint_path: str | None = None,
    stmt_tag: str = "support-mental-health",
):
    regions = REGIONS[tpu]
    ram = "400g" if tpu.startswith("v5p") else None
    resources = ResourceConfig.with_tpu(tpu, ram=ram, regions=regions)

    config = SimpleDPOConfig(
        resources=resources,
        per_device_parallelism=PER_DEVICE[tpu],
        per_device_eval_parallelism=PER_DEVICE_EVAL[tpu],
        train_batch_size=64,
        num_train_steps=steps,
        steps_per_eval=max(steps // 3, 1),
        learning_rate=lr,
        lr_schedule="cosine",
        warmup=0.1,
        wandb_project="dpo",
        tokenizer=marin_tokenizer,
        model_name_or_path=model_name_or_path,
        initialize_from_hf=initialize_from_hf,
        initialize_from_checkpoint_path=initialize_from_checkpoint_path,
        adapter=LoraAdaptationConfig(
            r=16,
            alpha=32,
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
        steps_per_checkpoint=steps,
        steps_per_hf_export=steps,
        hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
        seed=0,
    )

    lr_str = f"{lr:g}".replace("-", "m").replace(".", "p")
    tpu_short = tpu.replace("-", "")
    slug = f"stmt_dpo/{exp}/{stmt_tag}_lr{lr_str}_s{steps}_{tpu_short}"

    return default_dpo(
        name=f"dpo/{slug}",
        tokenized=tokenized_prefs,
        model_config=llama_8b,
        dpo_config=config,
        tags=[
            "dpo",
            "lora-dpo",
            "bloom",
            "per-stmt",
            stmt_tag,
            exp,
            f"lr{lr:g}",
            f"s{steps}",
            tpu,
        ],
    )


# Exp 1b: SFT base, 3 statements
def run_exp1b(lr: float, steps: int, tpu: str = "v6e-8"):
    step = _make_dpo_step("exp1b", lr, steps, tpu, tokenized_3stmt_prefs, stmt_tag="3stmt")
    executor_main(steps=[tokenized_3stmt_train, tokenized_3stmt_val, tokenized_full_val, step])


# Exp 2a: DPO base, 1 statement (continual alignment)
def run_exp2a(lr: float, steps: int, tpu: str = "v6e-8"):
    step = _make_dpo_step(
        "exp2a",
        lr,
        steps,
        tpu,
        tokenized_1stmt_prefs,
        model_name_or_path=DPO_BASE_CHECKPOINT,
        initialize_from_hf=True,
    )
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, step])


# Exp 2b: DPO base, 3 statements (continual alignment)
def run_exp2b(lr: float, steps: int, tpu: str = "v6e-8"):
    step = _make_dpo_step(
        "exp2b",
        lr,
        steps,
        tpu,
        tokenized_3stmt_prefs,
        model_name_or_path=DPO_BASE_CHECKPOINT,
        initialize_from_hf=True,
        stmt_tag="3stmt",
    )
    executor_main(steps=[tokenized_3stmt_train, tokenized_3stmt_val, tokenized_full_val, step])
