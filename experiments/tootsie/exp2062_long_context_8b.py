# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Long-context retrofit of the 8B Tootsie line.

Phases (all reuse the same data mix: 90% starling cooldown mix, 5% long-pdf,
5% reasoning/chat):
1) 4k seq len for ~100B tokens, warm-start from sensible starling.
2) Extend to 32k (RoPE theta 1.5M) for ~50B tokens.
3) Extend to 64k (RoPE theta 5M) for ~50B tokens.

Train configs set `train_seq_len` to the active context length so the executor
UI shows the intended target (field is new; "roll with it").
"""

import dataclasses

from experiments.marin_models import marin_tokenizer
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS
from fray.v2 import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_8b
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.posttrain.long_context_datasets import (
    finepdfs_edu_by_language,
    finepdfs_edu_token_counts,
    finepdfs_validation_by_language,
    longmino_bucket_token_counts,
    longmino_by_bucket,
)
from experiments.tootsie.exp600_tootsie import (
    tootsie_8b_sensible_starling,
    starling_cooldown_weights,
    starling_components,
    PHASE_3_START,
    PHASE_4_START,
    PHASE_4_END,
    PHASE_4_REWARMUP_DURATION,
    phase_4_warmup_weights,
    cooldown_mixture_weights_v1,
    phase_4_steady_state_weights,
    STARLING_END,
    cooldown_train_config,
)
from levanter.optim import AdamConfig
from marin.processing.tokenize.data_configs import (
    lm_mixture_data_config,
    interpolate_mixture_weights,
    lm_varying_mixture_data_config,
)
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

# ---------------------------
# Phases
# ---------------------------
# Giraffe phase 1 (4k) is a partial recooldown of starling with shifted data distribution toward long-context data.
# Phases 2 (32k) and 3 (64k) are context length extensions with the same data ditribution that show up as distinct runs
# We do this because changing context length means we have to reset data loader (b/c we permuted over sequences not docs)
GIRAFFE_4K_START = STARLING_END
GIRAFFE_4K_STEPS = 6000  # 6000 * 4096 * 4096 ≈ 100B tokens

GIRAFFE_4K_END = GIRAFFE_4K_START + GIRAFFE_4K_STEPS

GIRAFFE_16K_STEPS = 3000  # 3000 * 512 * 32768 ≈ 50B tokens
GIRAFFE_32K_STEPS = 3000  # 3000 * 256 * 65536 ≈ 50B tokens

# --------------------------
# Data: 90% base starling mix, 5% long-context PDFs, 5% reasoning/chat
# --------------------------
LONGMINO_BUCKETS = ["8k-16k", "16k-32k", "32k-64k"]

long_context_tokenized = {
    bucket: default_tokenize(
        name=f"longmino_{bucket}_llama3",
        dataset=longmino_by_bucket[bucket],
        tokenizer=marin_tokenizer,
    )
    for bucket in LONGMINO_BUCKETS
}

pdf_total = sum(longmino_bucket_token_counts[b] for b in LONGMINO_BUCKETS)
pdf_weights = {b: longmino_bucket_token_counts[b] / pdf_total for b in LONGMINO_BUCKETS}

long_context_mixture = lm_mixture_data_config(
    long_context_tokenized,
    pdf_weights,
)

finepdfs_edu_tokenized = {
    "finepdfs_edu_eng": default_tokenize(
        name="finepdfs_edu_eng_Latn_llama3",
        dataset=finepdfs_edu_by_language["eng_Latn"],
        tokenizer=marin_tokenizer,
    )
}

finepdfs_edu_mixture = lm_mixture_data_config(
    finepdfs_edu_tokenized,
    {"finepdfs_edu_eng": finepdfs_edu_token_counts["eng_Latn"]},
)

# Token-weighted blend of longmino buckets and finepdfs-edu inside the PDF slice
pdf_longmino_tokens = sum(longmino_bucket_token_counts[b] for b in LONGMINO_BUCKETS)
pdf_finepdf_tokens = finepdfs_edu_token_counts["eng_Latn"]
pdf_total_tokens = pdf_longmino_tokens + pdf_finepdf_tokens
pdf_mix_weights = [pdf_longmino_tokens / pdf_total_tokens, pdf_finepdf_tokens / pdf_total_tokens]

pdf_weights_combined = interpolate_mixture_weights(
    [pdf_weights, {"finepdfs_edu_eng": finepdfs_edu_token_counts["eng_Latn"]}],
    pdf_mix_weights,
)

pdf_combined_mixture = lm_mixture_data_config(
    {**long_context_tokenized, **finepdfs_edu_tokenized},
    pdf_weights_combined,
)

REASONING_DATASETS = [
    "nvidia/Nemotron-Post-Training-Dataset-v1/chat",
    "nvidia/Nemotron-Post-Training-Dataset-v1/code",
    "nvidia/Nemotron-Post-Training-Dataset-v1/math",
    "nvidia/Nemotron-Post-Training-Dataset-v1/stem",
    "nvidia/Nemotron-Post-Training-Dataset-v1/tool_calling",
    "HuggingFaceTB/smoltalk2/OpenThoughts3_1.2M_think",
]

reasoning_tokenized = {
    ds.split("/")[-1]: default_tokenize(
        name=ds.replace("/", "_") + "_llama3",
        dataset=get_instruction_dataset(ds) / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )
    for ds in REASONING_DATASETS
}

# Token-proportional reasoning weights with stem capped at ~25% and chat at ~5%
# Using size estimates (bytes) and then re-normalized after caps:
# stem 0.25, chat 0.00675, code 0.29124, math 0.25230, tool_calling 0.02038, ot3 0.17933
reasoning_weights = {
    "chat": 0.00675,
    "code": 0.29124,
    "math": 0.25230,
    "stem": 0.25,
    "tool_calling": 0.02038,
    "OpenThoughts3_1.2M_think": 0.17933,
}

reasoning_mixture = lm_mixture_data_config(reasoning_tokenized, reasoning_weights)

# Validation: finepdfs (original) eng_Latn test split
finepdfs_validation_tokenized = {
    "finepdfs/eng": default_tokenize(
        name="finepdfs_eng_Latn_val",
        dataset=finepdfs_validation_by_language["eng_Latn"],
        tokenizer=marin_tokenizer,
        is_validation=True,
    )
}


# Blend mixtures: 90% base, 5% pdf (longmino + finepdfs), 5% reasoning.
long_context_combined_weights = interpolate_mixture_weights(
    [starling_cooldown_weights, pdf_weights_combined, reasoning_weights],
    [0.90, 0.05, 0.05],
)


giraffe_components = {
    **starling_components,
    **long_context_tokenized,
    **finepdfs_edu_tokenized,
    **reasoning_tokenized,
    **finepdfs_validation_tokenized,
}

# for the first phase of "long context extension", we're not really extending context at all, but
# shifting the data distribution toward long-context data. To do that, we just do more continued
# training of the starling cooldown mix but use the long-context data weights.

giraffe_4K_mixture = lm_varying_mixture_data_config(
    components=giraffe_components,
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
        (PHASE_4_START, phase_4_warmup_weights),
        (PHASE_4_START + PHASE_4_REWARMUP_DURATION, phase_4_steady_state_weights),
        (PHASE_4_END, starling_cooldown_weights),
        (GIRAFFE_4K_START, long_context_combined_weights),
    ],
    mixture_block_size=1024,
)


giraffe_long_mixture = lm_mixture_data_config(
    giraffe_components,
    long_context_combined_weights,
)

# ensure we use the marin_tokenizer in the data config
giraffe_4K_mixture = dataclasses.replace(giraffe_4K_mixture, tokenizer=marin_tokenizer)
giraffe_long_mixture = dataclasses.replace(giraffe_long_mixture, tokenizer=marin_tokenizer)


# --------------------------
# Model configs per phase
# --------------------------
llama_8b_4k = dataclasses.replace(llama_8b, cross_entropy_block_size=32000)

llama_8b_32k = dataclasses.replace(
    llama_8b_4k, max_seq_len=32_768, rope=dataclasses.replace(llama_8b.rope, theta=1_500_000)  # type: ignore[arg-type]
)

llama_8b_64k = dataclasses.replace(
    llama_8b_4k,
    max_seq_len=65_536,
    rope=Llama3RotaryEmbeddingsConfig(theta=5_000_000),
    cross_entropy_block_size=16384,
)


# --------------------------
# Training configs
# --------------------------
def _train_config(
    *, num_steps: int, batch_size: int, train_seq_len: int, initialize_from, seed: int
) -> SimpleTrainConfig:
    return SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type="v4-512", slice_count=1),
        train_batch_size=batch_size,
        num_train_steps=num_steps,
        train_seq_len=train_seq_len,
        learning_rate=1e-4,
        min_lr_ratio=0.0,
        lr_schedule="linear",
        decay=num_steps,
        warmup=200,
        steps_per_eval=500,
        steps_per_task_eval=None,
        steps_per_export=1000,
        per_device_eval_parallelism=8,
        z_loss_weight=1e-4,
        initialize_from_hf=initialize_from,
        allow_partial_checkpoint=True,
        # unfortunately we don't have a great way of saving data loader state when we change the seq len
        # so we reset and reseed at each phase transition
        reset_data_loader_on_init=True,
        data_seed=seed,
    )


STARLING_WARMSTART_STEP = "1399923"
# needed to move step-1399923-patched/opt_state/inner_state/1 to opt_state/inner_state/0 for some reason
starling_checkpoint = tootsie_8b_sensible_starling.cd(
    f"checkpoints/step-{STARLING_WARMSTART_STEP}-patched"
).nonblocking()

# Phase 1: 4k -> ~100B tokens
PHASE1_STEPS = GIRAFFE_4K_STEPS

giraffe_4k_config = dataclasses.replace(
    cooldown_train_config,
    resources=ResourceConfig.with_tpu("v4-512", slice_count=1),
    train_seq_len=4096,
    num_train_steps=GIRAFFE_4K_END + 3,  # +3 to avoid fencepost issues
    initialize_from_checkpoint_path=starling_checkpoint,
    steps_per_export=1000,
    steps_per_hf_export=1000,
    optimizer_config=AdamConfig(
        lr_schedule="linear",
        learning_rate=1e-4,
        max_grad_norm=cooldown_train_config.max_grad_norm,
        # similar to phoenix, we abuse WSD-S api to rewarmup
        cycles=[STARLING_END, GIRAFFE_4K_END],
        decay=1.0,
        # decay=1.0,
        rewarmup=100,
        min_lr_ratio=0.1,
        adamc_weight_decay=True,
    ),
    allow_partial_checkpoint=False,
)


# the original had a botched lr schedule. classic me.
tootsie_8b_giraffe_phase1 = default_train(
    name="tootsie-8b-giraffe-4k-v4",
    tokenized=giraffe_4K_mixture,
    model_config=llama_8b_4k,
    train_config=giraffe_4k_config,
    tags=["llama", "8b", "giraffe", "phase1", "exp2062"],
    eval_harness_tasks=[],
)

# Phase 2: 32k with RoPE theta 1.5M for ~50B tokens
# 3000 * 512 * 32768 ≈ 50B tokens
phase1_final_checkpoint = tootsie_8b_giraffe_phase1.cd(f"hf/step-{GIRAFFE_4K_END}")
phase2_train_config = _train_config(
    num_steps=GIRAFFE_16K_STEPS, batch_size=512, train_seq_len=32_768, initialize_from=phase1_final_checkpoint, seed=2
)

tootsie_8b_giraffe_phase2 = default_train(
    name="tootsie-8b-giraffe-32k",
    tokenized=giraffe_long_mixture,
    model_config=llama_8b_32k,
    train_config=phase2_train_config,
    tags=["llama", "8b", "giraffe", "phase2", "exp2062"],
    eval_harness_tasks=[],
    # hash changed somehow
).with_output_path("checkpoints/tootsie-8b-giraffe-32k-293fef")

# Phase 3: 64k with RoPE theta 5M for ~50B tokens
PHASE3_STEPS = GIRAFFE_32K_STEPS  # 3000 * 256 * 65536 ≈ 50B tokens
# - 1 because of fencepost issues
phase2_final_checkpoint = tootsie_8b_giraffe_phase2.cd(f"hf/step-{GIRAFFE_16K_STEPS - 1}")
phase3_train_config = _train_config(
    num_steps=PHASE3_STEPS,
    batch_size=256,
    train_seq_len=65_536,
    initialize_from=phase2_final_checkpoint,
    seed=3,
)

tootsie_8b_giraffe_phase3 = default_train(
    name="tootsie-8b-giraffe-phase3-64k",
    tokenized=giraffe_long_mixture,
    model_config=llama_8b_64k,
    train_config=phase3_train_config,
    tags=["llama", "8b", "giraffe", "phase3", "exp2062"],
    eval_harness_tasks=[],
)

phase3_final_checkpoint = tootsie_8b_giraffe_phase3.cd(f"hf/step-{PHASE3_STEPS - 1}")


if __name__ == "__main__":
    executor_main(
        [
            *long_context_tokenized.values(),
            *finepdfs_edu_tokenized.values(),
            *reasoning_tokenized.values(),
            tootsie_8b_giraffe_phase1,
            tootsie_8b_giraffe_phase2,
            tootsie_8b_giraffe_phase3,
        ]
    )
