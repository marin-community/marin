"""Single‑doc sliding P(z) experiment for 1B on COMMA 10M.

This config trains a Llama ~1B model on the 10M seed head (one batch per dataset)
and runs P(z) on exactly one document using sliding windows (50‑token prefix,
50‑token suffix) with a 5‑token stride. Includes explicit trainer and data seeds
for reproducibility and a verbose flag to print selected doc details.
"""

from marin.execution.executor import executor_main

from experiments.defaults import default_train
from experiments.llama import llama_3_2_1b
from experiments.memorize.utils import REGION_TPU_DEFAULTS, make_comma_mixture_10m
from experiments.simple_train_config import SimpleTrainConfig
from levanter.eval_pz_innerloop import PzInnerLoopConfig
from marin.resources import TpuPodConfig


# -------------------------------
# Tunables / Reproducibility
# -------------------------------
REGION = "central1"
# Force a v5p-64 for this experiment regardless of regional defaults
TPU_TYPE = "v5p-64"

# Training seeds for reproducibility
TRAINER_SEED = 0
DATA_SEED = 0

# Batch and epochs
TRAIN_BATCH_SIZE = 128
EPOCHS_LIST = [1]  # adjust as needed; each "epoch" is 15 steps over the 10M head



def make_run(epochs: int):
    # Build the COMMA 10M mixture: one batch per dataset, feistel PRP, per-epoch reshuffle
    mixture, seed_batches = make_comma_mixture_10m()

    # P(z) configuration: one document across all datasets, sliding window, verbose
    pz_cfg = PzInnerLoopConfig(
        datasets=None,  # pool across all datasets; will pick exactly one eligible doc overall
        mode="sliding",
        num_documents=1,
        eval_batch_size=64,
        doc_tokens=None,
        min_doc_tokens=1024,
        chunk_size=100,
        prompt_tokens=50,
        cursor_inc_tokens=5,
        restrict_to_training_subset=True,
        initial_batch_size=TRAIN_BATCH_SIZE,
        doc_shuffle=True,  # deterministic head-of-eligible when not pooled; harmless when pooled
        doc_perm_type="feistel",
        doc_perm_seed=0,
        verbose=False,
        decode_preview=1,  # harmless hint; printing is controlled by verbose
    )

    # P(z) cadence: ~1% of total steps, min 1
    pz_steps = max(1, int((seed_batches * epochs) / 100 + 0.5))

    # Compose the training step
    step = default_train(
        name=f"memorize/comma_1b_10M_single_{epochs}epoch_{REGION}",
        tokenized=mixture,
        model_config=llama_3_2_1b,
        train_config=SimpleTrainConfig(
            resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
            train_batch_size=TRAIN_BATCH_SIZE,
            num_train_steps=seed_batches * epochs,
            learning_rate=0.003,
            lr_schedule="cosine",
            warmup=0.01,
            min_lr_ratio=0.0,
            z_loss_weight=0.0,
            steps_per_eval=1000,
            max_eval_batches=10,
            steps_per_task_eval=None,
            seed=TRAINER_SEED,
            data_seed=DATA_SEED,
        ),
        tags=["memorize", "comma", "1b", "10M", REGION, TPU_TYPE],
        eval_harness_tasks=(),
        pz_eval_config=pz_cfg,
        pz_eval_steps=pz_steps,
    )

    return step 


# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    steps = [make_run(e) for e in EPOCHS_LIST]
    executor_main(
        steps=steps,
        description="1B on ~10M COMMA seed set with single-doc sliding P(z) (us-central1).",
    )
