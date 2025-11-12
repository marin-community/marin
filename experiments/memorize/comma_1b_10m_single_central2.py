"""Single‑doc sliding P(z) experiment for 1B on COMMA 10M (us‑central2, v4‑64).

This config trains a Llama ~1B model on the 10M seed head (one batch per dataset)
and runs P(z) on exactly one document using sliding windows (50‑token prefix,
50‑token suffix) with a 5‑token stride. Uses a v4‑64 in us‑central2.
"""

import dataclasses
from datetime import datetime, timedelta

from marin.execution.executor import executor_main

from experiments.defaults import default_train
from experiments.llama import llama_3_2_1b, llama3_tokenizer
from experiments.common_pile.tokenize_common_pile import common_pile_tokenized
from experiments.simple_train_config import SimpleTrainConfig
from levanter.eval_pz_innerloop import PzInnerLoopConfig
from marin.resources import TpuPodConfig
from marin.processing.tokenize.data_configs import lm_mixture_data_config


# -------------------------------
# Tunables / Reproducibility
# -------------------------------
REGION = "central2"
# Explicitly target a v4-64 in us-central2
TPU_TYPE = "v4-64"

# Training seeds for reproducibility
TRAINER_SEED = 0
DATA_SEED = 1

# Batch and epochs
TRAIN_BATCH_SIZE = 128
EPOCHS_LIST = [1000]  # adjust as needed; each "epoch" is 1 step for single-dataset/1-batch


def make_run(epochs: int):
    # Build mixture to match YAML semantics: include all COMMA components, but
    # set weights=0 and max_train_batches=0 for all except wikimedia which is 1.
    tokenized_all = common_pile_tokenized(tokenizer=llama3_tokenizer)
    components = dict(tokenized_all)
    # initialize all weights and caps to zero
    weights = {name: 0.0 for name in components}
    max_train_batches = {name: 0 for name in components}
    # identify wikimedia key and set it active
    wikimedia_keys = [name for name in components if name.endswith("/wikimedia")]
    if not wikimedia_keys:
        raise RuntimeError("Could not find 'wikimedia' component in Common Pile tokenized mapping.")
    wikimedia_key = wikimedia_keys[0]
    weights[wikimedia_key] = 1.0
    max_train_batches[wikimedia_key] = 1

    mixture = lm_mixture_data_config(
        components=components,
        weights=weights,
        max_train_batches=max_train_batches,
        shuffle=False,
    )
    # Enable Feistel PRP and per-epoch reshuffle (matches YAML defaults)
    mixture = dataclasses.replace(mixture, permutation_type="feistel", shuffle_per_epoch=False)
    seed_batches = 1

    # P(z) configuration: one document across all datasets, sliding window
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
        doc_shuffle=True,
        doc_perm_type="feistel",
        doc_perm_seed=0,
        verbose=False,
        decode_preview=1,
    )

    # P(z) cadence: ~1% of total steps, min 1
    pz_steps = 150

    # Compose the training step
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step = default_train(
        name=f"memorize/comma_1b_10M_single_{epochs}epoch_{REGION}_{timestamp}",
        tokenized=mixture,
        model_config=llama_3_2_1b,
        train_config=SimpleTrainConfig(
            resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
            train_batch_size=TRAIN_BATCH_SIZE,
            num_train_steps=15000,
            learning_rate=0.003,
            lr_schedule="cosine",
            warmup=0.01,
            min_lr_ratio=0.0,
            weight_decay=0.0,
            z_loss_weight=0.0,
            steps_per_eval=1000,
            max_eval_batches=10,
            steps_per_task_eval=None,
            seed=TRAINER_SEED,
            data_seed=DATA_SEED,
            # Save permanent checkpoints every 10000 steps (match YAML)
            steps_per_export=10000,
        ),
        tags=["memorize", "comma", "1b", "10M", REGION, TPU_TYPE],
        eval_harness_tasks=(),
        pz_eval_config=pz_cfg,
        pz_eval_steps=pz_steps,
    )

    # Match YAML: enable time-based checkpointing (24h) and TP axes ["heads", "mlp"],
    # keep permanent step checkpoints every 10000 steps.
    tc = step.config.train_config
    trainer = dataclasses.replace(
        tc.trainer,
        checkpointer=dataclasses.replace(tc.trainer.checkpointer, save_interval=timedelta(hours=24)),
        tensor_parallel_axes=["heads", "mlp"],
    )
    new_train_config = dataclasses.replace(tc, trainer=trainer)
    new_pod_config = dataclasses.replace(step.config, train_config=new_train_config)
    step = dataclasses.replace(step, config=new_pod_config)

    return step


# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    steps = [make_run(e) for e in EPOCHS_LIST]
    executor_main(
        steps=steps,
        description="1B on ~10M COMMA seed set with single-doc sliding P(z) (us-central2, v4-64).",
    )
