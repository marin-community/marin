"""
Train 32B on on Nemotron with Starcoderdata and Proofpile 2
"""

import dataclasses

import haliax
from levanter.callbacks.watch import WatchConfig
from levanter.optim import AdamConfig
from levanter.optim.clip_update_norm import ClipUpdateNormConfig
from levanter.schedule import ScheduleStep

from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_train
from experiments.llama import llama_32b
from experiments.nemotron_cc.tokenize_nemotron import NEMOTRON_WEIGHTS, tokenize_nemotron_steps
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig

## 32b experiments

# on the v4-2048, with the 8192 batch size, we need to offload the carries
llama_32b_remat = dataclasses.replace(
    llama_32b, gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload")
)

llama_32b_train_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-2048", slice_count=1),
    # decreasing so we don't have padding at slice count 3
    # but we moved to v4 once we lost the v5 compute so we moved back to 8192 again
    train_batch_size=[
        ScheduleStep(start=0, value=8192),
        ScheduleStep(start=18500, value=7680),
        ScheduleStep(start=21010, value=8192),
    ],
    num_train_steps=1_000_000,
    weight_decay=0.05,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
    z_loss_weight=1e-4,
    # width is a little smaller than the 24B and we're using a much larger batch size
    # 4.2e-4 * sqrt(8192/3072) ≈ 7e-4
    learning_rate=7e-4,  ## ignored and overridden by the optimizer config
    watch=WatchConfig(watch_targets=["grads", "params", "updates", "opt_state"], interval=1),
    skip_bad_steps=True,
    max_grad_norm=0.2,  # we're almost always < .2 except during spikes
    allow_partial_checkpoint=True,
    optimizer_config=AdamConfig(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=0.2,  # we're almost always < .2 except during spikes
        # width is a little smaller than the 24B and we're using a much larger batch size
        # 4.2e-4 * sqrt(8192/3072) ≈ 7e-4
        learning_rate=7e-4,
        weight_decay=0.05,
        skip_bad_steps=True,
        # update_rms_clipping=1.0,  # added at 67522, removed at 72233
        lr_schedule="linear",
        warmup=0.01,
        decay=0.4,
        cycle_length=None,
        # this was inadvertently off from about 74k to 80k
        clip_update_norm=ClipUpdateNormConfig(rolling_interval_length=128, sigma_factor=2.0),
    ),
)

nemotron_steps = tokenize_nemotron_steps()
proofpile_2 = dclm_components_llama3["proofpile_2"]
starcoderdata = dclm_components_llama3["starcoderdata"]
nemotron_mix = lm_mixture_data_config(
    components={**nemotron_steps, "starcoderdata": starcoderdata, "proofpile_2": proofpile_2},
    weights={
        **NEMOTRON_WEIGHTS,
        "starcoderdata": 0.25,
        "proofpile_2": 0.055,
    },
)

llama_32b_tootsie = default_train(
    name="llama-32b-tootsie-2",
    tokenized=nemotron_mix,
    model_config=llama_32b_remat,
    train_config=llama_32b_train_config,
    tags=["llama", "32b", "ema", "exp859", "tootsie"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/llama-32b-tootsie-2")


if __name__ == "__main__":
    executor_main(
        [
            llama_32b_tootsie,
        ],
        description="Train 32B on Nemotron with Starcoderdata and Proofpile 2",
    )
