"""
Switch over to Qwen3 architecture, warmstarting from the llama-32b-tootsie
"""

import dataclasses

import haliax
from levanter.optim import AdamConfig
from levanter.optim.clip_update_norm import ClipUpdateNormConfig

from experiments.defaults import default_train
from experiments.qwen3 import qwen3_32b
from experiments.tootsie.exp1295_32b import llama_32b_tootsie, llama_32b_train_config, nemotron_mix
from marin.execution import executor_main
from marin.resources import TpuPodConfig

# We have doctored the opt state to include update history from
# gs://marin-us-central2/checkpoints/llama-32b-tootsie-2/checkpoints/step-77096 for clipping
warmstart_checkpoint = llama_32b_tootsie.cd("checkpoints/step-80000/").nonblocking()

qwen3_32b_remat = dataclasses.replace(
    qwen3_32b, gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload")
)


qwen_32b_warmstart_train = dataclasses.replace(
    llama_32b_train_config,
    initialize_from_checkpoint_path=warmstart_checkpoint,
    resources=TpuPodConfig("v4-2048", 1),
    reset_data_loader_on_init=False,
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
        rewarmup=1000,
        decay=0.4,
        # using WSD-S to rewarmup given that we're adding new weights
        cycles=[80000, 1_000_000_000],
        # this was inadvertently off from about 74k to 80k
        clip_update_norm=ClipUpdateNormConfig(rolling_interval_length=128, sigma_factor=2.0),
    ),
)

marin_32b_qwen = default_train(
    name="marin-32b-qwen",
    tokenized=nemotron_mix,
    model_config=qwen3_32b_remat,
    train_config=qwen_32b_warmstart_train,
    tags=["qwen", "32b", "ema", "exp859", "exp1395", "tootsie"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/marin-32b-qwen")


if __name__ == "__main__":
    executor_main(
        [marin_32b_qwen], description="Warmstart 32B Qwen3 from Llama 32B Tootsie checkpoint and train on Nemotron etc"
    )
