"""
Resurrect the 32B with doctored opt state to clip the updates
"""

import dataclasses

from levanter.optim import AdamConfig
from levanter.optim.clip_update_norm import ClipUpdateNormConfig

from experiments.defaults import default_train
from experiments.tootsie.exp1295_32b import llama_32b_remat, llama_32b_tootsie, llama_32b_train_config, nemotron_mix
from marin.execution import executor_main
from marin.resources import TpuPodConfig

# We have doctored the opt state to include update history from
# gs://marin-us-central2/checkpoints/llama-32b-tootsie-2/checkpoints/step-77096 for clipping
warmstart_checkpoint = llama_32b_tootsie.cd("checkpoints/step-80000/").nonblocking()


llama_32b_warmstart_train = dataclasses.replace(
    llama_32b_train_config,
    initialize_from_checkpoint_path=warmstart_checkpoint,
    resources=TpuPodConfig("v4-2048", 1),
    reset_data_loader_on_init=False,
    # Specifically don't want to allow partial checkpoints here because we want to
    # ensure that the opt state is fully loaded and we don't have any issues with
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

marin_32b_necro = default_train(
    name="marin-32b-necro-2",
    tokenized=nemotron_mix,
    model_config=llama_32b_remat,
    train_config=llama_32b_warmstart_train,
    tags=["llama", "32b", "ema", "exp859", "exp1380", "tootsie", "necro"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/marin-32b-necro-2")


if __name__ == "__main__":
    executor_main(
        [marin_32b_necro],
        description="Attempt to revive the 32B with doctored opt state to clip the updates",
    )
