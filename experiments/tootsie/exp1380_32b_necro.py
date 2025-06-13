"""
Resurect the 32B with doctored opt state to clip the updates
"""

import dataclasses

from levanter.optim import AdamConfig

from experiments.defaults import default_train
from experiments.tootsie.exp1295_32b import llama_32b_remat, llama_32b_tootsie, llama_32b_train_config, nemotron_mix
from marin.execution import executor_main
from marin.resources import TpuPodConfig

warmstart_checkpoint = llama_32b_tootsie.cd("checkpoints/step-80000/").nonblocking()


llama_32b_warmstart_train = dataclasses.replace(
    llama_32b_train_config,
    initialize_from_checkpoint_path=warmstart_checkpoint,
    resources=TpuPodConfig("v4-2048", 1),
    reset_data_loader_on_init=False,
    optimizer_config=AdamConfig(

    )

)

llama_32b_muon = default_train(
    name="marin-32b-muon-4",
    tokenized=nemotron_mix,
    model_config=llama_32b_remat,
    train_config=llama_32b_warmstart_train,
    tags=["llama", "32b", "ema", "exp859", "tootsie", "muon"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/marin-32b-muon-4")


if __name__ == "__main__":
    executor_main(
        [
            llama_32b_muon,
        ],
        description="Give muon a shot on 32B with Nemotron and Starcoderdata",
    )
