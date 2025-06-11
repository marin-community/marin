"""
Train 32B on Nemotron with Starcoderdata and Proofpile 2 using Muon
"""

import dataclasses

from levanter.optim import MuonConfig

from experiments.defaults import default_train
from experiments.llama import llama_32b
from experiments.tootsie.exp1295_32b import llama_32b_tootsie, llama_32b_train_config, nemotron_mix
from marin.execution import executor_main

warmstart_checkpoint = llama_32b_tootsie.cd("checkpoints/step-77096/").nonblocking()


muon_config = MuonConfig(
    # Yolo hypers for Muon from Kaiyue
    adam_lr=llama_32b_train_config.learning_rate,
    learning_rate=2e-3,
    weight_decay=0.1,
    momentum=0.98,
)

llama_32b_multislice_train = dataclasses.replace(
    llama_32b_train_config, initialize_from_checkpoint_path=warmstart_checkpoint, optimizer_config=muon_config
)

llama_32b_muon = default_train(
    name="marin-32b-muon-4",
    tokenized=nemotron_mix,
    model_config=llama_32b,
    train_config=llama_32b_multislice_train,
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
