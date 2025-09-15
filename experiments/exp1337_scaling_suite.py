"""
Creates a suite of training runs of different model sizes and number of tokens
trained on the same data ("modernized Pythia").
"""

from dataclasses import dataclass, replace
from math import sqrt

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import default_train
from experiments.llama import (
    compute_num_parameters,
    llama3_tokenizer_vocab_size,
    llama_1_4b,
    llama_1_4b_train_config,
    llama_8b,
    llama_8b_train_config,
    llama_32b,
    llama_70b,
    llama_150m,
    llama_150m_train_config,
    llama_300m,
    llama_300m_train_config,
)
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp600_tootsie import tootise_phase1_config
from experiments.tootsie.exp859_big_tootsies import nemotron_mix
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import LMMixtureDatasetConfig
from marin.resources import TpuPodConfig

# We want to have a consistent way of scaling up the learning rate.
# Strategy: figure out a "base LR" based on what we did with the Marin 8B model
# Then do base LR * sqrt(batch_size) / width
MARIN_8B_LR = tootise_phase1_config.learning_rate
MARIN_8B_BS = tootise_phase1_config.train_batch_size
MARIN_8B_WIDTH = llama_8b.hidden_dim
BASE_LR = MARIN_8B_LR / sqrt(MARIN_8B_BS) * MARIN_8B_WIDTH


def scaling_train_config(model_config: LlamaConfig, train_config: SimpleTrainConfig) -> SimpleTrainConfig:
    """Return a train config that scales the learning rate based on the model size."""
    # Use the same batch size for all models (chosen to work for the largest
    # 70B model, but okay-ish for smaller models)
    train_config = replace(
        train_config,
        train_batch_size=8192,
        weight_decay=0.05,
        z_loss_weight=1e-4,
        lr_schedule="wsd",
        decay=0,  # Don't decay
        min_lr_ratio=1,  # Keep the same learning rate (redundant)
    )

    # Scale the learning rate
    learning_rate = BASE_LR * sqrt(train_config.train_batch_size) / model_config.hidden_dim
    print(f"{learning_rate=} {BASE_LR=} {train_config.train_batch_size=} {model_config.hidden_dim=}")
    if not (0 < learning_rate < 1):
        raise ValueError(f"Learning rate {learning_rate} is out of range")
    return replace(
        train_config,
        learning_rate=learning_rate,
    )


# Like the one in exp859_big_tootsies but with a constant batch size
llama_32b_train_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-2048", slice_count=1),
    num_train_steps=1_000_000,  # To override later
    train_batch_size=-1,  # To override later
    learning_rate=float("nan"),  # To override later
    steps_per_eval=1000,
    steps_per_task_eval=10000,
)

# Like the one in exp750_tootsie70b but with a constant batch size
llama_70b_train_config_mk6 = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v6e-128", slice_count=6),
    num_train_steps=1_000_000,  # To override later
    train_batch_size=-1,  # To override later
    learning_rate=float("nan"),  # To override later
    steps_per_eval=1000,
    steps_per_task_eval=10000,
)


@dataclass(frozen=True)
class DataSpec:
    name: str
    data: LMMixtureDatasetConfig


@dataclass(frozen=True)
class TrainSpec:
    name: str
    model_config: LlamaConfig
    train_config: SimpleTrainConfig
    data_factor: int


data_specs = [
    DataSpec(name="dclm_mix", data=dclm_mixture_config_llama3),
    DataSpec(name="nemotron_mix", data=nemotron_mix),
]

train_specs = [
    TrainSpec(name="llama_150m", model_config=llama_150m, train_config=llama_150m_train_config, data_factor=8),
    TrainSpec(name="llama_300m", model_config=llama_300m, train_config=llama_300m_train_config, data_factor=8),
    TrainSpec(name="llama_1_4b", model_config=llama_1_4b, train_config=llama_1_4b_train_config, data_factor=8),
    TrainSpec(name="llama_8b", model_config=llama_8b, train_config=llama_8b_train_config, data_factor=8),
    TrainSpec(name="llama_32b", model_config=llama_32b, train_config=llama_32b_train_config, data_factor=2),
    TrainSpec(name="llama_70b", model_config=llama_70b, train_config=llama_70b_train_config_mk6, data_factor=1),
]


def create_train_step(data_spec: DataSpec, train_spec: TrainSpec):
    name = f"scaling-suite-{data_spec.name}-{train_spec.name}-{train_spec.data_factor}xC"
    model_config = train_spec.model_config

    # We are training with train_spec.data_factor x Chinchilla
    # Compute the number of tokens (assume = steps * batch_size * seq_len)
    num_parameters = compute_num_parameters(model_config, llama3_tokenizer_vocab_size)
    num_tokens = num_parameters * 20 * train_spec.data_factor
    num_train_steps = num_tokens // (train_spec.train_config.train_batch_size * model_config.seq_len)
    train_config = replace(train_spec.train_config, num_train_steps=num_train_steps)

    # Scale hyperparameters based on the model
    train_config = scaling_train_config(model_config, train_config)

    return default_train(
        name=name,
        tokenized=data_spec.data,
        model_config=model_config,
        train_config=train_config,
    )


steps = [create_train_step(data_spec, train_spec) for data_spec in data_specs for train_spec in train_specs]

if __name__ == "__main__":
    executor_main(
        steps=steps,
        description="Train scaling suite.",
    )
