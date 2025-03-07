"""An experiment comparing standard v.s. int8 cooldown

We cooldown a 8B model on DCLM baine using both bf16 and int8 training to compare the results.
Link to issue: https://github.com/stanford-crfm/marin/issues/848
"""

import dataclasses

from haliax.quantization import QuantizationConfig

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dclm = dclm_components_llama3["dclm_baseline"]

NUM_ANNEAL_TOKENS = 50_000_000_000

control_dataset_config = lm_mixture_data_config(
    components={"dclm": dclm},
    weights={"dclm": 1.0},
)
control_anneal_config = AnnealConfig(
    dataset_config=control_dataset_config, num_anneal_training_tokens=NUM_ANNEAL_TOKENS, tpu_type="v5litepod-128"
)

control_model = default_anneal(name="llama-8b-anneal-bf16-control", anneal_config=control_anneal_config)

# Keep everything fixed except for int8 training.
int8_trainer = dataclasses.replace(control_model.config.trainer, quantization=QuantizationConfig(int8=True))
int8_config = dataclasses.replace(control_model.config, trainer=int8_trainer)
int8_model = dataclasses.replace(control_model, config=int8_config, name="llama-8b-anneal-int8")


if __name__ == "__main__":
    executor_main(
        steps=[control_model, int8_model],
    )
