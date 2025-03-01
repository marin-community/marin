"""An experiment to evaluate the quality of individual splits of the Dolmino dataset.

We cooldown a 8B model on a 30/70 mixture of some high quality Dolmino split and Dolmino DCLM.
Link to issue: https://github.com/stanford-crfm/marin/issues/820
"""

from experiments.anneal_config import AnnealConfig
import dataclasses
from haliax.quantization import QuantizationConfig
from experiments.defaults import default_anneal
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolmino.tokenize_dolmino import get_dolmino_step, DOLMINO_DATASETS
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.exp72_baselines import fineweb_edu_tokenized
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_dclm = dclm_components_llama3["dclm_baseline"]

NUM_ANNEAL_TOKENS = 50_000_000_000

# Control model is 100% dolmino DCLM dataset
control_dataset_config = lm_mixture_data_config(
    components={"dolmino_dclm": dolmino_dclm},
    weights={"dolmino_dclm": 1.0},
)
control_anneal_config = AnnealConfig(
    dataset_config=control_dataset_config, num_anneal_training_tokens=NUM_ANNEAL_TOKENS, tpu_type="v5litepod-128"
)

control_model = default_anneal(name="llama-8b-anneal-control-eu-long", anneal_config=control_anneal_config)

int8_config = dataclasses.replace(control_model.config, quantization=QuantizationConfig(int8=True))
int8_model = dataclasses.replace(control_model, config=int8_config)


if __name__ == "__main__":
    executor_main(
        steps=[control_model, int8_model],
    )
