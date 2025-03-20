"""
Train models with a training distribution that varies over time.
"""

import dataclasses

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_150m, llama_150m_train_config
from experiments.pretraining_datasets import slimpajama_6b, starcoderdata
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

# Tokenize the datasets
slimpajama_tokenized = default_tokenize(name="SlimPajama-6B", dataset=slimpajama_6b, tokenizer=llama3_tokenizer)
starcoderdata_tokenized = default_tokenize(name="starcoderdata", dataset=starcoderdata, tokenizer=llama3_tokenizer)

llama_150m_train_config = dataclasses.replace(
    llama_150m_train_config,
    num_train_steps=20000,  # 20000 * 1024 * 1024 = 20B tokens
)

num_sequences = llama_150m_train_config.num_train_steps * llama_150m_train_config.train_batch_size

# Create varying mixture config that transitions from SlimPajama to Starcoderdata
# Start with 90% SlimPajama, then shift to 90% Starcoderdata at the halfway point

data_config = lm_varying_mixture_data_config(
    components={
        "slimpajama": slimpajama_tokenized,
        "starcoderdata": starcoderdata_tokenized,
    },
    weights_list=[
        (0, {"slimpajama": 0.9, "starcoderdata": 0.1}),  # At step 0, start with mostly SlimPajama
        (num_sequences // 2, {"slimpajama": 0.1, "starcoderdata": 0.9}),  # Halfway, transition to mostly Starcoderdata
    ],
)

# Train the model using the varying mixture
varying_mixture_model = default_train(
    name="slimpajama-to-starcoderdata-150m-demo",
    tokenized=data_config,
    model_config=llama_150m,
    train_config=llama_150m_train_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[varying_mixture_model],
        description="Train 150M model transitioning from SlimPajama to Starcoderdata.",
    )
