# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train 1.4B models on standard datasets (e.g., SlimPajama).
https://github.com/marin-community/marin/issues/72
"""

import os

from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config, llama_300m, llama_300m_train_config
from experiments.pretraining_datasets.simple import downloads, tokenized
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from marin.execution import step, StepContext, StepRef, executor_main, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, lm_mixture_data_config, tokenize

slimpajama_6b_tokenized = tokenized["slimpajama_6b"]
slimpajama_6b_config = lm_data_config(slimpajama_6b_tokenized, permutation_type="linear")
slimpajama_6b_model = default_train(
    name="SlimPajama-6B-300m",
    tokenized=slimpajama_6b_config,
    model_config=llama_300m,
    train_config=llama_300m_train_config,
)

@step(name=os.path.join("tokenized", "SlimPajama-627B"), fn=tokenize)
def slimpajama_tokenized_creator(ctx: StepContext):
    return TokenizeConfig(
        train_paths=[downloads["slimpajama"].cd("train")],
        validation_paths=[downloads["slimpajama"].cd("validation")],
        cache_path=ctx.output,
        tokenizer=versioned(llama3_tokenizer),
    )

slimpajama_tokenized = slimpajama_tokenized_creator()
slimpajama_config = lm_data_config(slimpajama_tokenized, permutation_type="linear")
slimpajama_model = default_train(
    name="SlimPajama-627B-1.4b",
    tokenized=slimpajama_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

fineweb_edu_tokenized = tokenized["fineweb_edu"]
fineweb_edu_config = lm_data_config(fineweb_edu_tokenized, permutation_type="linear")
fineweb_edu_model = default_train(
    name="fineweb-edu-1.4b",
    tokenized=fineweb_edu_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

nemotron_cc_steps = tokenize_nemotron(tokenizer=llama3_tokenizer)
nemotron_cc_config = lm_mixture_data_config(
    components=nemotron_cc_steps,
    weights=NEMOTRON_WEIGHTS,
    permutation_type="linear",
)
nemotron_cc_model = default_train(
    name="nemotron_cc-1.4b",
    tokenized=nemotron_cc_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            slimpajama_6b_model,
            slimpajama_model,
            fineweb_edu_model,
            nemotron_cc_model,
        ],
        description="Train 1.4B models on standard datasets (SlimPajama 6B, SlimPajama, FineWebEdu, Nemotron-CC).",
    )
