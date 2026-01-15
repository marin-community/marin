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

from marin.execution import deferred, executor_main, output, step, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, lm_mixture_data_config
from marin.processing.tokenize import tokenize as _tokenize

from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config, llama_300m, llama_300m_train_config
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.pretraining_datasets.simple import downloads

# Mark library functions as deferred
tokenize = deferred(_tokenize)


@step(name=os.path.join("tokenized", "SlimPajama-627B"))
def slimpajama_tokenized():
    return tokenize(
        TokenizeConfig(
            train_paths=[downloads["slimpajama"].cd("train")],
            validation_paths=[downloads["slimpajama"].cd("validation")],
            cache_path=output(),
            tokenizer=versioned(llama3_tokenizer),
        )
    )


############################################################


@step(name="baselines/all")
def run_all_baselines():
    """Entry point for baseline model training experiments."""
    slimpajama_6b_config = lm_data_config(slimpajama_6b_tokenized, permutation_type="linear")
    default_train(
        name="SlimPajama-6B-300m",
        tokenized=slimpajama_6b_config,
        model_config=llama_300m,
        train_config=llama_300m_train_config,
    )

    slimpajama_config = lm_data_config(slimpajama_tokenized(), permutation_type="linear")
    default_train(
        name="SlimPajama-627B-1.4b",
        tokenized=slimpajama_config,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )

    fineweb_edu_config = lm_data_config(fineweb_edu_tokenized, permutation_type="linear")
    default_train(
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
    default_train(
        name="nemotron_cc-1.4b",
        tokenized=nemotron_cc_config,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )


if __name__ == "__main__":
    executor_main(
        steps=[run_all_baselines()],
        description="Train 1.4B models on standard datasets (SlimPajama 6B, SlimPajama, FineWebEdu, Nemotron-CC).",
    )
