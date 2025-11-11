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

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config, llama_300m, llama_300m_train_config
from experiments.pretraining_datasets import fineweb_edu, nemotron_cc, slimpajama, slimpajama_6b
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize

slimpajama_6b_tokenized = default_tokenize(name="SlimPajama-6B", dataset=slimpajama_6b, tokenizer=llama3_tokenizer)
slimpajama_6b_config = lm_data_config(slimpajama_6b_tokenized, permutation_type="linear")
slimpajama_6b_model = default_train(
    name="SlimPajama-6B-300m",
    tokenized=slimpajama_6b_config,
    model_config=llama_300m,
    train_config=llama_300m_train_config,
)

slimpajama_tokenized = ExecutorStep(
    name=os.path.join("tokenized", "SlimPajama-627B"),
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[slimpajama.cd("train")],
        validation_paths=[slimpajama.cd("validation")],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
)
slimpajama_config = lm_data_config(slimpajama_tokenized, permutation_type="linear")
slimpajama_model = default_train(
    name="SlimPajama-627B-1.4b",
    tokenized=slimpajama_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

fineweb_edu_tokenized = default_tokenize(name="fineweb-edu", dataset=fineweb_edu, tokenizer=llama3_tokenizer)
fineweb_edu_config = lm_data_config(fineweb_edu_tokenized, permutation_type="linear")
fineweb_edu_model = default_train(
    name="fineweb-edu-1.4b",
    tokenized=fineweb_edu_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

nemotron_cc_tokenized = default_tokenize(name="nemotron_cc", dataset=nemotron_cc, tokenizer=llama3_tokenizer)
nemotron_cc_config = lm_data_config(nemotron_cc_tokenized, permutation_type="linear")
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
            nemotron_cc_tokenized,
            nemotron_cc_model,
        ],
        description="Train 1.4B models on standard datasets (SlimPajama 6B, SlimPajama, FineWebEdu, Nemotron-CC).",
    )
