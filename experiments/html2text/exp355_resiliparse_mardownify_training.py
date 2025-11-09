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
Test the Resiliparse custom fork on the fineweb-small dataset.
https://github.com/marin-community/marin/issues/355
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.evals.evals import default_eval
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config

logger = logging.getLogger("ray")
step_name = "fineweb-small-resiliparse-custom-fork"

fineweb_resiliparse_custom_fork_tokenized = default_tokenize(
    name=step_name,
    dataset="gs://marin-us-central2/documents/fineweb-small-resiliparse-custom-fork-ca2156/md/CC-MAIN-2024-18",
    tokenizer=llama3_tokenizer,
)
fineweb_resiliparse_custom_fork_1_4b_model = default_train(
    name=f"{step_name}-1.4b",
    tokenized=lm_data_config(fineweb_resiliparse_custom_fork_tokenized, permutation_type="linear"),
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

fineweb_resiliparse_custom_fork_1_4b_model_eval = default_eval(fineweb_resiliparse_custom_fork_1_4b_model)

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_resiliparse_custom_fork_tokenized,
            fineweb_resiliparse_custom_fork_1_4b_model,
            fineweb_resiliparse_custom_fork_1_4b_model_eval,
        ]
    )
