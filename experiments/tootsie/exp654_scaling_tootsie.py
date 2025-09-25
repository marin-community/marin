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

from thalas import executor_main

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3_wrong
from marin.training.scaling_laws import scaling_law_suite

TAG = ["654_scaling_tootsie"]

suite = scaling_law_suite(sweep_name="tootsie-scaling", tokenized=dclm_mixture_config_llama3_wrong, tags=TAG)

if __name__ == "__main__":
    executor_main(
        steps=[
            *suite,
        ],
        description="scaling law suite to predict performance of 8B model on DCLM mix",
    )
