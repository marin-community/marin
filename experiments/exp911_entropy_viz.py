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
This script evaluates log probabilities of the Tootsie 8b model and Llama 3.1 8B, as well as logging entropies.

Our goal is to see if there are any structural differences in the log probabilities of the two models.
"""

from experiments.defaults import default_tokenize, default_validation_sets
from experiments.tootsie.exp600_tootsie import llama3_tokenizer, llama_8b, llama_8b_tootsie_phase3
from experiments.tootsie.exp883_viz_compare_tootsie_llama import tulu_3_in_dolma
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.processing.tokenize.data_configs import mixture_for_evaluation

from marin.resources import TpuPodConfig

# We compare the models in CHECKPOINTS to Meta's Llama 3.1 8B  base model.
LLAMA = "meta-llama/Meta-Llama-3.1-8B"

CHECKPOINTS = [
    output_path_of(llama_8b_tootsie_phase3, "checkpoints/step-819924"),
]

eval_sets = default_validation_sets(tokenizer=versioned(llama3_tokenizer))
eval_sets = {
    **eval_sets,
    # TODO: this should really be a step.
    "tulu_sft": default_tokenize("tulu_sft", tulu_3_in_dolma, tokenizer=llama3_tokenizer, is_validation=True),
}
eval_set_mixture = mixture_for_evaluation(eval_sets)


all_steps = []
resource_config = TpuPodConfig(tpu_type="v4-8")

for checkpoint in CHECKPOINTS:
    all_steps.append(
        default_lm_log_probs(
            checkpoint, llama_8b, eval_set_mixture, resource_config, checkpoint_is_hf=False, max_samples_per_dataset=32
        )
    )

all_steps.append(
    default_lm_log_probs(
        LLAMA, llama_8b, eval_set_mixture, resource_config, checkpoint_is_hf=True, max_samples_per_dataset=32
    )
)

if __name__ == "__main__":
    executor_main(
        all_steps,
        description="Visualize log probabilities of Tootsie 8b and compare to Meta-Llama-3.1-8B",
    )
