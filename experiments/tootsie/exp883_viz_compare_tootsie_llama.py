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
This script visualizes the log probabilities of the Tootsie 8b model as compared to Llama 3.1 8B.
The goal was to see if there were any structural differences in the log probabilities of the two models.


"""

from experiments.defaults import default_validation_sets
from experiments.posttrain.instruction_datasets import tulu3_flat_llama_tokenized_as_validation
from experiments.tootsie.exp600_tootsie import llama3_tokenizer, llama_8b
from marin.evaluation.visualize import VizLmConfig, visualize_lm_log_probs
from marin.execution.executor import ExecutorStep, executor_main, versioned
from marin.processing.tokenize.data_configs import mixture_for_evaluation

# We compare the models in CHECKPOINTS to Meta's Llama 3.1 8B  base model.
COMPARISON_MODEL = "meta-llama/Meta-Llama-3.1-8B"

CHECKPOINTS = [
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/checkpoints/step-819924/",
]


def _path_to_step_name(path: str) -> str:
    """
    Converts a path pointing to a levanter checkpoint into a name we can use as an id for the viz step
    """
    # we want llama-8b-tootsie-phase2-730000
    components = path.split("/")
    step = components[-2].split("-")[-1]
    name = components[-4].split("/")[-1]
    return f"analysis/viz-compare/{name}-{step}"


eval_sets = default_validation_sets(tokenizer=versioned(llama3_tokenizer))
eval_sets = {
    **eval_sets,
    # TODO: this should really be a step.
    "tulu_sft": tulu3_flat_llama_tokenized_as_validation,
}
tulu_3_in_dolma = mixture_for_evaluation(eval_sets)


all_steps = []

for checkpoint in CHECKPOINTS:
    name = _path_to_step_name(checkpoint)
    all_steps.append(
        ExecutorStep(
            name=name,
            fn=visualize_lm_log_probs,
            config=VizLmConfig(
                checkpoint_path=checkpoint,
                model=llama_8b,
                datasets=tulu_3_in_dolma,
                num_docs_per_dataset=32,
                comparison_model_path=COMPARISON_MODEL,
                comparison_is_hf=True,
            ),
        )
    )

if __name__ == "__main__":
    executor_main(
        all_steps,
        description="Visualize log probabilities of Tootsie 8b and compare to Meta-Llama-3.1-8B",
    )
