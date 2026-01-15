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
This script visualizes the log probabilities of the Tootsie 8b model at various stages of training.

@dlwh was interested in the weird loss behavior of the model after we switched to a longer WSD-S cooldown.
This script visualizes the log probabilities of the model at various stages of training to see if we can
spot any differences.

The differences were structural formatting differences in the eval data:

* Reddit data started with `&gt;` (sic) instead of `>`, which the model didn't like.
* Similarly, the twitter data uniformally ended with a ` ` (space) character, which the model didn't like.

The cooldown seems to function as a kind of sharpening/annealing
"""

from experiments.defaults import default_validation_sets
from experiments.llama import llama_8b_old_rotary
from experiments.tootsie.exp600_tootsie import llama3_tokenizer, llama_8b
from marin.evaluation.visualize import VizLmConfig, visualize_lm_log_probs as _visualize_lm_log_probs
from marin.execution import step, deferred, output, executor_main, versioned
from marin.processing.tokenize.data_configs import mixture_for_evaluation

visualize_lm_log_probs = deferred(_visualize_lm_log_probs)

COMPARISON_MODEL = "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000/"

CHECKPOINTS = [
    COMPARISON_MODEL,
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/checkpoints/step-740000/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/checkpoints/step-780000/",
]


def path_to_step_name(path):
    # we want llama-8b-tootsie-phase2-730000
    components = path.split("/")
    step = components[-2].split("-")[-1]
    name = components[-4].split("/")[-1]
    return f"analysis/viz/{name}-{step}"


eval_sets = default_validation_sets(tokenizer=versioned(llama3_tokenizer))
eval_set_mixture = mixture_for_evaluation(eval_sets)


all_steps = []

for checkpoint in CHECKPOINTS:
    name = path_to_step_name(checkpoint)

    @step(name=name)
    def viz_step(ckpt=checkpoint):
        return visualize_lm_log_probs(VizLmConfig(
            checkpoint_path=ckpt,
            model=llama_8b,
            datasets=eval_set_mixture,
            num_docs_per_dataset=32,
            comparison_model_path=COMPARISON_MODEL if ckpt != COMPARISON_MODEL else None,
        ))

    all_steps.append(viz_step())


PHASE_1_CONFIG = llama_8b_old_rotary
PHASE_1_BASE = "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/checkpoints/step-180000/"

PHASE_1_CHECKPOINTS = [
    PHASE_1_BASE,
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/checkpoints/step-200000/",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/checkpoints/step-500000/",
]

for checkpoint in PHASE_1_CHECKPOINTS:
    name = path_to_step_name(checkpoint)

    @step(name=name)
    def viz_phase1_step(ckpt=checkpoint):
        return visualize_lm_log_probs(VizLmConfig(
            checkpoint_path=ckpt,
            model=PHASE_1_CONFIG,
            datasets=eval_set_mixture,
            num_docs_per_dataset=32,
            comparison_model_path=PHASE_1_BASE if ckpt != PHASE_1_BASE else None,
        ))

    all_steps.append(viz_phase1_step())


if __name__ == "__main__":
    executor_main(
        all_steps,
        description="Visualize log probabilities of Tootsie 8b at various stages of training",
    )
