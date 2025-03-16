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
from experiments.exp600_tootsie import llama3_tokenizer, llama_8b
from marin.evaluation.visualize import VizLmConfig, mixture_for_visualization, visualize_lm_log_probs
from marin.execution.executor import ExecutorStep, executor_main, versioned

COMPARISON_MODEL = "meta-llama/Meta-Llama-3.1-8B"

CHECKPOINTS = [
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/checkpoints/step-819924/",
]


def path_to_step_name(path):
    # we want llama-8b-tootsie-phase2-730000
    components = path.split("/")
    step = components[-2].split("-")[-1]
    name = components[-4].split("/")[-1]
    return f"analysis/viz-compare/{name}-{step}"


eval_sets = default_validation_sets(tokenizer=versioned(llama3_tokenizer))
eval_set_mixture = mixture_for_visualization(eval_sets)


all_steps = []

for checkpoint in CHECKPOINTS:
    name = path_to_step_name(checkpoint)
    all_steps.append(
        ExecutorStep(
            name=name,
            fn=visualize_lm_log_probs,
            config=VizLmConfig(
                checkpoint_path=checkpoint,
                model=llama_8b,
                datasets=eval_set_mixture,
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
