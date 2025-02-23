from experiments.defaults import default_validation_sets
from experiments.exp600_tootsie import llama_8b, llama3_tokenizer
from marin.evaluation.visualize import VizLmConfig, visualize_lm_lob_probs
from marin.execution.executor import executor_main

CHECKPOINTS = [
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/checkpoints/step-740000/",
]

def path_to_step_name(path):
    # we want llama-8b-tootsie-phase2-730000
    components = path.split("/")
    step = components[-2].split("-")[-1]
    name = components[-4].split("/")[-1]
    return f"analysis/viz/{name}-{step}"


eval_sets = default_validation_sets(tokenizer=llama3_tokenizer)


all_steps = []

for checkpoint in CHECKPOINTS:
    name = path_to_step_name(checkpoint)
    all_steps.append(
        visualize_lm_lob_probs(
            VizLmConfig(
                checkpoint_path=checkpoint,
                model=llama_8b,
                datasets=eval_sets,
                num_docs_per_dataset=256,
                output_path=name,
            )
        )
    )


if __name__ == "__main__":
    executor_main(
        [*all_steps],
        description="Visualize log probabilities of a language model.",
    )
