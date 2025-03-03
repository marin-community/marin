from experiments.defaults import default_validation_sets
from experiments.exp600_tootsie import llama_8b, llama3_tokenizer
from levanter.data.text import LMMixtureDatasetConfig
from marin.evaluation.visualize import VizLmConfig, mixture_for_visualization, visualize_lm_lob_probs
from marin.execution.executor import ExecutorStep, executor_main, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

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
eval_set_mixture = mixture_for_visualization(eval_sets)


all_steps = []

for checkpoint in CHECKPOINTS:
    name = path_to_step_name(checkpoint)
    all_steps.append(
        ExecutorStep(
            name=name,
            fn=visualize_lm_lob_probs,
            config=VizLmConfig(
                checkpoint_path=checkpoint,
                model=llama_8b,
                datasets=eval_set_mixture,
                num_docs_per_dataset=32,
                comparison_model_path=COMPARISON_MODEL if checkpoint != COMPARISON_MODEL else None,
            )
        )
    )


PHASE_1_CHECKPOINTS = [

]


if __name__ == "__main__":
    executor_main(
        [*all_steps],
        description="Visualize log probabilities of a language model.",
    )
