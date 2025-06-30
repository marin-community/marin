"""
This script visualizes the log probabilities of the OLMo 2 SFT model at various stages of training.

Similar to exp826_viz_tootsie.py, this script visualizes the log probabilities of the OLMo 2 model
at various checkpoints to observe how the model's predictions change during training.

Uses Hugging Face models instead of Levanter for visualization.
"""

from levanter.models.olmo import Olmo2Config

from experiments.defaults import default_validation_sets
from experiments.instruction_datasets import tulu_3_in_dolma
from experiments.models import get_model_local_path, olmo2_7b
from marin.evaluation.visualize import VizLmConfig, visualize_lm_log_probs
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import mixture_for_evaluation

# Get the base model path using the models.py utilities
BASE_MODEL_PATH = get_model_local_path(olmo2_7b)

# SFT checkpoints to analyze
# These should be HF model paths
CHECKPOINTS = [
    "gs://marin-us-central2/checkpoints/olmo2_sft/hf/seed_0/nsxwiwl7/step-1000/",
    "gs://marin-us-central2/checkpoints/olmo2_sft/hf/seed_0/nsxwiwl7/step-2000/",
    "gs://marin-us-central2/checkpoints/olmo2_sft/hf/seed_0/nsxwiwl7/step-2999/",
]


def path_to_step_name(path):
    """Convert a checkpoint path to a step name for visualization.

    Args:
        path (str): Path to a checkpoint directory
            Example: "gs://marin-us-central2/checkpoints/olmo2_sft/hf/seed_0/nsxwiwl7/step-1000/"

    Returns:
        str: A formatted step name for visualization
            Example: "analysis/viz/olmo2-sft-step-1000"
    """
    # Format: olmo2-sft-step-XXXX
    components = path.split("/")
    step_component = components[-2] if path.endswith("/") else components[-1]
    step = step_component.split("-")[1]  # Extract the number after "step-"
    return f"analysis/viz/olmo2-sft-step-{step}"


# Create the OLMo 2 config based on the values in olmo2_sft.yaml
olmo2_config = Olmo2Config(
    seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=11008,
    num_layers=32,
    num_heads=32,
    num_kv_heads=32,
    use_flash_attention=True,
    flash_attention_block_size=512,
    use_bias=False,
    use_layer_norm_weight=True,
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,
    activation_function="silu",
    attention_bias=False,
    upcast_attn=True,
)

# Use a string for the tokenizer instead of a function
TOKENIZER_ID = "allenai/OLMo-2-1124-7B-SFT"

# Create a tokenize step for Tulu using OLMo tokenizer
tulu_tokenized = ExecutorStep(
    name="tokenized/tulu3_dolma",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=versioned([]),  # Empty list for train paths since we're only using validation
        validation_paths=[tulu_3_in_dolma],
        cache_path=this_output_path(),
        tokenizer=versioned(TOKENIZER_ID),
    ),
)

# Get evaluation datasets and add Tulu
eval_sets = default_validation_sets(tokenizer=TOKENIZER_ID)
eval_sets = {
    **eval_sets,
    "tulu3_dolma": tulu_tokenized,
}
eval_set_mixture = mixture_for_evaluation(eval_sets)

all_steps = []
# Add the base model visualization step
all_steps.append(
    ExecutorStep(
        name="analysis/viz/olmo2-base",
        fn=visualize_lm_log_probs,
        config=VizLmConfig(
            checkpoint_path=BASE_MODEL_PATH,
            model=olmo2_config,
            datasets=eval_set_mixture,
            num_docs_per_dataset=32,
            comparison_model_path=None,
            comparison_is_hf=True,
            checkpoint_is_hf=True,
        ),
    )
)

# Add steps for each checkpoint
for checkpoint in CHECKPOINTS:
    name = path_to_step_name(checkpoint)
    all_steps.append(
        ExecutorStep(
            name=name,
            fn=visualize_lm_log_probs,
            config=VizLmConfig(
                checkpoint_path=checkpoint,
                model=olmo2_config,
                datasets=eval_set_mixture,
                num_docs_per_dataset=32,
                comparison_model_path=BASE_MODEL_PATH,  # Compare against the base model
                comparison_is_hf=True,  # Set comparison_is_hf flag to True
                checkpoint_is_hf=True,
            ),
        )
    )


if __name__ == "__main__":
    executor_main(
        all_steps,
        description="Visualize log probabilities of OLMo 2 SFT at various stages of training using HuggingFace models",
    )
