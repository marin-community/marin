#!/usr/bin/env python3
"""
Example script demonstrating how to use the Evalchemy evaluator with Marin.

This script shows how to run Evalchemy evaluations on a model using the Marin framework.
"""

import logging

from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.evaluator_factory import get_evaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Example of running Evalchemy evaluations on the Deeper Starling SFT model.
    
    This example evaluates the SFT model from:
    gs://marin-us-central2/checkpoints/sft/deeper_starling_sft_nemotron_and_openthoughts3/hf/step-1490000
    
    The model is a fine-tuned version of the Deeper Starling base model with
    instruction following capabilities, trained for 1.49M steps.
    Uses HuggingFace format checkpoint for compatibility with Evalchemy.
    """

    # Define the model configuration
    # model_config = ModelConfig(
    #     name="deeper-starling-sft",  # Model name for identification
    #     path="gs://marin-us-central2/checkpoints/sft/deeper_starling_sft_nemotron_and_openthoughts3/hf/step-1490000",  # HuggingFace format checkpoint
    #     engine_kwargs=None,
    #     apply_chat_template=True,  # SFT models typically use chat templates
    # )
    model_config = ModelConfig(
        name="meta-llama/Llama-3.2-1B",  # Small model for testing (117M parameters)
        path=None,  # Use HuggingFace model directly
        engine_kwargs=None,
        apply_chat_template=True,  # DialoGPT doesn't use chat templates
    )

    # Define evaluation tasks
    # Use evalchemy tasks
    evals = [
        EvalTaskConfig(
            name="AIME24",  # Instruction following evaluation
            num_fewshot=1,
        ),
        # EvalTaskConfig(
        #     name="IFEval",  # Multiple choice benchmark
        #     num_fewshot=1,
        # ),
    ]

    # Create evaluation configuration
    eval_config = EvaluationConfig(
        evaluator="evalchemy",
        model_name=model_config.name,
        evaluation_path="gs://marin-us-central2/evals/evalchemy/test_llama32_1b",  # Results path
        evals=evals,
        max_eval_instances=100,  # Limit for testing
        launch_with_ray=True,
        apply_chat_template=True,  # Match model config
    )

    # Get the evaluator
    evaluator = get_evaluator(eval_config)

    # Run the evaluation
    logger.info("Starting Evalchemy evaluation...")
    if eval_config.launch_with_ray:
        evaluator.launch_evaluate_with_ray(
            model=model_config,
            evals=evals,
            output_path=eval_config.evaluation_path,
            max_eval_instances=eval_config.max_eval_instances,
            resource_config=eval_config.resource_config,
        )
    else:
        evaluator.evaluate(
            model=model_config,
            evals=evals,
            output_path=eval_config.evaluation_path,
            max_eval_instances=eval_config.max_eval_instances,
        )
    logger.info("Evalchemy evaluation completed!")


if __name__ == "__main__":
    main()
