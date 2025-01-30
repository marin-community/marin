import dataclasses
import logging
from collections.abc import Sequence


def default_scaling_laws_suite(
    experiment: ExecutorStep,
    scaling_law_config: ScalingLawConfig,
    widths: Sequence[int] = (512, 768, 1024, 1536, 2048),
    base_lr: float = 3e-4 * 4096,
    max_lr: float = 5e-3,
    intermediate_scale: float = 8,
) -> list[ExecutorStep]:
    """
    Creates a suite of model configurations and scaling law analysis steps based on width scaling.

    Given a base experiment configuration, generates a sequence of models with increasing widths
    and appropriately scaled hyperparameters, following the approach from:
    https://arxiv.org/pdf/2412.04403 and https://arxiv.org/pdf/2407.21783

    Args:
        experiment: Base experiment configuration
        scaling_law_config: Configuration for scaling law analysis
        widths: Sequence of model widths to generate (hidden dimensions)
        base_lr: Base learning rate to scale from
        max_lr: Maximum learning rate cap
        intermediate_scale: Scaling factor for intermediate dimension relative to width

    Returns:
        List of ExecutorSteps including ladder models and analysis step

    The implementation maintains key relationships:
        - Uses 128 head dimension throughout
        - Scales intermediate dimension proportionally with width
        - Adjusts learning rate inversely with width
        - Maintains appropriate head counts and KV head ratios
    """
    steps = []
    base_config = experiment.config

    for width in widths:
        # Calculate model dimensions
        intermediate_dim = _round_to_multiple(intermediate_scale * width, 128)
        head_size = base_config.model.hidden_dim // base_config.model.num_heads
        num_heads = width // head_size
        num_kv_heads = min(num_heads, 8)

        # Adjust num_kv_heads if needed
        if num_heads % num_kv_heads != 0:
            num_kv_heads = num_heads

        # Create model config
        model_config = dataclasses.replace(
            base_config.model,
            hidden_dim=width,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        # Scale learning rate
        lr = min(base_lr / width, max_lr)
        train_config = dataclasses.replace(base_config.trainer, learning_rate=lr)

        logging.info(f"Creating ladder step with width {width} and lr {lr}")

        # Create step with modified configs
        step = dataclasses.replace(
            experiment,
            name=f"{experiment.name}_ladder_{width}",
            config=dataclasses.replace(base_config, model=model_config, trainer=train_config),
        )
        steps.append(step)

    # Add scaling law analysis step
    analysis_step = ExecutorStep(
        name=f"{experiment.name}_scaling_analysis",
        function=run_scaling_law_analysis,
        config=scaling_law_config,
        dependencies=steps,
    )

    return steps + [analysis_step]


def _round_to_multiple(x: float, multiple: int) -> int:
    return int(multiple * round(x / multiple))
