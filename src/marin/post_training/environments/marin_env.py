from typing import Any, NamedTuple

import numpy as np


class EnvStep(NamedTuple):
    """Container for a single interactive environment step.

    This class encapsulates all the data generated during one step of interaction
    with an environment, including the input problems (prompts), model responses,
    rewards computed, and additional metrics collected.

    Attributes:
        examples (list[dict[str, Any]]): A list of problem instances sampled from
            the dataset. Each instance contains data with keys:
                - 'prompt': Problem description
                - 'answer': Ground truth solution used for grading

        responses (list[list[dict[str, np.ndarray]]]): A nested list structure where
            responses[i][j] contains the j-th generated sample for the i-th problem
            in the batch. Each inner dict contains:
            - 'tokens': numpy array of generated token IDs
            - 'logprobs': numpy array of log probabilities for each generated token
            - Other generation-specific metadata

        rewards (np.ndarray): A 2D numpy array with shape
            (number of examples, number of generations per example) containing
            the computed reward for each generated response. Rewards are typically
            binary (0.0 or 1.0) indicating correctness, but can be continuous values.

        metrics (dict[str, float]): Additional scalar metrics computed during this
            environment step, such as:
            - Average reward across all responses
            - Format validation success rate
            - Average response length
            - Problem-specific evaluation metrics

    Example:
        >>> env_step = EnvStep(
        ...     examples=[{'prompt': 'What is 2+2?', 'answer': '4'}],
        ...     responses=[[[{'tokens': np.array([1, 2, 3]), 'logprobs': np.array([0.1, 0.2, 0.3])}]]],
        ...     rewards=np.array([[1.0]]),
        ...     metrics={'avg_reward': 1.0, 'avg_length': 3.0}
        ... )
    """

    examples: list[dict[str, Any]]
    responses: list[list[dict[str, np.ndarray]]]
    rewards: np.ndarray
    metrics: dict[str, float]


class MarinEnv:
    """Abstract base class for RL environments.

    This class defines the interface that all environment implementations must follow.
    Environments are responsible for:
    1. Managing datasets of problems/tasks
    2. Sampling problems for training or evaluation
    3. Running inference to generate responses
    4. Computing rewards based on responses
    5. Collecting metrics for monitoring training progress

    Subclasses must implement the `step` method to define environment-specific
    behavior for problem sampling, inference, and reward computation.

    Example:
        >>> class MyEnv(MarinEnv):
        ...     def __init__(self, dataset_path, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.dataset = load_dataset(dataset_path)
        ...
        ...     def step(self, sampler, params, n_examples, prng_key, **kwargs):
        ...         # Sample problems, run inference, compute rewards
        ...         return EnvStep(examples, responses, rewards, metrics)
    """

    def __init__(self, **kwargs):
        """Initialize the environment with environment-specific configuration.

        This method should be overridden by subclasses to perform any necessary
        setup such as:
        - Loading datasets and perform preprocessing
        - Configuring tokenizers (used for reward computation)
        """
        pass

    def step(self, **kwargs) -> EnvStep:
        """Execute one step of environment interaction.

        This is the main interface method that subclasses must implement. It should:
        1. Sample a batch of problems from the dataset
        2. Generate model responses using the provided sampler and parameters
        3. Compute rewards by comparing responses to ground truth
        4. Collect metrics for monitoring and logging
        5. Return all data packaged in an EnvStep container

        Returns:
            EnvStep: A container with the sampled examples, generated responses,
                computed rewards, and collected metrics from this environment step.
        """
        raise NotImplementedError("Subclasses must implement the step method")
