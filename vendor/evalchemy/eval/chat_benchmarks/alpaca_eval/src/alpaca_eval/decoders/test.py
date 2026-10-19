import logging
from typing import Sequence

from .. import utils

__all__ = ["test_completions"]


def test_completions(
    prompts: Sequence[str],
    model_name="test",
    value: str = "{'name': 'test'}",
    **decoding_kwargs,
) -> dict[str, list]:
    """Completion function for testing purposes. Returns the same value for all prompts."""

    n_examples = len(prompts)

    kwargs = dict(model_name=model_name, **decoding_kwargs)
    logging.info(f"Kwargs to completion: {kwargs}")
    with utils.Timer() as t:
        responses = [value for _ in prompts]
    avg_time = [t.duration / n_examples] * len(responses)
    price_per_example = [0] * len(responses)
    return dict(
        completions=responses, price_per_example=price_per_example, time_per_example=avg_time, completions_all=responses
    )
