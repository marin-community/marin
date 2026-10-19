import copy
import functools
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Union

import anthropic
import numpy as np
import tqdm

from .. import constants, utils

__all__ = ["anthropic_completions"]


def anthropic_completions(
    prompts: Sequence[str],
    max_tokens_to_sample: Union[int, Sequence[int]] = 2048,
    model_name="claude-v1",
    num_procs: int = constants.ANTHROPIC_MAX_CONCURRENCY,
    price_per_token: Optional[float] = None,
    client_function_name: Optional[str] = "messages",  # newer anthropic models
    requires_chatml: bool = True,
    **decoding_kwargs,
) -> dict[str, list]:
    """Decode with Anthropic API.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model to use for decoding.

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    price_per_token : float, optional
        Price per token for the model.

    client_function_name : bool, optional
        Name of the function that should be called on the client object. "messages for newer anthropic models.

    requires_chatml : bool, optional
        Whether client_function_name requires chatML format.

    decoding_kwargs :
        Additional kwargs to pass to `anthropic.Anthropic.create`.
    """
    num_procs = num_procs or constants.ANTHROPIC_MAX_CONCURRENCY
    if client_function_name == "completions":
        requires_chatml = False

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return dict(completions=[], price_per_example=[], time_per_example=[], completions_all=[])
    else:
        to_log = f"Using `anthropic_completions` on {n_examples} prompts using {model_name} and num_procs={num_procs}."
        logging.info(to_log)

    if isinstance(max_tokens_to_sample, int):
        max_tokens_to_sample = [max_tokens_to_sample] * n_examples

    if requires_chatml:
        prompts = [utils.prompt_to_chatml(prompt) for prompt in prompts]

    inputs = zip(prompts, max_tokens_to_sample)

    kwargs = dict(model=model_name, client_function_name=client_function_name, **decoding_kwargs)
    kwargs_to_log = {k: v for k, v in kwargs.items() if "api_key" not in k}
    logging.info(f"Kwargs to completion: {kwargs_to_log}")
    with utils.Timer() as t:
        if num_procs == 1:
            responses = [_anthropic_completion_helper(inp, **kwargs) for inp in tqdm.tqdm(inputs, desc="prompts")]
        else:
            with ThreadPoolExecutor(max_workers=num_procs) as p:
                partial_completion_helper = functools.partial(_anthropic_completion_helper, **kwargs)
                responses = list(
                    tqdm.tqdm(
                        p.map(partial_completion_helper, inputs),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    completions = [response["text"] for response in responses]

    # anthropic doesn't return total tokens but 1 token approx 4 chars
    price = [
        (len(p) + len(c)) / 4 * _get_price_per_token(model_name, price_per_token) for p, c in zip(prompts, completions)
    ]

    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time, completions_all=responses)


def _anthropic_completion_helper(
    args: tuple[str, int],
    sleep_time: int = 2,
    anthropic_api_keys: Optional[Sequence[str]] = (constants.ANTHROPIC_API_KEY,),
    temperature: Optional[float] = 0.7,
    n_retries: Optional[int] = 10,
    client_function_name: Optional[str] = "messages",
    **kwargs,
):
    prompt, max_tokens = args

    anthropic_api_keys = anthropic_api_keys or (constants.ANTHROPIC_API_KEY,)
    anthropic_api_key = random.choice(anthropic_api_keys)

    if not utils.check_pkg_atleast_version("anthropic", "0.18.0"):
        raise ValueError("Anthropic version must be at least 0.18.0. Use `pip install -U anthropic`.")

    client = anthropic.Anthropic(api_key=anthropic_api_key, max_retries=n_retries)

    kwargs.update(dict(max_tokens=max_tokens, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)

    response = None
    for _ in range(n_retries):
        try:
            response = getattr(client, client_function_name).create(messages=prompt, **curr_kwargs)
            response = response.model_dump()
            response["text"] = response["content"][0]["text"]

            break

        except anthropic.RateLimitError as e:
            logging.warning(f"API RateLimitError: {e}.")
            if len(anthropic_api_keys) > 1:
                anthropic_api_key = random.choice(anthropic_api_keys)
                client = anthropic.Anthropic(api_key=anthropic_api_key, max_retries=n_retries)
                logging.info(f"Switching anthropic API key.")
            logging.warning(f"Rate limit hit. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)

        except anthropic.APITimeoutError as e:
            logging.warning(f"API TimeoutError: {e}. Retrying request.")

        except anthropic.APIError as e:
            response = dict(text="", stop_reason="api_error")
            break

    if response is None:
        logging.warning(f"Max retries reached. Returning empty completion.")
        response = dict(text="", stop_reason="max_retries_exceeded")

    return response


def _get_price_per_token(model, price_per_token=None):
    """Returns the price per token for a given model"""
    if price_per_token is not None:
        return float(price_per_token)

    elif "claude-v1" in model or "claude-2" in model:
        # https://www-files.anthropic.com/production/images/model_pricing_dec2023.pdf
        return (
            8 / 1e6
        )  # that's not completely true because decoding is 32.68 but close enough given that most is context
    else:
        logging.warning(f"Unknown model {model} for computing price per token.")
        return np.nan
