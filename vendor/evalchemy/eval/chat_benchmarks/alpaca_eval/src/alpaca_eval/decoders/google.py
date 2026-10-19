import functools
import logging
import multiprocessing
import random
import time
from typing import Optional, Sequence, Union

import google.generativeai as genai
import numpy as np
import tqdm

from .. import constants, utils

__all__ = ["google_completions"]


def google_completions(
    prompts: Sequence[str],
    max_output_tokens: Union[int, Sequence[int]] = 2048,
    model_name="gemini-pro",
    num_procs: int = constants.API_MAX_CONCURRENCY,
    **decoding_kwargs,  # ,
) -> dict[str, list]:
    """Decode with Anthropic API.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    max_output_tokens : int or list of int, optional
        Number of tokens to sample for each prompt. If a list, must be the same length as `prompts`.

    model_name : str, optional
        Name of the model to use for decoding.

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    decoding_kwargs :
        Additional kwargs to pass to `genai.types.GenerationConfig`.
    """
    num_procs = num_procs or constants.API_MAX_CONCURRENCY

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return dict(completions=[], price_per_example=[], time_per_example=[], completions_all=[])
    else:
        to_log = f"Using `google_completions` on {n_examples} prompts using {model_name} and num_procs={num_procs}."
        logging.info(to_log)

    if isinstance(max_output_tokens, int):
        max_output_tokens = [max_output_tokens] * n_examples

    inputs = zip(prompts, max_output_tokens)

    kwargs = dict(model_name=model_name, **decoding_kwargs)
    kwargs_to_log = {k: v for k, v in kwargs.items() if "api_key" not in k}
    logging.info(f"Kwargs to completion: {kwargs_to_log}")
    with utils.Timer() as t:
        if num_procs == 1:
            responses = [_google_completion_helper(inp, **kwargs) for inp in tqdm.tqdm(inputs, desc="prompts")]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(_google_completion_helper, **kwargs)
                responses = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, inputs),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    # anthropic doesn't return total tokens but 1 token approx 4 chars
    price = [_get_price(len(p), len(r), model_name) for p, r in zip(prompts, responses)]

    avg_time = [t.duration / n_examples] * len(responses)

    return dict(completions=responses, price_per_example=price, time_per_example=avg_time, completions_all=responses)


def _google_completion_helper(
    args: tuple[str, int],
    sleep_time: int = 2,
    temperature: Optional[float] = 0.7,
    model_name: str = "gemini-pro",
    google_api_keys: Optional[Sequence[str]] = None,
    max_tries=10,
    **kwargs,
):
    prompt, max_output_tokens = args

    google_api_keys = google_api_keys or (constants.GOOGLE_API_KEY,)
    google_api_key = random.choice(google_api_keys)

    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(model_name)
    n_tries = 0

    while True:
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    **kwargs,
                ),
                # don't block anything for evaluation
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "block_none",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
                    "HARM_CATEGORY_HATE_SPEECH": "block_none",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
                },
            )
            text = response.candidates[0].content.parts[0].text
            # num_tokens = model.count_tokens(text)

            return text

        # error code 429 is rate limit
        except Exception as e:
            if "429" in str(e):
                logging.info(f"Rate limit reached. Sleeping {sleep_time} seconds.")
                time.sleep(sleep_time)

            else:
                # TODO: better catching of errors when rate limits
                logging.exception(f"Unknown error, so we are retrying. Retry #{n_tries}/{max_tries}. Error:")
                time.sleep(sleep_time)
                n_tries += 1
                if n_tries > max_tries:
                    break

    return ""


def _get_price(n_in_char: int, n_out_char: int, model: str) -> float:
    """Returns the price per token for a given model"""
    if model == "gemini-pro":
        return (n_in_char * 0.00025 + n_out_char * 0.0005) / 1000

    else:
        logging.warning(f"Unknown model {model} for computing price per token.")
        return np.nan
