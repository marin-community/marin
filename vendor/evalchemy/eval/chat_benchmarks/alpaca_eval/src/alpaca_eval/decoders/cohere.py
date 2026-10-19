import copy
import functools
import logging
import multiprocessing
import random
from typing import Optional, Sequence, Tuple

import cohere
import tqdm
from cohere import CohereError

from .. import constants, utils

__all__ = ["cohere_completions"]


def cohere_completions(
    prompts: Sequence[str],
    model_name="command",
    num_procs: int = 5,
    **decoding_kwargs,
) -> dict[str, list]:
    """Decode with Cohere API.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model to use for decoding.

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    decoding_kwargs :
        Additional kwargs to pass to `cohere.Client.generation`.
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `cohere_completions` on {n_examples} prompts using {model_name}.")

    kwargs = dict(model=model_name, **decoding_kwargs)
    logging.info(f"Kwargs to completion: {kwargs}")

    with utils.Timer() as t:
        if num_procs == 1:
            completions_and_token_counts = [
                _cohere_completion_helper(prompt, **kwargs) for prompt in tqdm.tqdm(prompts, desc="prompts")
            ]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(_cohere_completion_helper, **kwargs)
                completions_and_token_counts = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, prompts),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")
    completions, num_tokens = zip(*completions_and_token_counts)
    price_per_token = 0.000015  # cohere charges $0.000015 per token.
    price_per_example = [price_per_token * n for n in num_tokens]
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=list(completions), price_per_example=price_per_example, time_per_example=avg_time)


def _cohere_completion_helper(
    prompt: str,
    cohere_api_keys: Optional[Sequence[str]] = (constants.COHERE_API_KEY,),
    max_tokens: int = 1000,
    temperature: Optional[float] = 0.7,
    max_tries=5,
    **kwargs,
) -> Tuple[str, int]:
    cohere_api_key = random.choice(cohere_api_keys)
    client = cohere.Client(cohere_api_key)

    kwargs.update(dict(max_tokens=max_tokens, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)

    for trynum in range(max_tries):  # retry errors
        try:
            response = client.generate(prompt=prompt, return_likelihoods="ALL", **curr_kwargs)
            text = response[0].text
            num_tokens = len(response[0].token_likelihoods)

            if text == "":
                raise CohereError("Empty string response")

            return text, num_tokens

        except CohereError as e:
            logging.info(f"Try #{trynum+1}/{max_tries}: Error running prompt {repr(prompt)}: {e}")

    return " ", 0  # placeholder response for errors, doesn't allow empty string
