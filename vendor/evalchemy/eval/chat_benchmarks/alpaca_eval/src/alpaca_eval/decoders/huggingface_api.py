import functools
import logging
import multiprocessing
import time
from typing import Sequence

import numpy as np
import tqdm
from huggingface_hub import InferenceClient

from .. import constants, utils

__all__ = ["huggingface_api_completions"]


def huggingface_api_completions(
    prompts: Sequence[str],
    model_name: str,
    do_sample: bool = False,
    num_procs: int = 1,
    **kwargs,
) -> dict[str, list]:
    """Decode with the API from hugging face hub.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    do_sample : bool, optional
        Whether to use sampling for decoding.

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    kwargs :
        Additional kwargs to pass to `InferenceClient.__call__`.
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `huggingface_api_completions` on {n_examples} prompts using {model_name}.")

    inference = InferenceClient(
        model_name,
        token=constants.HUGGINGFACEHUB_API_TOKEN,
    )

    default_kwargs = dict(do_sample=do_sample, return_full_text=False)
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")

    with utils.Timer() as t:
        partial_completion_helper = functools.partial(
            inference_helper, inference=inference.text_generation, params=default_kwargs
        )
        if num_procs == 1:
            completions = [partial_completion_helper(prompt) for prompt in tqdm.tqdm(prompts, desc="prompts")]
        else:
            with multiprocessing.Pool(num_procs) as p:
                completions = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, prompts),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Time for {n_examples} completions: {t}")

    # unclear pricing
    price = [np.nan] * len(completions)
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)


def inference_helper(prompt: str, inference, params, n_retries=100, waiting_time=2) -> str:
    for _ in range(n_retries):
        try:
            # TODO: check why doesn't stop after </s>
            output = inference(prompt=prompt, **params)
        except Exception as error:
            if n_retries > 0:
                if "Rate limit reached" in error:
                    logging.warning(f"Rate limit reached... Trying again in {waiting_time} seconds.")
                    time.sleep(waiting_time)
                elif "Input validation error" in error and "max_new_tokens" in error:
                    params["max_new_tokens"] = int(params["max_new_tokens"] * 0.8)
                    logging.warning(
                        f"`max_new_tokens` too large. Reducing target length to {params['max_new_tokens']}, "
                        f"Retrying..."
                    )
                    if params["max_new_tokens"] == 0:
                        raise ValueError(f"Error in inference. Full error: {error}")
                else:
                    raise ValueError(f"Error in inference. Full error: {error}")
            else:
                raise ValueError(f"Error in inference. We tried {n_retries} times and failed. Full error: {error}")
        return output
