import functools
import logging
import multiprocessing
from typing import Sequence

import numpy as np
import replicate
import tqdm

from .. import utils

__all__ = ["replicate_completions"]


def replicate_completions(
    prompts: Sequence[str],
    model_name: str,
    num_procs: int = 32,
    **decoding_kwargs,
) -> dict[str, list]:
    r"""Get completions using a model hosted on https://replicate.com/.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str
        Name of the model endpoint on replicate. Format: owner/name:version

    num_procs : int, optional
        Number of parallel processes to use for decoding.

    decoding_kwargs :
        Additional kwargs to pass to `replicate.run(input={...})`. E.g. `temperature`, `max_length`, `top_p`, etc.
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `replicate` on {n_examples} prompts using {model_name} using {num_procs} processes.")

    logging.info(f"Kwargs to completion: {decoding_kwargs}")

    with utils.Timer() as t:
        partial_completion_helper = functools.partial(
            _replicate_completion_helper, model_name=model_name, **decoding_kwargs
        )
        if num_procs == 1:
            completions = [
                partial_completion_helper(model_name, prompt, **decoding_kwargs)
                for prompt in tqdm.tqdm(prompts, desc="prompts")
            ]
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

    # unclear pricing because it will depend on
    price = [np.nan] * len(completions)
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)


def _replicate_completion_helper(prompt: str, model_name: str, **decoding_kwargs):
    """Get a single generation."""
    return "".join(
        replicate.run(
            model_name,
            input={"prompt": prompt, **decoding_kwargs},
        )
    ).strip()
