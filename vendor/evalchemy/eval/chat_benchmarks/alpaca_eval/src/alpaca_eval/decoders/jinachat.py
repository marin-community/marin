import json
import logging
import multiprocessing
import os
import time
from functools import partial
from typing import Optional, Sequence

import requests

from .. import utils

__all__ = ["jina_chat_completions"]


def jina_chat_completions(
    prompts: Sequence[str],
    num_procs: Optional[int] = 4,
) -> dict[str, list]:
    """Get jina chat completions for the given prompts. Allows additional parameters such as tokens to avoid and
    tokens to favor.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.
    num_procs : int, optional
        Number of parallel processes to use for decoding.
    """
    n_examples = len(prompts)
    api_key = os.environ.get("JINA_CHAT_API_KEY")

    if n_examples == 0:
        logging.info("No samples to annotate.")
        return {}
    else:
        logging.info(f"Using `jina_chat_completions` on {n_examples} prompts.")

    prompts = [utils.prompt_to_chatml(prompt.strip()) for prompt in prompts]
    num_processes = min(multiprocessing.cpu_count(), num_procs)
    with utils.Timer() as t:
        with multiprocessing.Pool(processes=num_processes) as pool:
            logging.info(f"Number of processes: {pool._processes}")
            get_chat_completion_with_key = partial(_get_chat_completion, api_key)
            completions_and_num_tokens = pool.map(get_chat_completion_with_key, prompts)

    completions = [text for text, _ in completions_and_num_tokens]
    num_tokens = [tokens for _, tokens in completions_and_num_tokens]

    logging.info(f"Completed {n_examples} examples in {t}.")

    # refer to https://chat.jina.ai/billing
    price_per_example = [0.08 if msg_tokens > 300 else 0 for msg_tokens in num_tokens]
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price_per_example, time_per_example=avg_time)


def _get_chat_completion(api_key, prompt):
    url = "https://api.chat.jina.ai/v1/chat/completions"
    headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
    json_payload = {"messages": prompt}

    max_retries = 10

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_payload)
            response.raise_for_status()  # Will raise an HTTPError if one occurred.
            message = response.json()["choices"][0]["message"]["content"]
            message_tokens = response.json()["usage"]["completion_tokens"]
            return message, message_tokens
        except (json.JSONDecodeError, requests.exceptions.HTTPError) as e:
            logging.warning(f"Error occurred: {e}, Attempt {attempt + 1} of {max_retries}")
            time.sleep(5)
            if attempt + 1 == max_retries:
                logging.exception("Max retries reached. Raising exception.")
                logging.exception(f"Request data -> URL: {url}, Headers: {headers}, JSON Payload: {json_payload}")
                raise
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {e}")
            raise
