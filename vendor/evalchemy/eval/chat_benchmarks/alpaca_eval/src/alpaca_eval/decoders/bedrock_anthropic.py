import copy
import functools
import json
import logging
import multiprocessing
import time
from typing import Optional, Sequence, Union

import boto3
import botocore.exceptions
import tqdm

from .. import utils

__all__ = ["bedrock_anthropic_completions"]

DEFAULT_NUM_PROCS = 3


def bedrock_anthropic_completions(
    prompts: Sequence[str],
    max_tokens_to_sample: Union[int, Sequence[int]] = 2048,
    model_name: str = "anthropic.claude-v1",
    num_procs: int = DEFAULT_NUM_PROCS,
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

    decoding_kwargs :
        Additional kwargs to pass to Bedrock Anthropic.
    """
    num_procs = num_procs or DEFAULT_NUM_PROCS

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        to_log = f"Using `bedrock_anthropic_completions` on {n_examples} prompts using {model_name} and num_procs={num_procs}."
        logging.info(to_log)

    if isinstance(max_tokens_to_sample, int):
        max_tokens_to_sample = [max_tokens_to_sample] * n_examples

    inputs = zip(prompts, max_tokens_to_sample)

    kwargs = dict(model_name=model_name, **decoding_kwargs)
    kwargs_to_log = {k: v for k, v in kwargs.items() if "api_key" not in k}
    logging.info(f"Kwargs to completion: {kwargs_to_log}")
    with utils.Timer() as t:
        if num_procs == 1:
            responses = [
                _bedrock_anthropic_completion_helper(inp, **kwargs) for inp in tqdm.tqdm(inputs, desc="prompts")
            ]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(_bedrock_anthropic_completion_helper, **kwargs)
                responses = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, inputs),
                        desc="prompts",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    completions = responses

    ## Token counts are not returned by Bedrock for now
    price = [0 for _ in prompts]

    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time, completions_all=responses)


def _bedrock_anthropic_completion_helper(
    args: tuple[str, int],
    sleep_time: int = 2,
    region: Optional[str] = "us-west-2",
    model_name: str = "anthropic.claude-v1",
    temperature: Optional[float] = 0.7,
    **kwargs,
):
    prompt, max_tokens = args

    if not utils.check_pkg_atleast_version("boto3", "1.28.58"):
        raise ValueError("boto3 version must be at least 1.28.58 Use `pip install -U boto3`.")

    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)
    accept = "application/json"
    contentType = "application/json"

    kwargs.update(dict(max_tokens_to_sample=max_tokens, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)
    while True:
        try:
            body = json.dumps({**{"prompt": prompt}, **curr_kwargs})
            response = bedrock.invoke_model(body=body, modelId=model_name, accept=accept, contentType=contentType)
            response = json.loads(response.get("body").read()).get("completion")
            break
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                logging.warning(f"Hit throttling error: {e}.")
                logging.warning(f"Rate limit hit. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
        except Exception as e:
            logging.error(f"Hit unknown error : {e}")
            raise e

    return response
