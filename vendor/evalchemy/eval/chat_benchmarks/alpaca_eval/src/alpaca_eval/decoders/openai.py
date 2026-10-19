import copy
import functools
import json
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence, Union

import numpy as np
import openai
import tiktoken
import tqdm
from openai import OpenAI

from .. import constants, utils

__all__ = ["openai_completions"]


def openai_completions(
    prompts: Sequence[str],
    model_name: str,
    max_tokens: Union[int, Sequence[int]] = 2048,
    tokens_to_favor: Optional[Sequence[str]] = None,
    tokens_to_avoid: Optional[Sequence[str]] = None,
    is_skip_multi_tokens_to_avoid: bool = True,
    is_strip: bool = True,
    num_procs: Optional[int] = constants.OPENAI_MAX_CONCURRENCY,
    batch_size: Optional[int] = None,
    price_per_token: Optional[float] = None,
    **decoding_kwargs,
) -> dict[str, list]:
    r"""Get openai completions for the given prompts. Allows additional parameters such as tokens to avoid and
    tokens to favor.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str
        Name of the model to use for decoding.

    tokens_to_favor : list of str, optional
        Substrings to favor in the completions. We will add a positive bias to the logits of the tokens constituting
        the substrings.

    tokens_to_avoid : list of str, optional
        Substrings to avoid in the completions. We will add a large negative bias to the logits of the tokens
        constituting the substrings.

    is_skip_multi_tokens_to_avoid : bool, optional
        Whether to skip substrings from tokens_to_avoid that are constituted by more than one token => avoid undesired
        side effects on other tokens.

    is_strip : bool, optional
        Whether to strip trailing and leading spaces from the prompts.

    price_per_token : float, optional
        Price per token for the model. If not provided, we will try to infer it from the model name.

    decoding_kwargs :
        Additional kwargs to pass to `openai.Completion` or `openai.ChatCompletion`.

    Example
    -------
    >>> prompts = ["Respond with one digit: 1+1=", "Respond with one digit: 2+2="]
    >>> openai_completions(prompts, model_name="text-davinci-003", tokens_to_avoid=["2"," 2"])['completions']
    ['\n\nAnswer: \n\nTwo (or, alternatively, the number "two" or the numeral "two").', '\n\n4']
    >>> openai_completions(prompts, model_name="text-davinci-003", tokens_to_favor=["2"])['completions']
    ['2\n\n2', '\n\n4']
    >>> openai_completions(prompts, model_name="text-davinci-003",
    ... tokens_to_avoid=["2 a long sentence that is not a token"])['completions']
    ['\n\n2', '\n\n4']
    >>> chat_prompt = ["<|im_start|>user\n1+1=<|im_end|>", "<|im_start|>user\nRespond with one digit: 2+2=<|im_end|>"]
    >>> openai_completions(chat_prompt, "gpt-3.5-turbo", tokens_to_avoid=["2"," 2"])['completions']
    ['As an AI language model, I can confirm that 1+1 equals  02 in octal numeral system, 10 in decimal numeral
    system, and  02 in hexadecimal numeral system.', '4']
    """
    num_procs = num_procs or constants.OPENAI_MAX_CONCURRENCY

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `openai_completions` on {n_examples} prompts using {model_name}.")

    if tokens_to_avoid or tokens_to_favor:
        tokenizer = tiktoken.encoding_for_model(model_name)

        logit_bias = decoding_kwargs.get("logit_bias", {})
        if tokens_to_avoid is not None:
            for t in tokens_to_avoid:
                curr_tokens = tokenizer.encode(t)
                if len(curr_tokens) != 1 and is_skip_multi_tokens_to_avoid:
                    logging.warning(f"'{t}' has more than one token, skipping because `is_skip_multi_tokens_to_avoid`.")
                    continue
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = -100  # avoids certain tokens

        if tokens_to_favor is not None:
            for t in tokens_to_favor:
                curr_tokens = tokenizer.encode(t)
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = 7  # increase log prob of tokens to match

        decoding_kwargs["logit_bias"] = logit_bias

    if is_strip:
        prompts = [p.strip() for p in prompts]

    requires_chatml = decoding_kwargs.pop("requires_chatml", _requires_chatml(model_name))
    decoding_kwargs["is_chat"] = decoding_kwargs.get("is_chat", requires_chatml)
    if requires_chatml:
        prompts = [utils.prompt_to_chatml(prompt) for prompt in prompts]
        num_procs = num_procs or 2
        batch_size = batch_size or 1

        if batch_size > 1:
            logging.warning("batch_size > 1 is not supported yet for chat models. Setting to 1")
            batch_size = 1

    else:
        num_procs = num_procs or 3
        batch_size = batch_size or 10

    n_batches = int(math.ceil(n_examples / batch_size))

    prompt_batches = [prompts[batch_id * batch_size : (batch_id + 1) * batch_size] for batch_id in range(n_batches)]

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * n_examples

    inputs = zip(prompt_batches, max_tokens)

    kwargs = dict(model=model_name, **decoding_kwargs)
    kwargs_to_log = {k: v for k, v in kwargs.items() if "api_key" not in k}
    logging.info(f"Kwargs to completion: {kwargs_to_log}. num_procs={num_procs}")

    with utils.Timer() as t:
        if num_procs == 1:
            completions = [
                _openai_completion_helper(inp, **kwargs)
                for inp in tqdm.tqdm(inputs, desc="prompt_batches", total=len(prompts))
            ]
        else:
            with ThreadPoolExecutor(max_workers=num_procs) as p:
                partial_completion_helper = functools.partial(_openai_completion_helper, **kwargs)
                completions = list(
                    tqdm.tqdm(
                        p.map(partial_completion_helper, inputs),
                        desc="prompt_batches",
                        total=len(prompts),
                    )
                )
    logging.info(f"Completed {n_examples} examples in {t}.")

    # flatten the list and select only the text
    completions_all = [completion for completion_batch in completions for completion in completion_batch]
    completions_text = [completion["text"] for completion in completions_all]

    price = [
        completion["total_tokens"] * _get_price_per_token(model_name, price_per_token)
        for completion_batch in completions
        for completion in completion_batch
    ]
    avg_time = [t.duration / n_examples] * len(completions_text)

    return dict(
        completions=completions_text,
        price_per_example=price,
        time_per_example=avg_time,
        completions_all=completions_all,
    )


def _openai_completion_helper(
    args: tuple[Sequence[str], int],
    is_chat: bool,
    sleep_time: int = 2,
    top_p: Optional[float] = 1.0,
    temperature: Optional[float] = 0.7,
    client_config_path: utils.AnyPath = constants.OPENAI_CLIENT_CONFIG_PATH,  # see `client_configs/README.md`
    # following is only for backward compatibility and should be avoided
    openai_organization_ids: Optional[Sequence[str]] = constants.OPENAI_ORGANIZATION_IDS,
    openai_api_keys: Optional[Sequence[str]] = constants.OPENAI_API_KEYS,
    openai_api_base: Optional[str] = os.getenv("OPENAI_API_BASE") if os.getenv("OPENAI_API_BASE") else openai.base_url,
    ############################
    client_kwargs: Optional[dict[str, Any]] = None,
    n_retries: Optional[int] = 10,
    **kwargs,
):
    client_kwargs = client_kwargs or dict()
    prompt_batch, max_tokens = args
    all_clients = utils.get_all_clients(
        client_config_path,
        model_name=kwargs["model"],
        get_backwards_compatible_configs=_get_backwards_compatible_configs,
        default_client_class="openai.OpenAI",
        backward_compatibility_kwargs=dict(
            openai_organization_ids=openai_organization_ids,
            openai_api_keys=openai_api_keys,
            openai_api_base=openai_api_base,
        ),
        **client_kwargs,
    )

    # randomly select the client
    client_idcs = range(len(all_clients))
    curr_client_idx = random.choice(client_idcs)
    logging.info(f"Using OAI client number {curr_client_idx+1} out of {len(client_idcs)}.")
    client = all_clients[curr_client_idx]

    # copy shared_kwargs to avoid modifying it
    kwargs.update(dict(max_tokens=max_tokens, top_p=top_p, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)

    # ensure no infinite loop
    choices = None
    for _ in range(n_retries):
        try:
            if is_chat:
                completion_batch = client.chat.completions.create(messages=prompt_batch[0], **curr_kwargs)

                choices = completion_batch.choices
                for i, choice in enumerate(choices):
                    # openai now returns pydantic objects => convert to dict to keep all code
                    # TODO should just rewrite code to use pydantic objects
                    choices[i] = choice.model_dump()
                    assert choice.message.role == "assistant"
                    if choice.message.content == "":
                        choices[i]["text"] = " "  # annoying doesn't allow empty string
                    else:
                        choices[i]["text"] = choice.message.content

                    # backward compatibility for function calls # TODO: remove once function calls are removed
                    if choice.message.function_call:
                        # currently we only use function calls to get a JSON object => return raw text of json
                        choices[i]["text"] = choice.message.function_call.arguments

                    if choice.message.tool_calls is not None:
                        # currently we only use function calls to get a JSON object => return raw text of json
                        choices[i]["text"] = choice.message.tool_calls[0].function.arguments

            else:
                completion_batch = client.completions.create(prompt=prompt_batch, **curr_kwargs)
                choices = completion_batch.choices
                for i, choice in enumerate(choices):
                    choices[i] = choice.model_dump()

            for choice in choices:
                choice["total_tokens"] = completion_batch.usage.total_tokens / len(prompt_batch)
            break
        except openai.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce" in str(e):
                kwargs["max_tokens"] = int(kwargs["max_tokens"] * 0.8)
                logging.warning(f"Reducing target length to {kwargs['max_tokens']}, Retrying...")
                if kwargs["max_tokens"] == 0:
                    logging.exception("Prompt is already longer than max context length. Error:")
                    raise e
            elif "Please try again with a different prompt." in str(e):
                logging.warning(
                    f"We got an obscure error from openAI. It's likely the spam filter so we are "
                    f"skipping this example."
                )
                # TODO: cleaner way to handle this? this batch will get filtered out as there is no completion
                choices = [dict(text="", total_tokens=0)] * len(prompt_batch)
                return choices

            else:
                if "rate limit" in str(e).lower():
                    logging.warning(f"Hit request rate limit; retrying...")
                else:
                    logging.warning(f"Unknown error. \n It's likely a rate limit so we are retrying...")
                if len(all_clients) > 1:
                    curr_client_idx = random.choice([idx for idx in client_idcs if idx != curr_client_idx])
                    client = all_clients[curr_client_idx]
                    logging.info(f"Switching OAI client to client number {curr_client_idx}.")
                logging.info(f"Sleeping {sleep_time} before retrying to call openai API...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    if choices is None:
        logging.warning(f"Max retries reached. Returning empty completions.")
        # TODO: the return is a tmp hack, should be handled better
        choices = [dict(text="", total_tokens=0)] * len(prompt_batch)

    return choices


def _requires_chatml(model: str) -> bool:
    """Whether a model requires the ChatML format."""
    # TODO: this should ideally be an OpenAI function... Maybe it already exists?
    not_chatml = ("instruct" in model) or ("gpt-3" in model and "turbo" not in model) or (model.startswith("text-"))
    return not not_chatml


def _get_price_per_token(model, price_per_token=None):
    """Returns the price per token for a given model"""
    if price_per_token is not None:
        return float(price_per_token)
    if "gpt-4o-mini-2024-07-18" in model:
        return (
            0.15 / 1_000_000
        )  # that's not completely true because decoding is 0.03 but close enough given that most is context
    elif "gpt-4-turbo" in model:
        return 0.01 / 1000
    elif "gpt-4-1106" in model:
        return (
            0.01 / 1000
        )  # that's not completely true because decoding is 0.03 but close enough given that most is context
    elif "gpt-4" in model:
        return (
            0.03 / 1000
        )  # that's not completely true because decoding is 0.06 but close enough given that most is context
    elif "gpt-3.5-turbo" in model:
        return 0.002 / 1000
    elif "text-davinci-003" in model:
        return 0.02 / 1000
    else:
        logging.warning(f"Unknown model {model} for computing price per token.")
        return np.nan


def _get_backwards_compatible_configs(
    openai_api_keys=[], openai_organization_ids=[None], openai_api_base=None
) -> list[dict[str, Any]]:
    if isinstance(openai_api_keys, str) or openai_api_keys is None:
        openai_api_keys = [openai_api_keys]
    if isinstance(openai_organization_ids, str) or openai_organization_ids is None:
        openai_organization_ids = [openai_organization_ids]

    client_configs = []
    for api_key in openai_api_keys:
        for org in openai_organization_ids:
            client_kwargs = dict(api_key=api_key)

            if org is not None:
                client_kwargs["organization"] = org

            if openai_api_base is not None:
                client_kwargs["base_url"] = openai_api_base

            client_configs.append(client_kwargs)

    return client_configs
