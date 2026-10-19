import logging
from typing import Sequence

import numpy as np

try:
    from transformers import AutoTokenizer
except ImportError:
    pass
from vllm import LLM, SamplingParams

from .. import utils

__all__ = ["vllm_local_completions"]

llm = None
llmModelName = None
tokenizer = None


def vllm_local_completions(
    prompts: Sequence[str],
    model_name: str,
    max_new_tokens: int,
    is_chatml_prompt: bool = False,
    batch_size: int | None = None,  # default of vllm is 256
    model_kwargs=None,
    **decoding_kwargs,
) -> dict[str, list]:
    """Decode locally using vllm transformers pipeline.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    max_new_tokens : int
        Maximum number of tokens to generate for each prompt.

    is_chatml_prompt : bool
        Whether the prompt is given in chatML format (like OpenAI chat models). If so this will be converted to a list
        of dict and then passed through tokenizer.apply_chat_template(prompt, add_generation_prompt=True,tokenize=False)
        to be converted in the right chat format for that model.

    batch_size : int, optional
        Batch size to use for decoding. If None uses the default batch size of vllm.

    model_kwargs : dict, optional
        Additional kwargs to pass to `vllm.LLM` constructor.

    decoding_kwargs :
        Additional kwargs to SamplingParams
    """
    global llm, llmModelName, tokenizer
    model_kwargs = model_kwargs or {}
    if batch_size is not None:
        model_kwargs["max_num_seqs"] = batch_size

    if model_name != llmModelName:
        logging.info(f"vllm already loaded model: {llmModelName} but requested {model_name}. Let's switch...")
        llm = None

    if llm is None:
        logging.info(f"vllm: loading model: {model_name}, {model_kwargs}")
        llm = LLM(model=model_name, tokenizer=model_name, **model_kwargs)
        llmModelName = model_name
        if is_chatml_prompt:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info(f"Sampling kwargs: {decoding_kwargs}")
    sampling_params = SamplingParams(max_tokens=max_new_tokens, **decoding_kwargs)

    if is_chatml_prompt:
        # convert the linear prompt to chatml
        prompts = [
            tokenizer.apply_chat_template(utils.prompt_to_chatml(prompt), add_generation_prompt=True, tokenize=False)
            for prompt in prompts
        ]

    with utils.Timer() as t:
        outputs = llm.generate(prompts, sampling_params)
    completions = [output.outputs[0].text for output in outputs]
    price = [np.nan] * len(completions)
    avg_time = [t.duration / len(prompts)] * len(completions)
    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)
