import logging
from typing import Sequence
import os
import numpy as np
import requests
import copy
import json
from tqdm import tqdm
import time

try:
    from transformers import AutoTokenizer
except ImportError:
    pass

from .. import utils

__all__ = ["sambanova_completions"]
NUM_SECONDS_TO_SLEEP = 30


def sambanova_completions(
    prompts: Sequence[str],
    model_name: str = "llama3-405b",
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
    if is_chatml_prompt:
        # convert the linear prompt to chatml
        prompts = [
            tokenizer.apply_chat_template(utils.prompt_to_chatml(prompt), add_generation_prompt=True, tokenize=False)
            for prompt in prompts
        ]
    else:
        prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]

    completions = []
    for prompt in tqdm(prompts):
        # while True:
        #     messages = prompts
        #     breakpoint()
        #     tokenized_messages = tokenizer.apply_chat_template(messages)
        #     if len(tokenized_messages) < 3600:
        #         break
        #     ratio = 4000/len(tokenized_messages)
        #     ratio = min(ratio, .95)
        #     query = query[-int(ratio * len(query)):]
        #     print("lessening")
        messages = prompt
        payload = {
            "messages": messages,
            "max_tokens": 1024,
            "stop": ["[INST", "[INST]", "[/INST]", "[/INST]"],
            "model": model_name,
            "stream": "true",
        }

        key = os.getenv("SAMBAKEY")
        url = os.getenv("SAMBAURL")

        headers = {"Authorization": f"Basic {key}", "Content-Type": "application/json"}

        while True:
            post_response = requests.post(
                f"https://{url}/v1/chat/completions", json=payload, headers=headers, stream=True
            )
            if (
                post_response.status_code == 503
                or post_response.status_code == 504
                or post_response.status_code == 401
                or post_response.status_code == 429
            ):
                print(post_response.content)
                print(f"Attempt failed due to rate limit or gate timeout. Trying again...")
                time.sleep(NUM_SECONDS_TO_SLEEP)
                continue
            debug_response = copy.deepcopy(post_response)
            if post_response.status_code != 200:
                breakpoint()
            response_text = ""
            for line in post_response.iter_lines():
                if line.startswith(b"data: "):
                    data_str = line.decode("utf-8")[6:]
                    try:
                        line_json = json.loads(data_str)
                        if "choices" in line_json and "content" in line_json["choices"][0]["delta"]:
                            try:
                                response_text += line_json["choices"][0]["delta"]["content"]
                            except:
                                breakpoint()
                    except json.JSONDecodeError as e:
                        pass
            break
        completions.append(response_text)

    avg_time = [np.nan] * len(completions)
    price = [np.nan] * len(completions)
    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)
