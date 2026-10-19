import logging
from typing import Optional, Sequence

import numpy as np
import torch
import transformers
from peft import PeftModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import constants, utils

__all__ = ["huggingface_local_completions"]


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def huggingface_local_completions(
    prompts: Sequence[str],
    model_name: str,
    do_sample: bool = False,
    batch_size: int = 1,
    model_kwargs=None,
    cache_dir: Optional[str] = constants.DEFAULT_CACHE_DIR,
    remove_ending: Optional[str] = None,
    is_fast_tokenizer: bool = True,
    adapters_name: Optional[str] = None,
    **kwargs,
) -> dict[str, list]:
    """Decode locally using huggingface transformers pipeline.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    do_sample : bool, optional
        Whether to use sampling for decoding.

    batch_size : int, optional
        Batch size to use for decoding. This currently does not work well with to_bettertransformer.

    model_kwargs : dict, optional
        Additional kwargs to pass to from_pretrained.

    cache_dir : str, optional
        Directory to use for caching the model.

    remove_ending : str, optional
        The ending string to be removed from completions. Typically eos_token.

    kwargs :
        Additional kwargs to pass to `InferenceClient.__call__`.
    """
    model_kwargs = model_kwargs or {}
    if "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "auto"
    if "torch_dtype" in model_kwargs and isinstance(model_kwargs["torch_dtype"], str):
        model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `huggingface_local_completions` on {n_examples} prompts using {model_name}.")

    if not torch.cuda.is_available():
        model_kwargs["load_in_8bit"] = False
        model_kwargs["torch_dtype"] = None

    #  faster but slightly less accurate matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",
        use_fast=is_fast_tokenizer,
        **model_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs).eval()

    if adapters_name:
        logging.info(f"Merging adapter from {adapters_name}.")
        model = PeftModel.from_pretrained(model, adapters_name)
        model = model.merge_and_unload()

    if batch_size == 1:
        try:
            model = model.to_bettertransformer()
        except:
            # could be not implemented or natively supported
            pass

    logging.info(f"Model memory: {model.get_memory_footprint() / 1e9} GB")

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1])))
        prompts = list(prompts)

    if not tokenizer.pad_token_id:
        # set padding token if not set
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    default_kwargs = dict(
        do_sample=do_sample,
        model_kwargs={k: v for k, v in model_kwargs.items() if k != "trust_remote_code"},
        batch_size=batch_size,
    )
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **default_kwargs,
        trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )

    ## compute and log the time for completions
    prompts_dataset = ListDataset(prompts)
    completions = []

    with utils.Timer() as t:
        for out in tqdm(
            pipeline(
                prompts_dataset,
                return_full_text=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        ):
            generated_text = out[0]["generated_text"]
            if remove_ending is not None and generated_text.endswith(remove_ending):
                generated_text = generated_text[: -len(remove_ending)]
            completions.append(generated_text)

    logging.info(f"Time for {n_examples} completions: {t}")

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))
        completions = list(completions)

    # local => price is really your compute
    price = [np.nan] * len(completions)
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)
