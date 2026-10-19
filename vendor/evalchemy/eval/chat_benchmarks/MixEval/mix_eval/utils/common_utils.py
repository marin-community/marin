import json
import os
import numpy as np
import random
import re

import tiktoken


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        os.environ["PYTHONHASHSEED"] = str(seed)
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def cache_status(args, status):
    status_path = os.path.join(
        args.output_dir, args.model_name, args.benchmark, args.version, f"status_{args.split}.json"
    )
    with open(status_path, "w") as f:
        arg_dict = args.__dict__
        status_dict = {"args": arg_dict, "status": status}
        json.dump(status_dict, f, indent=4)


def read_status(args):
    status_path = os.path.join(
        args.output_dir, args.model_name, args.benchmark, args.version, f"status_{args.split}.json"
    )
    with open(status_path, "r") as f:
        return json.load(f)


def dict_equal(dict1, dict2, keys_to_ignore=["resume", "compute_score_from_judged_file", "inference_only"]):
    modified_dict1 = dict1.copy()
    modified_dict2 = dict2.copy()
    for key in keys_to_ignore:
        modified_dict1.pop(key, None)
        modified_dict2.pop(key, None)
    return modified_dict1 == modified_dict2


def log_error(message, path):
    with open(path, "a") as f:
        f.write(f"{message}\n")


def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    import torch

    gpu_memory = []
    num_gpus = torch.cuda.device_count() if max_gpus is None else min(max_gpus, torch.cuda.device_count())

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def is_meaningful(s):
    # Check for alphabetic characters (including other languages) and numeric digits
    if re.search(
        r"[\u0030-\u0039\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F\u0180-\u024F\u0370-\u03FF\u0400-\u04FF\u0500-\u052F\u2C00-\u2C5F\uA640-\uA69F\uAC00-\uD7AF\u4E00-\u9FFF]",
        s,
    ):
        return True
    else:
        return False


def extract_basemodel_response_3e(response):
    _response = response.split("\n\n\nQuestion")[0]
    if is_meaningful(_response):  # non-trival response
        return _response
    else:
        return response


def extract_basemodel_response_2e(response):
    _response = response.split("\n\n")[0]
    if is_meaningful(_response):  # non-trival response
        return _response
    else:
        return response


def num_tokens_from_message(message, model="gpt-3.5-turbo-0613"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
        num_tokens = len(encoding.encode(message))
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
