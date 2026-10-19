import json
from pathlib import Path
from typing import Sequence

from alpaca_eval.decoders import get_fn_completions
from alpaca_eval.types import AnyPath

__all__ = ["cache_completions"]


def cache_completions(prompts: Sequence[str], fn_completions: str, cache_path: AnyPath, **completions_kwargs):
    """Simple wrapper around a completion function to cache the results to JSON on disk.
    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    fn_completions : str
        Function in `decoders.py` to use for decoding the output.

    cache_path : str
        Path to the cache file.

    completions_kwargs : dict
            kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

    """
    assert isinstance(fn_completions, str), "fn_completions must be a string to be hashable."
    all_args = [dict(prompt=p, fn_completions=fn_completions, completions_kwargs=completions_kwargs) for p in prompts]

    cache_path = Path(cache_path)

    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {}

    outs = []
    fn_completions = get_fn_completions(fn_completions)
    for args in all_args:
        hashable_args = json.dumps(args, sort_keys=True)
        if hashable_args not in cache:
            cache[hashable_args] = fn_completions(prompts=[args["prompt"]], **args["completions_kwargs"])
        outs.append(cache[hashable_args])

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    return outs
