import copy
import logging
import os
import pathlib
import random
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import datasets
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml


# don't load from utils to avoid unnecessary dependencies
AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyData = Union[Sequence[dict[str, Any]], pd.DataFrame, datasets.Dataset]
DUMMY_EXAMPLE = dict(instruction="1+1=", output_1="2", input="", output_2="3")


def read_or_return(to_read: Union[AnyPath, str], relative_to: Optional[str] = None, **kwargs):
    """Read a file or return the input if it is already a string."""
    is_str_input = isinstance(to_read, str)

    if relative_to is not None:
        to_read = Path(relative_to) / to_read

    try:
        with open(Path(to_read), **kwargs) as f:
            out = f.read()
    except FileNotFoundError as e:
        if is_str_input and Path(to_read).is_absolute():
            # The path is not absolute, so it's not just a string
            raise e

        logging.warning(f"Returning input because file not found. Error: {e}")
        out = to_read

    return out


def random_seeded_choice(seed: Union[int, str, float], choices, **kwargs):
    """Random choice with a (potentially string) seed."""
    return random.Random(seed).choices(choices, k=1, **kwargs)[0]


def make_prompts(
    df: pd.DataFrame,
    template: str,
    batch_size: int = 1,
) -> tuple[list[str], pd.DataFrame]:
    r"""Helper function to make batch prompts for a single template.

    Parameters
    ----------
    df : pd.DataFrame
        Examples to annotate

    template : str
        Template for the prompt. Should have batch_size number of placeholder {key} where key is a column in df.

    batch_size : int
        Number of examples to batch in a single prompt.

    Returns
    -------
    prompts : list[str]
        List of formatted prompts.

    df_out : pd.DataFrame
        All examples. Will be df with potential padding examples.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"instruction": ["solve", "write backwards", "other 1", "pad"],
    ...                    "input": ["1+1", "'abc'", "", "pad_in"]})
    >>> make_prompts(df, template="first: {instruction} {input}, second: {instruction} {input}", batch_size=2)[0]
    ["first: solve 1+1, second: write backwards 'abc'", 'first: other 1 , second: pad pad_in']
    """

    if df.empty:
        return [], df

    text_to_format = re.findall(r"{([^ \s]+?)}", template)
    n_occurrences = Counter(text_to_format)

    if not all([n == batch_size for n in n_occurrences.values()]):
        raise ValueError(f"All placeholders should be repeated batch_size={batch_size} times but {n_occurrences}.")

    if len(df) % batch_size > 0:
        raise ValueError(
            f"The number of rows should be dividable by the batch_size={batch_size} but got {len(df)}."
            "You should use PaddingForBatchesProcessor"
        )

    df_out = df.copy()
    prompts = []
    # ugly for loops, not trivial to vectorize because of the batching
    for i in range(0, len(df_out), batch_size):
        current_prompt = copy.deepcopy(template)
        for j in range(batch_size):
            for to_format in n_occurrences.keys():
                # replace only first occurrence (that's why we don't use .format)
                current_prompt = current_prompt.replace("{" + to_format + "}", str(df_out.iloc[i + j][to_format]), 1)
        prompts.append(current_prompt)

    return prompts, df_out


def convert_ordinal_to_binary_preference(
    preferences: Union[pd.DataFrame, list[dict[str, Any]]],
    ordinal_preference_key: str = "preference",
    binary_preference_key: str = "preference",
):
    """Convert ordinal preference annotations to preference annotations. By merging multiple subcategories together,
    eg A/a/b/B into A/B, or AA/A/a/b/B/BB into A/B.

    Parameters
    ----------
    preferences : pd.DataFrame or list of dicts
        List of dictionaries or a dataframe that contains ordinal preference A/a/b/B in ordinal_preference_key.

    ordinal_preference_key : str
        Key in the dictionaries or column name of the ordinal preference annotations.

    binary_preference_key : str
        Key in the dictionaries or column name of the binary preference annotations. This can be the same
        as ordinal_preference_key if you want to overwrite the ordinal preference annotations.

    Returns
    -------
    binary_preferences
        List of dictionary or a dataframe (same type as the input) that contains binary preferences A/B in
        binary_preference_key.

    Examples
    --------
    >>> from alpaca_eval.utils import convert_ordinal_to_binary_preference
    >>> preferences = [dict(output="test A", preference=1),
    ...                dict(output="test a", preference=2),
    ...                dict(output="test b", preference=3),
    ...                dict(output="test B", preference=4),
    ...                dict(output="test None", preference=0)]
    >>> convert_ordinal_to_binary_preference(preferences, ordinal_preference_key="preference",
    ...     binary_preference_key="preference")
    [{'output': 'test A', 'preference': 1},
     {'output': 'test a', 'preference': 1},
     {'output': 'test b', 'preference': 2},
     {'output': 'test B', 'preference': 2},
     {'output': 'test None', 'preference': 0}]
    """
    if isinstance(preferences, pd.DataFrame):
        is_df = True
    else:
        is_df = False
        preferences = pd.DataFrame.from_records(preferences)

    preferences[binary_preference_key] = (preferences[ordinal_preference_key].round().astype(int) - 1) // 2 + 1

    if not is_df:
        preferences = preferences.to_dict(orient="records")

    return preferences


def convert_to_dataframe(data: AnyData) -> pd.DataFrame:
    """Convert input that AlpacaEval accepts into a dataframe."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, datasets.Dataset):
        return data.data.to_pandas()
    elif isinstance(data, list):
        return pd.DataFrame.from_records(data)
    else:
        # try
        return pd.DataFrame(data)


def load_configs(configs: Union[AnyPath, dict], relative_to: Optional[AnyPath] = None):
    """Load the config yaml files, or return if it's already a dict."""
    if not isinstance(configs, dict):
        if relative_to is not None:
            configs = Path(relative_to) / configs
        configs = Path(configs)
        if configs.is_dir():
            configs = configs / "configs.yaml"
        with open(configs, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.exception(exc)
        assert isinstance(configs, dict)

    return configs


def get_module_attribute(module, func_name):
    """getattr but only if it's in __all__"""
    if func_name in module.__all__:
        return getattr(module, func_name)
    elif hasattr(module, func_name):
        raise AttributeError(
            f"The function {func_name} is not allowed,add it to __all__ if needed."
            f" Available functions: {module.__all__}"
        )
    else:
        raise AttributeError(f"The function {func_name} does not exist. Available functions: {module.__all__}")


def validate_alpacaeval_preference(x: float, is_allow_nan: bool = True) -> bool:
    """Validate the preference annotation."""
    return (1 <= x <= 2) or (is_allow_nan and np.isnan(x))


def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    r"""Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = (
    ... "<|im_start|>system\n"
    ... "You are a helpful assistant.\n<|im_end|>\n"
    ... "<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\n"
    ... "Who's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    ... )
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'content': 'You are a helpful assistant.', 'role': 'system'},
      {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
      {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
      {'content': 'Orange.', 'role': 'user'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = _string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


def _string_to_dict(to_convert):
    r"""Converts a string with equal signs to dictionary. E.g.
    >>> _string_to_dict(" name=user university=stanford")
    {'name': 'user', 'university': 'stanford'}
    """
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}


def get_package_version(package_name: str) -> str:
    """Get the version of a package."""
    return version(package_name)


def get_multi_package_version(package_names: Sequence[str]) -> str:
    """Get the version of multiple packages."""
    return " ".join([f"{p}=={get_package_version(p)}" for p in package_names])
