import contextlib
import copy
import glob
import itertools
import logging
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import datasets
import numpy as np
import numpy.typing as npt
import pandas as pd
import pkg_resources
import tqdm
import yaml

from . import constants

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


def is_derangement(arr1, arr2):
    """Whether 2 arrays are derangements of one another"""
    return all([a != b for a, b in zip(arr1, arr2)])


def random_derangement(arr, max_loop=10, seed=None):
    """
    Make random derangement of an array. I.e. shuffle without keeping any element in place. To be efficient,
    we first try `max_loop` rejection sampling. If didn't work then computes all possible derangement.
    """
    if len(arr) < 2:
        return arr

    rng = random.Random(seed)

    idcs = list(range(len(arr)))
    shuffled = list(range(len(arr)))

    for _ in range(max_loop):
        rng.shuffle(shuffled)
        if is_derangement(idcs, shuffled):
            return arr[shuffled]

    # if no luck then computes all possibilities
    deranged_order = list(set([s for s in itertools.permutations(idcs) if is_derangement(s, idcs)]))
    return arr[list(rng.choice(deranged_order))]


def _find_first_match(text: str, outputs_to_match: dict[str, Any]) -> tuple[Any, Any]:
    """Given text to parse and a dictionary of compiled regex to match, return the first match and corresponding key."""
    first_match = None
    first_key = None

    for key, compiled_regex in outputs_to_match.items():
        match = compiled_regex.search(text)
        if match and (not first_match or match.start() < first_match.start()):
            first_match = match
            first_key = key

    return first_match, first_key


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


def check_imports(modules: Sequence[str], to_use: str = "this fnction"):
    """Check whether the given module is imported."""
    modules = list(modules)
    for module in modules:
        if module not in sys.modules:
            error = f"You need {modules} to use {to_use}. Try `pip install {' '.join(modules)}`."
            raise ImportError(error)


def check_pkg_atleast_version(package, atleast_version):
    curr_version = get_package_version(package)
    return pkg_resources.parse_version(curr_version) > pkg_resources.parse_version(atleast_version)


def load_or_convert_to_dataframe(df=Union[AnyPath, AnyData, Callable, list, tuple], **kwargs):
    """Load a dataframe from a path or convert the input to a dataframe if it's not a path."""
    if isinstance(df, Callable):
        df = df(**kwargs)

    if isinstance(df, (tuple, list)) and isinstance(df[0], (str, os.PathLike, pathlib.Path)):
        df = pd.concat(
            [load_or_convert_to_dataframe(f, **kwargs) for f in df],
        )

    if isinstance(df, (str, os.PathLike, pathlib.Path)):
        df = Path(df)

        # check if it's a globbing pattern
        if "*" in str(df):
            df = pd.concat(
                [load_or_convert_to_dataframe(f, **kwargs) for f in glob.glob(str(df))],
            )
        else:
            suffix = df.suffix
            if suffix == ".json":
                df = pd.read_json(df, **kwargs)
            elif suffix == ".csv":
                df = pd.read_csv(df, **kwargs)
                if df.columns[0] == "Unnamed: 0":
                    df.set_index(df.columns[0], inplace=True)
                    df.index.name = None
            elif suffix == ".tsv":
                df = pd.read_table(df, sep="\t", **kwargs)
            else:
                raise ValueError(f"File format {suffix} not supported.")
    else:
        df = convert_to_dataframe(df, **kwargs)

    return df


class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.1f} seconds"


@contextlib.contextmanager
def silent():
    """Context manager to remove all outputs and warnings."""
    import IPython

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), DisableLogger(), IPython.utils.io.capture_output():
        yield


class DisableLogger:
    def __enter__(self):
        logging.disable(50)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


def contains_list(text: str, is_return_count: bool = False) -> Union[bool, int]:
    """
    Check if the text contains list items or count the number of list items.

    Parameters:
    - text (str): The input text to check.
    - is_return_count (bool): If True, return the count of list items. If False, return a boolean indicating presence of list items.

    Returns:
    - int or bool: Returns the count of list items if is_return_count is True, otherwise returns True/False.
    """

    # Patterns to match different types of list items
    bullet_point_pattern = r"(?m)^\s*[\*\•\-]\s*[^\w]*\w+"  # Matches bullets like "* item" or "• item" or "- item"
    number_list_pattern = r"(?m)^\s*\d+[\.\)]\s*[^\w]*\w+"  # Matches numbered lists like "1. item" or "1) item"
    alpha_list_pattern = r"(?m)^\s*[a-zA-Z][\.\)]\s+[^\w]*\w+"  # Matches lettered lists like "a. item" or "A) item"

    patterns = [
        bullet_point_pattern,
        number_list_pattern,
        alpha_list_pattern,
    ]

    count = 0

    for pattern in patterns:
        matches = re.findall(pattern, text)
        count += len(matches)

    if is_return_count:
        return count
    else:
        return count > 0


def prioritize_elements(lst: list, elements: Sequence) -> list:
    """Prioritize elements in a list. If elements are not in the list, they will be appended to the end of the list."""
    elements = list(elements)
    for el in elements:
        if el in lst:
            lst.remove(el)
    return elements + lst


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


def get_precomputed_leaderboard(precomputed_leaderboard, reference_outputs, annotators_config):
    if precomputed_leaderboard == "auto":
        try:
            precomputed_leaderboard = constants.PRECOMPUTED_LEADERBOARDS[
                (str(reference_outputs), str(annotators_config))
            ]
        except KeyError:
            try:
                if Path(reference_outputs).is_absolute():
                    logging.warning(
                        f"precomputed_leaderboard = 'auto'. But we have found no corresponding leaderboard for"
                        f" {reference_outputs} and {annotators_config}"
                    )
            except:
                logging.warning(f"precomputed_leaderboard = 'auto'. But we have found no corresponding leaderboard")
            precomputed_leaderboard = None

    if precomputed_leaderboard is not None:
        try:
            leaderboard = load_or_convert_to_dataframe(precomputed_leaderboard).to_dict(orient="index")
        except FileNotFoundError:
            logging.warning(f"precomputed_leaderboard = {precomputed_leaderboard} not found => computing from scratch.")
            leaderboard = dict()
    else:
        leaderboard = dict()
    return leaderboard, precomputed_leaderboard


def get_output_path(output_path, model_outputs, name, dflt_dir="results", annotators_config=None):
    if output_path == "auto":
        if model_outputs is None:
            output_path = None
        else:
            try:
                if Path(model_outputs).exists():
                    output_path = Path(model_outputs).parent
            except:
                pass

            if output_path == "auto":
                if name is not None:
                    output_path = Path(dflt_dir) / name
                else:
                    output_path = "."
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        if isinstance(annotators_config, str) and "/" not in annotators_config:
            output_path = Path(output_path) / annotators_config
            output_path.mkdir(exist_ok=True, parents=True)

    return output_path


def print_leaderboard(df_leaderboard, leaderboard_mode_or_models, cols_to_print, current_name=None):
    cols_to_print = list(cols_to_print)
    # make sure no duplicates and keep in order
    cols_to_print = list(dict.fromkeys(cols_to_print))

    if isinstance(leaderboard_mode_or_models, str):
        if "mode" in df_leaderboard.columns:
            # select all modes that come before
            current_idx = constants.ORDERED_LEADERBOARD_MODES.index(leaderboard_mode_or_models)
            df_leaderboard["mode_idx"] = df_leaderboard["mode"].apply(constants.ORDERED_LEADERBOARD_MODES.index)

            is_smaller_mode = df_leaderboard["mode_idx"] <= current_idx
            is_selected = is_smaller_mode | (df_leaderboard["mode"].isnull())

    elif isinstance(leaderboard_mode_or_models, Sequence):
        # check the index of the models
        is_selected = df_leaderboard.index.isin(leaderboard_mode_or_models)

    elif "mode" in df_leaderboard.columns:
        cols_to_print = cols_to_print + ["mode"]
        is_selected = [True] * len(df_leaderboard)

    if current_name is not None:
        is_selected |= df_leaderboard.index == current_name

    df_leaderboard = df_leaderboard[is_selected]

    print(df_leaderboard[cols_to_print].to_string(float_format="%.2f"))


def get_generator_name(name, model_outputs):
    if name is None:
        try:
            assert len(model_outputs["generator"].unique()) == 1
            name = model_outputs["generator"].iloc[0]
        except:
            name = "Current model"
    return name


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


def dataframe_chunk_generator(df: pd.DataFrame, chunksize: Optional[int] = None, tqdm_desc: Optional[str] = None):
    """Generator that yields chunks of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split into chunks.

    chunksize : int, optional
        The size of the chunks. If None, the chunksize will be the length of the dataframe.

    tqdm_desc : bool, optional
        Description to display in the tqdm progress bar. If None, no progress bar will be displayed.
    """
    if chunksize is None:
        chunksize = max(1, len(df))

    iterator = range(0, len(df), chunksize)

    if tqdm_desc is not None:
        iterator = tqdm.tqdm(iterator, desc=tqdm_desc)

    n_iter = len(df) // chunksize

    for i in iterator:
        df_chunk = df.iloc[i : i + chunksize]

        # if many iterations then better to copy the dataframe to avoid memory issues
        if n_iter > 1:
            df_chunk = df_chunk.copy()

        yield df_chunk


def validate_alpacaeval_preference(x: float, is_allow_nan: bool = True) -> bool:
    """Validate the preference annotation."""
    return (1 <= x <= 2) or (is_allow_nan and np.isnan(x))


def get_all_clients(
    client_config_path: AnyPath,
    model_name: str,
    default_client_class: str,
    get_backwards_compatible_configs: Callable,
    backward_compatibility_kwargs: dict = {},
    **kwargs,
) -> list:
    """Returns a list of different kwargs to pass to the client, each element corresponds to one possible client.
    For more information see `client_configs/README.md`.
    """

    client_config_path = Path(client_config_path)
    if client_config_path.is_file():
        with open(client_config_path) as f:
            all_client_configs = yaml.safe_load(f)

        client_configs = []

        if model_name in all_client_configs:
            if "default" in all_client_configs[model_name]:
                assert "default" in all_client_configs, "default client was asked for but not found"
                client_configs = client_configs + all_client_configs["default"]
                # remove "default" from the list of configs for this model
                all_client_configs[model_name] = [
                    config for config in all_client_configs[model_name] if config != "default"
                ]

            client_configs = client_configs + all_client_configs[model_name]

        else:
            assert (
                "default" in all_client_configs
            ), f"default client config is required as there are no model specific configs for {model_name}"
            client_configs = all_client_configs["default"]

    else:
        # backward compatibility
        logging.warning(
            f"{client_config_path} wasn't found. We are using environment variables to construct the client configs."
            "This is the old and non-recommended way of doing it. Please see `client_configs/README.md` for the "
            "recommended way of specifying client configs."
        )
        client_configs = get_backwards_compatible_configs(**backward_compatibility_kwargs)

    all_clients = []
    for config in client_configs:
        client_class = config.pop("client_class", default_client_class)
        ClientClass = import_class(client_class)
        all_clients.append(ClientClass(**config, **kwargs))

    return all_clients


def import_class(full_class_string):
    """
    Dynamically import a class from a string.

    Parameters
    ----------
    full_class_string:
        The full class string. E.g., 'openai.OpenAI' return OpenAI
    """
    module_name, class_name = full_class_string.rsplit(".", 1)

    # Import the module
    module = __import__(module_name, fromlist=[class_name])

    # Get the class
    cls = getattr(module, class_name)

    return cls


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
    return pkg_resources.get_distribution(package_name).version


def get_multi_package_version(package_names: Sequence[str]) -> str:
    """Get the version of multiple packages."""
    return " ".join([f"{p}=={get_package_version(p)}" for p in package_names])
