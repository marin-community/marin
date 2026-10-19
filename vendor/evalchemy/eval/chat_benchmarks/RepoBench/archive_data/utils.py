import gzip
import pickle
import json
from tqdm import tqdm
from typing import Union, Optional
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_data(split: str, task: str, language: str, settings: Optional[Union[str, list]], length: Optional[str] = None):
    """
    Load data from the specified task and language.

    Args:
        split: the split to load (train or test)
        task: the task to load (retrieval/r, completion/c, pipeline/p)
        language: the language to load (python or java)
        settings: the settings to load (xf-f, xf-r, or if)

    Returns:
        data: the loaded data
    """

    # some assertions

    # split must be one of train or test
    assert split.lower() in ["train", "test"], "split must be one of train or test"

    # task must be one of r/R/retrieval, c/C/completion, or p/P/pipeline
    if task.lower() == "r":
        task = "retrieval"
    elif task.lower() == "c":
        task = "completion"
    elif task.lower() == "p":
        task = "pipeline"

    # language must be one of python or java
    if language.lower() == "py":
        language = "python"

    # settings must be one of xf-f, xf-r, or if or a list of their combinations
    if not settings:
        # if settings is not specified, use all settings
        if task == "retrieval":
            settings = ["cross_file_first", "cross_file_random"]
        else:
            settings = ["cross_file_first", "cross_file_random", "in_file"]

    if isinstance(settings, str):
        settings = [settings]

    for i, setting in enumerate(settings):
        if setting.lower() == "xf-f":
            settings[i] = "cross_file_first"
        elif setting.lower() == "xf-r":
            settings[i] = "cross_file_random"
        elif setting.lower() == "if":
            settings[i] = "in_file"

    # check if the task is valid
    assert task.lower() in ["r", "c", "p", "retrieval", "completion", "pipeline"], "task must be one of R, C, or P"

    # check if the language is valid
    assert language.lower() in ["python", "java", "py"], "language must be one of python, java"

    if task == "retrieval":
        assert all(
            [setting.lower() in ["cross_file_first", "cross_file_random"] for setting in settings]
        ), "For RepoBench-R, settings must be one of xf-f or xf-r"
    else:
        assert all(
            [setting.lower() in ["cross_file_first", "cross_file_random", "in_file"] for setting in settings]
        ), "Settings must be one of xf-f, xf-r, or if"

        if length:
            assert length.lower() in ["2k", "8k"], "length must be one of 2k or 8k"

        # for completion, length must be specified
        if task == "completion":
            assert length is not None, "length must be specified for completion task"

    # load data
    if split == "train":
        data = {}

        for setting in tqdm(settings, desc=f"Loading data"):
            with gzip.open(f"{ROOT}/train/{task}/{language}/{setting}.pkl.gz", "rb") as f:
                data[setting] = pickle.load(f)

                # it has train, dev, and test, we combine them together
                if task == "retrieval":
                    # retrieval has easy and hard for each split
                    data[setting] = {
                        "easy": data[setting]["train"]["easy"]
                        + data[setting]["dev"]["easy"]
                        + data[setting]["test"]["easy"],
                        "hard": data[setting]["train"]["hard"]
                        + data[setting]["dev"]["hard"]
                        + data[setting]["test"]["hard"],
                    }
                elif task == "completion":
                    data[setting] = data[setting]["train"] + data[setting]["dev"] + data[setting]["test"]

    elif split == "test":
        data = {}

        for setting in tqdm(settings, desc=f"Loading data"):
            if length not in ["2k", "8k"] and task != "completion":
                with gzip.open(f"{ROOT}/test/{task}/{language}/{setting}.gz", "rb") as f:
                    data[setting] = pickle.load(f)
            else:
                with gzip.open(f"{ROOT}/test/{task}/{language}/{length}/{setting}.gz", "rb") as f:
                    data[setting] = pickle.load(f)

    if len(settings) == 1:
        return data[settings[0]]
    else:
        return [data[setting] for setting in settings]


def construct_trainable_data(data: list, language: Optional[str] = None):
    """
    construct the data that can be used for training

    Args:
        dic: the dictionary of data from load_data
        language: the language of the data

    Returns:
        return_data: the data that can be used for training (a list of dictionaries)
    """
    # data is a dictionary with keys: file_path, context, import_statement, code, next_line

    # check if the language is valid
    assert language is not None, "language must be set"

    assert language in ["python", "java"], "language must be one of [python, java]"

    return_data = []

    for d in data:
        # check if context is a string
        assert isinstance(d["context"], str), "context must be a string instead of snippet list"

        if language == "python":
            path = f"# Path: {d['file_path']}"

        elif language == "java":
            path = f"// Path: {d['file_path']}"

        _d = f"""{d['context']}
{path}
{d['import_statement']}

{d['code']}"""

        label = d["next_line"]

        return_data.append({"data": _d, "label": label})

    return return_data


# crop lines given a threshold and tokenized code
def crop_code_lines(code: Union[str, list], threshold: int):
    """
    Crop the code to the last threshold lines.

    Args:
        code: the code to crop (either a string or a list of tokens)
        threshold: the number of lines to keep

    Returns:
        code: the cropped code
    """

    # if the code is string, meaning it is not tokenized
    if isinstance(code, str):
        code = code.split("\n")

        # return the last threshold lines if the code is longer than the threshold
        if len(code) > threshold:
            return "\n".join(code[-threshold:])
        else:
            return "\n".join(code)

    # if the code is tokenized
    elif isinstance(code, list):
        # set the current number of lines to -1, since the last line is always a newline
        cur_lines = -1
        # iterate over the code list from the end to the beginning
        for i in range(len(code) - 1, -1, -1):
            # "Ċ" is the token for newline
            if "Ċ" in code[i]:
                cur_lines += 1

            # if the current number of lines reaches the threshold,
            # return the code from the current index to the end (do not include the newline token)
            if cur_lines == threshold:
                return code[i + 1 :]

        # if the code is shorter than the threshold, return the whole code
        return code


# comment code
def comment(code: str, language: str):
    """
    Comment the code.

    Args:
        code: the code to comment
        language: the language of the code

    Returns:
        code: the commented code
    """
    if language == "python":
        return "\n".join([f"# {line}" for line in code.split("\n")])
    elif language == "java":
        return "\n".join([f"// {line}" for line in code.split("\n")])
    else:
        raise ValueError("language must be one of [python, java]")


# get first line that is not a comment
def get_first_line_not_comment(code: str, language: str = "python"):
    """
    This function gets the first line of code that is not a comment.

    Args:
    code: Str, the code

    Returns:
    Str, the first line of code that is not a comment or the first line of code if there is no line that is not a comment
    """

    # check if the language is valid
    assert language in ["python", "java"], "language must be one of [python, java]"

    # first remove the \n at the beginning of the code
    code = code.lstrip("\n")

    lines = code.split("\n")
    in_multiline_comment = False

    if language == "python":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith("#"):
                continue
            # if the line is not a comment, then return the line
            return line

    elif language == "java":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and line.strip().startswith("/*"):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and line.strip().endswith("*/"):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith("//"):
                continue
            # if the line is not a comment, then return the line
            return line

    # if we cannot find a line that is not a comment, then return the first line
    return lines[0]
