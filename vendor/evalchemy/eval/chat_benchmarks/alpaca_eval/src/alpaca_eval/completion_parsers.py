"""
All the parsers are functions that take as completion the output of the model `completion` and return a list of
annotations, one for each example in the batch. `completion` is typically a string but can actually be controlled by
`completion_key`, e.g., if you want to have access to logprobs. Parsers are pretty general but for AlpacaEval, the
output of the parser should be a list of float that follows the following format:
- `[np.nan]` if there was an issue in the parsing
- 1 means that the first example was better
- 2 means that the second example was better
- 0 means that both examples were equally good
- any other value is interpreted as a score between 1 and 2, where 1 means that the first example was better

If the desired parser does not follow this format, you can use `pipeline_meta_parser` to apply a sequence of parsers
until you get the desired output. E.g. using `replace_parser` to replace a string with a number.
"""

import ast
import copy
import json
import logging
import re
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from scipy.special import logsumexp

from . import utils

__all__ = [
    "regex_parser",
    "lmsys_parser",
    "ranking_parser",
    "json_parser",
    "eval_parser",
    "logprob_parser",
    "pipeline_meta_parser",
]


def regex_parser(completion: str, outputs_to_match: dict[Any, Any]) -> list[Any]:
    r"""Parse a single batch of completions, by returning a sequence of keys in the order in which outputs_to_match
    was matched.

    Parameters
    ----------
    completion : str
        Completion to parse.

    outputs_to_match : dict[str, Any]
        Dictionary of compiled regex to match. Keys are the keys to return in the order in which they are matched.
        The values can be either a compiled regex or a string. If a string, it will be compiled to a regex and that will
        be modified inplace.

    Examples
    --------
    >>> completion = ('\n(b)\n\n### Best output for example 8:\n(a)\n\n### Best output for example 9:\n(b)\n\n### Best'\
    ...               ' output for example 10:\n(a)\n\n### Best output for example 11:\n(a)')
    >>> regex_parser(completion, {1: r"\n\(a\)", 2: r"\n\(b\)"})
    [2, 1, 2, 1, 1]
    >>> regex_parser(' (a)', {1: r" \(a\)", 2: r" \(b\)"})
    [1]
    >>> completion = ('### Preferred output in JSON format for example 4:\r\n{{\r\n"Concise explanation": "Both'\
    ... ' outputs are incorrect, but Output (a) is less confusing and more concise.",\r\n"Output (a) is better than'\
    ... ' Output (b)": true\r\n}}\r\n\r\n### Preferred output in JSON format for example 5:\r\n{{\r\n"Concise'\
    ... ' explanation": "Both outputs are incomplete, but Output (b) seems to start with a more relevant source."'\
    ... ',\r\n"Output (a) is better than Output (b)": false\r\n}}\r\n\r\n### Preferred output in JSON format for'\
    ... ' example 6:\r\n{{\r\n"Concise explanation": "Both outputs are incorrect, but Output (a) is less confusing and'\
    ... ' more concise.",\r\n"Output (a) is better than Output (b)": true\r\n}}\r\n\r\n### Preferred output in JSON' \
    ... ' format for example 7:\r\n{{\r\n"Concise explanation": "Both outputs are incomplete, but Output (b) seems to'\
    ... ' start with a more relevant source.", \r\n"Output (a) is better than Output (b)": false\r\n}}')
    >>> regex_parser(completion, {1: ' true', 2: ' false'})
    [1, 2, 1, 2]
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    for k, v in outputs_to_match.items():
        if not isinstance(v, re.Pattern):
            # inplace modification, which is bad practice but useful to speedup
            outputs_to_match[k] = re.compile(v)

    completion = copy.deepcopy(completion)
    responses = []
    while True:
        match, key = utils._find_first_match(completion, outputs_to_match)
        if not match:
            break
        responses.append(key)
        # avoid matching the same output twice
        completion = completion[match.end() :]
    return responses


# modified from: https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/eval_gpt_review.py#L47
# does not work with batched completions
def lmsys_parser(completion: str) -> list[Any]:
    r"""Parse a pair of scores from a single completion and returns which is better.

    Examples
    --------
    >>> lmsys_parser("1 7\n ...")
    [2]
    >>> lmsys_parser("7 1\n more text")
    [1]
    >>> lmsys_parser("1 1\n ...")
    [0]
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    try:
        score_pair = completion.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            lmsys_score_1 = float(sp[0])
            lmsys_score_2 = float(sp[1])
            if lmsys_score_1 > lmsys_score_2:
                return [1]
            elif lmsys_score_1 < lmsys_score_2:
                return [2]
            else:
                return [0]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logging.error(f"{e}\nContent: {completion}\n" "You must manually fix the score pair.")
        return [np.nan]


def ranking_parser(completion: str, model_1_name: str = "model_1") -> list[Any]:
    r"""Parse a completion that contains a list of dictionary and returns the name of the preferred model.

    Examples
    --------
    >>> ranking_parser("[{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}]")
    [1]
    >>> ranking_parser("[{'model': 'model_1', 'rank': 2}, {'model': 'model_2', 'rank': 1}]")
    [2]
    >>> ranking_parser("[{'model': 'model_1', 'rank': 3}, {'model': 'model_2', 'rank': 1}]")
    [nan]
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    try:
        if isinstance(completion, str):
            ordered_completions = ast.literal_eval(completion)
        else:
            ordered_completions = completion

        rank = [c for c in ordered_completions if c["model"] == model_1_name][0]["rank"]
        assert rank in [1, 2]

        return [rank]
    except Exception as e:
        logging.error(f"{e}\nContent: {completion}\n" "You must manually fix the score pair.")
        return [np.nan]


def json_parser(completion: str, annotation_key: Optional[str]) -> list[Any]:
    r"""Parse the completion by reading it as a JSON and selecting "annotation_key".

    Examples
    --------
    >>> completion = '{"short_explanation": "that is why", "is_incorporated": true}'
    >>> json_parser(completion, "is_incorporated")
    [True]
    >>> completion = '[{"short_explanation": "that is why", "is_incorporated": true}, {"is_incorporated": false}]'
    >>> json_parser(completion, "is_incorporated")
    [True, False]
    >>> completion = 'blah ```json\n{"short_explanation": "that is why", "integer": 1}```'
    >>> json_parser(completion, "integer")
    [1]
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    # search for a pattern "```json{...}```" and take what is inside the curly brackets
    if "```json" in completion:
        completion = re.search(r"```json(.*?)```", completion, re.DOTALL).group(1)

    json_loaded = json.loads(completion)
    if isinstance(json_loaded, dict):
        return [json_loaded[annotation_key] if annotation_key is not None else json_loaded]
    return [d[annotation_key] if annotation_key is not None else d for d in json.loads(completion)]


def eval_parser(completion: str) -> list[Any]:
    """Parse the completion by evaluating it.

    Examples
    --------
    >>> eval_parser("True")
    [True]
    >>> eval_parser("(True,1,'False')")
    [(True, 1, 'False')]
    >>> eval_parser("[True,1,'False']")
    [True, 1, 'False']
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    evaluated_completion = ast.literal_eval(completion)
    if not isinstance(evaluated_completion, list):
        evaluated_completion = [evaluated_completion]
    return evaluated_completion


def replace_parser(completion: str, replacer: dict, default_replacer: Any = "auto") -> list[str]:
    """Parser that replaces part of the completion using a dictionary. This is useful if it's more natural for a
    prompt to ask a completion that is different from the one you want to store.

    Parameters
    ----------
    completion : str
        Output from the model to parse.

    replacer : dict
        Dictionary with keys that are the substring of the completion that you want to replace and values that are the
        replacements.

    default_replacer : any, optional
        If a key is not found in `replacer`, use this value instead. If "auto" then use the key itself.

    Examples
    --------
    >>> replace_parser("True", replacer={"True": 1})
    [1]
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    return [replacer.get(completion, completion if default_replacer == "auto" else default_replacer)]


def logprob_parser(
    completion: dict,
    numerator_token: str,
    denominator_tokens: Sequence[str],
    is_binarize: bool = True,
    log_prob_index: Union[Literal["batch"], int] = "batch",
) -> list[float]:
    """Parser that computes the logprob of a numerator token divided by the sum of the logprobs of the denominator
    tokens.

    Parameters
    ----------
    completion : dict
        Output from the model to parse.

    numerator_token : str
        Token of the numerator, i.e., what is used to say that the first output is better.

    denominator_tokens : Sequence[str]
        Tokens to use as denominator, i.e., all the other valid tokens.

    is_binarize : bool, optional
        If True, will binarize the output to 1 if the numerator token has the highest logprob and 2 otherwise.
        If False, will return the ratio of the numerator and denominator probabilities.

    log_prob_index : Union[Literal["batch"], int], optional
        If "batch", will return a list of logprobs for each example in the batch. In this case, the answer for batch i
        should be in position i. If an integer, then we batch size should be 1 and the desired answer should be the
        token of that position. E.g. `log_prob_index=-1` is useful when using chain for thought reasoning.
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    # completion should be like
    # [{'finish_reason': 'length', 'index': 0, 'logprobs': {'content': [{'token': 'M', 'bytes': [77], 'logprob': -0.017597131, 'top_logprobs': [{'token': 'M', 'bytes': [77], 'logprob': -0.017597131}, {'token': 'm', 'bytes': [109], 'logprob': -4.048847}, {'token': 'Both', 'bytes': [66, 111, 116, 104], 'logprob': -14.908222}, {'token': 'The', 'bytes': [84, 104, 101], 'logprob': -15.705097}, {'token': 'Based', 'bytes': [66, 97, 115, 101, 100], 'logprob': -15.955097}]}]}, 'message': {'content': 'M', 'role': 'assistant', 'function_call': None, 'tool_calls': None}, 'text': 'M', 'total_tokens': 390.0}]}
    # make sure completion["logprobs"]["content"][0]["top_logprobs"] exists
    assert "logprobs" in completion
    assert "content" in completion["logprobs"]

    def single_logprob_parser(top_logprobs: list[dict[str, Any]]) -> float:
        map_tokens_to_logprobs = {
            t["token"]: t["logprob"] for t in top_logprobs if t["token"] in denominator_tokens + [numerator_token]
        }

        # if it's not present we say it's probability is 0, which only makes sense if at least one is present
        missing = float("-inf")
        if len(map_tokens_to_logprobs) == 0:
            logging.warning(f"Cannot find any logprobs from {denominator_tokens + [numerator_token]} in {completion}.")
            return np.nan

        baseline_logprob = map_tokens_to_logprobs.get(numerator_token, missing)
        denominator_logprob = logsumexp([map_tokens_to_logprobs.get(t, missing) for t in denominator_tokens])

        if is_binarize:
            # in the binary case, we want to know whether the baseline token has a higher logprob than all the others
            denominator_not_numerator_tokens = [t for t in denominator_tokens if t != numerator_token]
            denominator_not_baseline_logprobs = [
                map_tokens_to_logprobs.get(t, missing) for t in denominator_not_numerator_tokens
            ]
            is_baseline_best = all([baseline_logprob > t for t in denominator_not_baseline_logprobs])
            out = 1 if is_baseline_best else 2

        else:
            out_logprob = baseline_logprob - denominator_logprob  # typecheck doesn't recognize it's a float
            probability = np.exp(out_logprob)
            # if you have probability 1 of preferring first then say 1 if 0 say 2
            # output between 1 and 2 for historical reasons
            out = 2 - probability

        return out

    if log_prob_index == "batch":
        batch_size = len(completion["logprobs"]["content"])
        assert batch_size > 0
        for i in range(batch_size):
            assert "top_logprobs" in completion["logprobs"]["content"][i]

        out = [single_logprob_parser(completion["logprobs"]["content"][i]["top_logprobs"]) for i in range(batch_size)]
    else:
        out = [single_logprob_parser(completion["logprobs"]["content"][log_prob_index]["top_logprobs"])]

    return out


def pipeline_meta_parser(
    completion: str, parsers_to_kwargs: dict[str, dict], is_squeeze: bool = True, _depth=0
) -> list[Any]:
    r"""Applies a list of parsers in sequence to a completion.

    Parameters
    ----------
    completion : str
        The completion to parse.

    parsers_to_kwargs : dictionary of str to dict
        A dictionary mapping parser functions to kwargs to pass to them. The parsing functions will be applied in the
        order they are given.

    is_squeeze : bool, optional
        If True, will squeeze the output of each parser if it's a singleton.

    Examples
    --------
    >>> completion = '{"ex": "...", "rank": [{"model": "model_1", "rank": 1}, {"model": "model_2", "rank": 2}]}'
    >>> parsers_to_kwargs = {"json_parser": {"annotation_key": "rank"}, "ranking_parser": {}}
    >>> pipeline_meta_parser(completion, parsers_to_kwargs)
    [1]
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    all_parsers = list(parsers_to_kwargs.keys())
    all_kwargs = list(parsers_to_kwargs.values())

    out = globals()[all_parsers[0]](completion, **all_kwargs[0])
    rest_of_parsers_to_kwargs = dict(zip(all_parsers[1:], all_kwargs[1:]))
    if len(rest_of_parsers_to_kwargs) > 0:
        out = [
            pipeline_meta_parser(
                o, parsers_to_kwargs=rest_of_parsers_to_kwargs, is_squeeze=is_squeeze, _depth=_depth + 1
            )
            for o in out
        ]

    if is_squeeze and len(out) == 1 and _depth != 0:
        assert isinstance(out, list)
        out = out[0]

    return out
