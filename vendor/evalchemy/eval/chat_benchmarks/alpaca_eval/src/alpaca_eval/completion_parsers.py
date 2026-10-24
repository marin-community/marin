"""Parse rankings and token log probabilities from AlpacaEval judge responses."""

import ast
import logging
from typing import Any, Literal, Sequence, Union

import numpy as np
from scipy.special import logsumexp

__all__ = ["ranking_parser", "logprob_parser"]


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
