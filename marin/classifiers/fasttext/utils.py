"""
utils.py

Utility functions for training fastText models.
"""

import regex as re


def preprocess(text: str) -> str:
    """
    Preprocesses text for fastText training by stripping newline characters.
    """
    return re.sub(r"[\n\r]", " ", text)


def format_example(data: dict) -> str:
    """
    Converts example to fastText training data format.
    """
    return f'{data["label"]}' + " " + preprocess(data["text"])
