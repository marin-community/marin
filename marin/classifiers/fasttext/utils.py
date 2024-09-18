"""
utils.py

Utility functions for training fastText models.
"""


def preprocess(text: str) -> str:
    """
    Preprocesses text for fastText training by stripping newline characters.
    """
    return text.replace("\n", " ")


def format_example(data: dict) -> str:
    """
    Converts example to fastText training data format.
    """

    return f' __label__{data["label"]}' + " " + preprocess(data["text"]) + "\n"
