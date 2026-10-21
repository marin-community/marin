import functools

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def load_tokenizer():
    """Return a loader that reads each pretrained tokenizer once per session.

    Loading a tokenizer costs up to a second, and many tests render the same few templates. Callers must not
    mutate the returned tokenizer; load a private copy with `AutoTokenizer.from_pretrained` for that.
    """
    return functools.cache(AutoTokenizer.from_pretrained)
