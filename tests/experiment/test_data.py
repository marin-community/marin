# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.experiment.data import tokenized

_TOKENIZER = "gpt2"
_V = "2026.06.28"


def test_tokenized_requires_exactly_one_raw_input():
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", tokenizer=_TOKENIZER, version=_V)
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", source="org/corpus", paths=["gs://b/x"], tokenizer=_TOKENIZER, version=_V)
