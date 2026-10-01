# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover archive entries without fetching the rest of their dictionary page."""

import base64
from io import BytesIO

import pytest

from experiments.post_training.task_curation_prefix_sampling import snappy_dictionary_prefix


@pytest.mark.parametrize(
    "compressed, expected",
    [
        ("DjQDAAAAb25lAwAAAHR3bw==", [b"one", b"two"]),
        # Independently encoded with cramjam.snappy.compress_raw; later dictionary
        # entries were removed from this fixture. Prefix extraction must still work.
        (
            "/KQEHIAAAABhYmNk/gQA7gQAAYQMZWZnaP4EAO4EABBwEQEAaf4BAP4BAP4BAP4BAP4=",
            [b"abcd" * 32, b"efgh" * 32],
        ),
    ],
)
def test_snappy_dictionary_prefix_decodes_entries_without_the_remaining_page(compressed, expected):
    assert snappy_dictionary_prefix(BytesIO(base64.b64decode(compressed)), len(expected)) == expected
