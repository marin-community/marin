# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training run names, the key that gives a physical run one observation id in every table."""

import re

# A training run's directory is its name plus the six-hex-digit hash the launcher appends.
RUN_DIRECTORY = re.compile(r".+-[0-9a-f]{6}")


def training_run_name_from_uri(uri: str) -> str:
    """The training run name in an evaluation-metrics URI: the one path component that ends in the run's hash.

    `gs://.../<run>-8cf646/checkpoints/eval_metrics.jsonl` gives `<run>-8cf646`. Raises ValueError when no component,
    or more than one, ends in a hash.
    """
    names = [part for part in uri.split("/") if RUN_DIRECTORY.fullmatch(part)]
    if len(names) != 1:
        raise ValueError(f"expected one hashed run directory in {uri!r}, found {names}")
    return names[0]
