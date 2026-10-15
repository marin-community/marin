# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training run names and seeds from evaluation-metrics URIs; the run name gives a physical run one observation id."""

import re

# A training run's directory is its name plus the six-hex-digit hash the launcher appends.
RUN_DIRECTORY = re.compile(r".+-[0-9a-f]{6}")
# The directory that holds a run names its data seed and, except for trainer seed 0 in older launches, its trainer seed.
SEED_GROUP = re.compile(r".+_seed(?P<data_seed>\d+)(?:_t(?P<trainer_seed>\d+))?")


def training_run_name_from_uri(uri: str) -> str:
    """The training run name in an evaluation-metrics URI: the one path component that ends in the run's hash.

    `gs://.../<run>-8cf646/checkpoints/eval_metrics.jsonl` gives `<run>-8cf646`. Raises ValueError when no component,
    or more than one, ends in a hash.
    """
    names = [part for part in uri.split("/") if RUN_DIRECTORY.fullmatch(part)]
    if len(names) != 1:
        raise ValueError(f"expected one hashed run directory in {uri!r}, found {names}")
    return names[0]


def seeds_from_uri(uri: str) -> tuple[int, int]:
    """Data and trainer seed from the group directory that holds the run in an evaluation-metrics URI.

    `gs://.../table9_seed662009_t1/<run>-8cf646/checkpoints/eval_metrics.jsonl` gives (662009, 1); a group without
    `_t<seed>` holds trainer seed 0. Raises ValueError when the run's parent directory names no data seed.
    """
    parts = uri.split("/")
    group = parts[parts.index(training_run_name_from_uri(uri)) - 1]
    match = SEED_GROUP.fullmatch(group)
    if match is None:
        raise ValueError(f"no seed group directory above the run in {uri!r}")
    return int(match["data_seed"]), int(match["trainer_seed"] or 0)
