# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed original development-panel family denominators, independent of execution backends."""

FAMILY_COUNTS = {
    "calendar": 167,
    "competitive_coding": 167,
    "reasoning_gym": 167,
    "stem": 166,
    "structural": 243,
}


def evaluation_dataset_uid(uid: str, step: int | None) -> str:
    """Read the dataset UID from the step evaluator's length-prefixed instance ID."""
    if step is None:
        return uid
    prefix = f"eval-{step}-"
    if not uid.startswith(prefix):
        raise ValueError("Periodic evaluation UID does not match the requested update")
    name_length, separator, tail = uid[len(prefix) :].partition("-")
    if not separator or not name_length.isdecimal():
        raise ValueError("Periodic evaluation UID lacks a validation-name length")
    length = int(name_length)
    if len(tail) <= length + 1 or tail[length] != "-":
        raise ValueError("Periodic evaluation UID has an invalid validation-name boundary")
    return tail[length + 1 :]
