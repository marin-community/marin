# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every converter, indexed by :class:`ConverterKey`.

A converter reads the per-task data files its template ships, maps them onto one verifier mode,
edits the task's Dockerfile, and sets the selection tags. It never guesses: a task whose key has
no converter is reported as ``no_converter``, and a task the template cannot grade soundly comes
back as a typed :class:`Rejected`.
"""

from experiments.post_training.tasktrove.converters import (
    agent_calendar,
    all_puzzles,
    code_contests,
    codeforces,
    judge_rubric,
    nemotron_competitive,
    nemotron_gym,
    nemotron_if_structured,
    nemotron_ifeval,
    nemotron_multichallenge,
    nemotron_openqa,
    nemotron_reasoning,
    nemotron_structured_outputs,
    nl2bash,
    prompt_injection,
    swe_patched,
    swe_trusted_paths,
    taco,
)
from experiments.post_training.tasktrove.converters.converted_task import Converter, ConverterKey

CONVERTERS: tuple[Converter, ...] = (
    *nemotron_gym.CONVERTERS,
    agent_calendar.CONVERTER,
    all_puzzles.CONVERTER,
    code_contests.CONVERTER,
    codeforces.CONVERTER,
    judge_rubric.CONVERTER,
    nemotron_competitive.CONVERTER,
    nemotron_if_structured.CONVERTER,
    nemotron_ifeval.CONVERTER,
    nemotron_multichallenge.CONVERTER,
    nemotron_openqa.CONVERTER,
    nemotron_reasoning.CONVERTER,
    nemotron_structured_outputs.CONVERTER,
    nl2bash.CONVERTER,
    prompt_injection.CONVERTER,
    swe_patched.CONVERTER,
    swe_trusted_paths.CONVERTER,
    taco.CONVERTER,
)


def converter_index() -> dict[ConverterKey, Converter]:
    index: dict[ConverterKey, Converter] = {}
    for converter in CONVERTERS:
        for key in converter.keys:
            if key in index:
                raise ValueError(f"{key} is claimed by both {index[key].name} and {converter.name}")
            index[key] = converter
    return index
