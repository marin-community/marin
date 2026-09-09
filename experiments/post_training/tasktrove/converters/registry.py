# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter registry keyed by template id.

A converter takes the decoded task files and returns a :class:`ConvertedTask`. It never inspects
``instruction.md`` beyond passing it through; everything it needs is in the per-task data files
the template fingerprint identified. Template ids come from ``templates.json`` produced by the
fingerprint step; a task whose template has no converter is reported, never guessed at.
"""

from experiments.post_training.tasktrove.converters import nemotron_gym
from experiments.post_training.tasktrove.converters.converted_task import Converter

CONVERTERS: dict[str, Converter] = {**nemotron_gym.CONVERTERS}
