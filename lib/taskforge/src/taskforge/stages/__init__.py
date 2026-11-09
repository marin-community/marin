# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage dispatch contract for the coordinator."""

from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.model.stage import Exit, Stage
from taskforge.model.workpiece import Workpiece
from taskforge.resources.bundle import Resources
from taskforge.stages.author import author
from taskforge.stages.review import review
from taskforge.stages.triage import triage
from taskforge.stages.validate import validate

type StageFn = Callable[[Context, Workpiece, TaskforgeConfig, Resources], Awaitable[Exit]]

STAGES: Mapping[Stage, StageFn] = MappingProxyType(
    {
        Stage.AUTHOR: author,
        Stage.VALIDATE: validate,
        Stage.REVIEW: review,
        Stage.TRIAGE: triage,
    }
)
