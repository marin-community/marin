# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage dispatch contract for the coordinator."""

from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.model.stage import Exit, Stage
from forge.model.workpiece import Workpiece
from forge.resources.bundle import Resources

type StageFn = Callable[[Context, Workpiece, ForgeConfig, Resources], Awaitable[Exit]]

STAGES: Mapping[Stage, StageFn] = MappingProxyType({})
