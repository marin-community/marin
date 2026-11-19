# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stored teacher top-K next-token distributions carried beside packed LM rows."""

from dataclasses import dataclass

import jax
from jax.tree_util import register_dataclass
from levanter.data.text.examples import GrugLmExample


@register_dataclass
@dataclass(frozen=True)
class TeacherTopK:
    """A teacher's top-K distribution for the token after each position.

    ``ids`` and ``logprobs`` have shape ``[..., S, K]``; row ``i`` scores token ``i + 1``, as
    ``loss_weight[i]`` does. ``mask`` has shape ``[..., S]`` and is 1 where the position carries
    teacher targets: a teacher-cache row at a position with positive loss weight.
    """

    ids: jax.Array
    logprobs: jax.Array
    mask: jax.Array


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class TeacherLmExample(GrugLmExample):
    """A packed LM row with teacher targets; rows without a teacher carry a zero mask."""

    teacher: TeacherTopK
