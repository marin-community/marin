# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild registered curriculum identities from their published pool manifests."""

from collections.abc import Callable, Mapping

from marin.rl.curriculum import CurriculumValidation, RlCurriculum

from experiments.post_training.task_curation.datasets.long_loop_rl import TEACHER_ROUTE_LABELS
from experiments.post_training.task_curation.exp4_curriculum import (
    NEMOTRON_IF_LABEL,
    NEMOTRON_SOURCE,
    exp4_rl_curriculum,
    exp4_validation_from_manifest,
)
from experiments.post_training.task_curation.exp5_curriculum import (
    HARDMATH_LABEL,
    HARDMATH_REPO,
    IFBENCH_LABEL,
    IFBENCH_SOURCE,
    OPENRESEARCHER_LABEL,
    OPENRESEARCHER_SOURCE,
    exp5_rl_curriculum,
    exp5_validation_from_manifest,
)

_MANIFEST_BUILDERS: dict[
    str, tuple[Callable[[Mapping], CurriculumValidation], Callable[[CurriculumValidation], RlCurriculum]]
] = {
    "long-loop/exp4-rl": (exp4_validation_from_manifest, exp4_rl_curriculum),
    "long-loop/exp5-rl": (exp5_validation_from_manifest, exp5_rl_curriculum),
}


def curriculum_from_manifest(manifest: Mapping) -> RlCurriculum:
    """Rebuild a known curriculum and refuse any mismatch with its recorded identity."""
    identity = manifest["curriculum"]
    name = identity.partition("@")[0]
    if name not in _MANIFEST_BUILDERS:
        raise ValueError(f"Unregistered pool curriculum: {identity}")
    read_validation, build = _MANIFEST_BUILDERS[name]
    curriculum = build(read_validation(manifest["validation"]))
    if curriculum.identity != identity:
        raise ValueError(f"Rebuilt curriculum {curriculum.identity} differs from pool manifest {identity}")
    return curriculum


def rollout_source_routes(teacher_routes: Mapping[str, str]) -> dict[str, str]:
    """Classify stamped rows by their unchanged TaskSpec source dataset names."""
    routes = dict(teacher_routes)
    aliases = {
        **TEACHER_ROUTE_LABELS,
        NEMOTRON_SOURCE: NEMOTRON_IF_LABEL,
        IFBENCH_SOURCE: IFBENCH_LABEL,
        OPENRESEARCHER_SOURCE: OPENRESEARCHER_LABEL,
        HARDMATH_REPO: HARDMATH_LABEL,
    }
    for source, teacher in aliases.items():
        if teacher in teacher_routes:
            route = teacher_routes[teacher]
            if source in routes and routes[source] != route:
                raise ValueError(f"Conflicting rollout routes for {source} and {teacher}")
            routes[source] = route
    return routes
