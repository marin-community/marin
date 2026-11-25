# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild registered curriculum identities from their published pool manifests."""

from collections.abc import Callable, Mapping

import draccus
from marin.rl.curriculum import CurriculumValidation, RlCurriculum, ValidationRow

from experiments.post_training.task_curation.datasets.long_loop_rl import TEACHER_ROUTE_LABELS
from experiments.post_training.task_curation.exp3_curriculum import exp3_curriculum_from_validation
from experiments.post_training.task_curation.exp4_curriculum import (
    NEMOTRON_IF_LABEL,
    NEMOTRON_SOURCE,
    exp4_rl_curriculum,
)
from experiments.post_training.task_curation.exp5_curriculum import (
    HARDMATH_LABEL,
    HARDMATH_REPO,
    IFBENCH_LABEL,
    IFBENCH_SOURCE,
    OPENRESEARCHER_LABEL,
    OPENRESEARCHER_SOURCE,
    exp5_rl_curriculum,
)

_MANIFEST_BUILDERS: dict[str, Callable[[CurriculumValidation], RlCurriculum]] = {
    "long-loop/exp3-rl": exp3_curriculum_from_validation,
    "long-loop/exp4-rl": exp4_rl_curriculum,
    "long-loop/exp5-rl": exp5_rl_curriculum,
}


def curriculum_from_manifest(manifest: Mapping) -> RlCurriculum:
    """Rebuild a known curriculum and refuse any mismatch with its recorded identity."""
    identity = manifest["curriculum"]
    name = identity.partition("@")[0]
    if "curriculum_definition" in manifest:
        curriculum = draccus.decode(RlCurriculum, dict(manifest["curriculum_definition"]))
        if curriculum.identity != identity:
            raise ValueError("Recorded curriculum definition differs from the manifest identity")
        if curriculum.validation != CurriculumValidation(
            size=manifest["validation"]["size"],
            seed=manifest["validation"]["seed"],
            rows=tuple(ValidationRow(**row) for row in manifest["validation"]["rows"]),
        ):
            raise ValueError("Recorded curriculum definition differs from the manifest holdout")
        return curriculum
    if name not in _MANIFEST_BUILDERS:
        raise ValueError(f"Unregistered pool curriculum: {identity}")
    payload = manifest["validation"]
    validation = CurriculumValidation(
        size=payload["size"], seed=payload["seed"], rows=tuple(ValidationRow(**row) for row in payload["rows"])
    )
    curriculum = _MANIFEST_BUILDERS[name](validation)
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
