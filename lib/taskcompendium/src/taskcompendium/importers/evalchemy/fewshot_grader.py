# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package canonical completion scorers without changing retained catalog grader identities."""

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from verifyit.spec import EmptyOutputPolicy, JudgeSpec

from taskcompendium.grader import GraderPackage
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    EnvironmentRequirements,
    FileReward,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskResource,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

SCORERS = Path(__file__).parent / "scorers"
DEPENDENCIES = {
    "gsm8k_cot": ("minerva.py", "amc.py"),
    "minerva_math": ("minerva.py", "amc.py"),
    "bbh_cot": (),
    "harness_generation": ("harness_completion.py", "harness_generation.py", "minerva.py", "amc.py"),
}


@cache
def scorer_files(scorer: str) -> tuple[TaskResource, ...]:
    paths = ["__init__.py", "fewshot.py", "lm-evaluation-harness-MIT.txt", *DEPENDENCIES[scorer]]
    return tuple(inline_resource("source_grader/" + path, (SCORERS / path).read_bytes()) for path in paths)


def scorer_revision(scorer: str) -> str:
    """Include the scorer, entry point and dependency bytes in the corpus identity."""
    digest = canonical_sha256(
        {
            "package.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **{resource.path: hashlib.sha256(resource_bytes(resource)).hexdigest() for resource in scorer_files(scorer)},
        }
    )
    return "source-script-v1-" + digest[:16]


def source_package(
    scorer: str,
    config: Mapping[str, Any],
    *,
    environment: EnvironmentRequirements,
    judge: JudgeSpec | None = None,
    timeout: float = 60,
) -> GraderPackage:
    """Package the pinned source scorer for a fresh Shellbox verifier machine."""
    grader = ScriptGrader(
        argv=("python3", "-m", "source_grader.fewshot", scorer),
        cwd="/tests",
        environment=environment,
        judge=judge,
        empty_output=EmptyOutputPolicy.GRADE,
        answer_path="/app/answer.txt",
        timeout=timeout,
        reward=FileReward(
            files=(RewardFile(path="/app/verdict.json", format=RewardFileFormat.VERDICT_JSON),),
        ),
    )
    resource = inline_resource("config.json", json.dumps(dict(config), allow_nan=False, sort_keys=True).encode())
    return GraderPackage(grader, (*scorer_files(scorer), resource))
