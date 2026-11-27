# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned format repair for the retained Polaris source; not an admission receipt."""

import hashlib
import inspect
import json
from pathlib import Path

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from taskcompendium.convert.script_grader import GRADE_ARGV
from taskcompendium.models import PlainText, ScriptGrader, TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.modes import extract
from verifyit.reasoning_markup import REASONING_DELIMITERS
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import EnvironmentArtifact, identity_digest

POLICY = "polaris-final-text-v3"
VERIFYIT_REVISION = "ef09a1e250fdada00c2420c12534d452735a7c8a"
GRADER_ENVIRONMENT = Environment(
    requirements=(
        "mpmath==1.3.0",
        "pylatexenc==2.10",
        "sympy==1.13.3",
        "tomli-w==1.2.0",
        "math-verify==0.9.0",
        "latex2sympy2-extended==1.11.0",
        f"marin-verifyit @ https://github.com/marin-community/marin/archive/{VERIFYIT_REVISION}.tar.gz"
        "#subdirectory=lib/verifyit",
    )
)
UPSTREAM_REVISION = "296f8e34132e63f4a1d70e0dcc8bddebb43f03e4"
LEGACY_GRADER_SHA256 = "9236329bd606f45ccd6951557d1440fa3f6568d6b716ca7380d77542228a1624"
LEGACY_MATH_SHA256 = "9bdc85ecb9c9255c387c04df5ae2d9a858a44d169418f05c262b78d2c33c963b"
LEGACY_PACKAGES_LOCK = (
    "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/polaris/2026.11.16.1/"
    "artifacts/images/env-8300a4bdb429d567/2026.10.08/requirements.lock"
)
GRADE_PATH = Path(__file__).with_name("resources") / "polaris_grade.py"
EXTRACTION_PATH = "polaris_answer_extract.py"
FINAL_TEXT_PATH = "polaris_final_text.py"
LINEAGE_PATH = "reference/polaris-final-answer-policy.json"


def grader_files() -> dict[str, bytes]:
    """Return the exact guest code included in both artifact identity and TaskSpecs."""
    # Ship the existing transport normalizer without importing the SkyRL runtime in the guest.
    final_text = (
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n"
        f"import re\n\nREASONING_DELIMITERS = {REASONING_DELIMITERS!r}\n\n" + inspect.getsource(final_answer_text)
    ).encode()
    return {
        "grade.py": GRADE_PATH.read_bytes(),
        EXTRACTION_PATH: Path(extract.__file__).read_bytes(),
        FINAL_TEXT_PATH: final_text,
    }


def with_final_text_grader(task: TaskSpec, built: EnvironmentArtifact) -> TaskSpec:
    """Replace the legacy grader, preserving its original resources and requirements.

    The result needs a new source release and grading qualification before admission.
    It must never overwrite the retained source or a running pool.
    """
    if built.identity != identity_digest(GRADER_ENVIRONMENT):
        raise ValueError("The Polaris repair requires its built pinned grader environment")
    if task.source.dataset != "POLARIS-Project/Polaris-Dataset-53K" or task.source.revision != UPSTREAM_REVISION:
        raise ValueError("The Polaris repair requires the pinned upstream source")
    if (
        not isinstance(task.answer_format, PlainText)
        or not isinstance(task.grader, ScriptGrader)
        or task.grader.argv != GRADE_ARGV
        or task.grader.answer_path != DEFAULT_OUTPUT
    ):
        raise ValueError("The Polaris repair requires the legacy final-text script contract")
    resources = {item.path: item for item in task.resources.verifier}
    for name, expected in (("grade.py", LEGACY_GRADER_SHA256), ("polaris_math.py", LEGACY_MATH_SHA256)):
        if name not in resources or hashlib.sha256(resource_bytes(resources[name])).hexdigest() != expected:
            raise ValueError(f"The Polaris repair refuses a different {name}")
    if any(path in resources for path in (EXTRACTION_PATH, FINAL_TEXT_PATH, LINEAGE_PATH)):
        raise ValueError("The Polaris task already carries a format repair")
    if task.grader.environment.packages_lock != LEGACY_PACKAGES_LOCK:
        raise ValueError("The Polaris repair requires the retained grader package lock")

    files = grader_files()
    grade, extraction, final_text = (files[path] for path in ("grade.py", EXTRACTION_PATH, FINAL_TEXT_PATH))
    lineage = {
        "policy": POLICY,
        "previous_grader_sha256": LEGACY_GRADER_SHA256,
        "math_checker_sha256": LEGACY_MATH_SHA256,
        "grader_sha256": hashlib.sha256(grade).hexdigest(),
        "extraction_sha256": hashlib.sha256(extraction).hexdigest(),
        "final_text_sha256": hashlib.sha256(final_text).hexdigest(),
        "grader_environment_identity": built.identity,
        "grader_packages_lock": built.lock_url,
        "verifyit_revision": VERIFYIT_REVISION,
    }
    verifier = (
        *(inline_resource("grade.py", grade) if item.path == "grade.py" else item for item in task.resources.verifier),
        inline_resource(EXTRACTION_PATH, extraction),
        inline_resource(FINAL_TEXT_PATH, final_text),
        inline_resource(LINEAGE_PATH, json.dumps(lineage, sort_keys=True).encode()),
    )
    grader = task.grader.model_copy(
        update={"environment": task.grader.environment.model_copy(update={"packages_lock": built.lock_url})}
    )
    return task.model_copy(
        update={"resources": task.resources.model_copy(update={"verifier": verifier}), "grader": grader}
    )
