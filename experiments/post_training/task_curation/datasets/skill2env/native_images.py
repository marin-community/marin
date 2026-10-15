# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build native actor and workspace-copy verifier images without importing the Harbor SDK."""

from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.post_training.task_curation.images.push import push_image


def build_image(root: Path, task_ref: str, repository: str) -> tuple[str, str]:
    """Build the original context and a verifier image that accepts the submitted workspace."""
    tag = f"{repository}:skill2env-{task_ref.rsplit(':', 1)[1]}"
    actor = f"{repository}@{push_image(root / 'environment', tag)}"
    # A pristine workspace would resurrect files deleted by the submitted solution.
    with TemporaryDirectory(prefix="skill2env-verifier-build-") as directory:
        context = Path(directory)
        (context / "Dockerfile").write_text(
            f"FROM {actor}\nUSER root\nRUN rm -rf /workspace && mkdir -p /workspace\nWORKDIR /workspace\n"
        )
        grader = f"{repository}@{push_image(context, tag + '-verifier')}"
    return actor, grader
