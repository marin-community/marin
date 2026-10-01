# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation code and task-content identities."""

import hashlib
import inspect
from pathlib import Path

import verifyit

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import ResourceVisibility, TaskSpec
from taskcompendium.pipeline.models import DatasetRecipe


def code_digest(recipe: DatasetRecipe) -> str:
    """Fingerprint the semantic package, shared scorers, and recipe module."""
    roots = (Path(__file__).parents[1], Path(verifyit.__file__).parent)
    digest = hashlib.sha256()
    for root in roots:
        for file in sorted(root.rglob("*.py")):
            digest.update(str(file.relative_to(root)).encode())
            digest.update(file.read_bytes())
    module_path = inspect.getsourcefile(recipe.normalize)
    if module_path is None:
        raise ValueError("A recipe normalizer must be defined in a Python source file")
    digest.update(Path(module_path).read_bytes())
    return digest.hexdigest()


def semantic_digest(task: TaskSpec, include_reference: bool) -> str:
    """Hash public task semantics, optionally including its private reference."""
    content = task.model_dump(mode="json", exclude={"id", "source"})
    # Control scripts are executable witnesses, not task semantics.
    content["resources"] = [
        resource for resource in content["resources"] if resource["visibility"] != ResourceVisibility.CONTROL
    ]
    if not include_reference:
        content.pop("verifier")
        content["resources"] = [
            resource for resource in content["resources"] if resource["visibility"] == ResourceVisibility.AGENT
        ]
    return canonical_sha256(content)
