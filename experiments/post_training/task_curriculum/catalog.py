# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the canonical task curriculum catalog."""

from pathlib import Path

import yaml

from experiments.post_training.task_curriculum.models import CurriculumCatalog


def catalog_from_yaml(contents: bytes) -> CurriculumCatalog:
    """Validate a serialized curriculum catalog."""
    return CurriculumCatalog.model_validate(yaml.safe_load(contents))


def load_catalog(path: Path) -> CurriculumCatalog:
    return catalog_from_yaml(path.read_bytes())
