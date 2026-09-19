# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the canonical task curriculum catalog."""

from pathlib import Path

import yaml

from experiments.post_training.task_curriculum.models import CurriculumCatalog


def load_catalog(path: Path) -> CurriculumCatalog:
    return CurriculumCatalog.model_validate(yaml.safe_load(path.read_text()))
