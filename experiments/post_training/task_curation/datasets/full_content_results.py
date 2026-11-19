# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reviewed transport metadata for final whole-source content Results."""

from pathlib import Path

from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.reference_data import load_referenced_document

DERIVED_INDEX = Path(__file__).with_name("derived") / "index.json"


def full_content_result_bindings() -> dict[str, FullContentResultBinding]:
    """Read metadata staged by offline receipt verification, without remote data access."""
    payload = load_referenced_document(DERIVED_INDEX, "full_content_results.json")
    bindings = {key: FullContentResultBinding.model_validate(value) for key, value in payload.items()}
    if any(key != value.source_id for key, value in bindings.items()):
        raise ValueError("Final content binding key differs from its source identity")
    return bindings
