# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from taskcompendium.pipeline.curriculum.alignment import CurriculumAnchor, CurriculumVectors

from experiments.post_training.task_curation.harrier_parity import measure_parity


def curriculum() -> CurriculumVectors:
    anchors = tuple(
        CurriculumAnchor(
            kind=kind,
            subject_id=subject,
            capability_id=subject if kind == "capability" else None,
            facet="subject_domain",
            text=f"{subject} {kind}",
        )
        for subject in ("a", "b")
        for kind in ("subject", "capability")
    )
    return CurriculumVectors(anchors, np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32))


def test_parity_accepts_small_roundoff_with_identical_areas():
    reference = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    candidate = np.asarray([[1, 0.001], [0.001, 1]], dtype=np.float32)
    measured = measure_parity(reference, candidate, curriculum())
    measured.require_accepted()
    assert measured.area_agreement == 1


def test_parity_rejects_area_flip_despite_nearly_identical_vectors():
    reference = np.asarray([[1, 0.999]], dtype=np.float32)
    candidate = np.asarray([[0.999, 1]], dtype=np.float32)
    measured = measure_parity(reference, candidate, curriculum())
    assert measured.min_cosine > 0.999
    with pytest.raises(ValueError, match="cross-device parity failed"):
        measured.require_accepted()


def test_parity_rejects_vector_drift_even_when_area_is_unchanged():
    reference = np.asarray([[1, 0]], dtype=np.float32)
    candidate = np.asarray([[1, 0.1]], dtype=np.float32)
    measured = measure_parity(reference, candidate, curriculum())
    assert measured.area_agreement == 1
    with pytest.raises(ValueError, match="cross-device parity failed"):
        measured.require_accepted()


@pytest.mark.parametrize("flipped,accepted", [(10, True), (11, False)])
def test_area_agreement_acceptance_boundary_for_2048_keys(flipped, accepted):
    reference = np.tile(np.asarray([[1, 0.999]], dtype=np.float32), (2048, 1))
    candidate = reference.copy()
    candidate[:flipped] = [0.999, 1]
    measured = measure_parity(reference, candidate, curriculum())
    if accepted:
        measured.require_accepted()
    else:
        with pytest.raises(ValueError, match="cross-device parity failed"):
            measured.require_accepted()
