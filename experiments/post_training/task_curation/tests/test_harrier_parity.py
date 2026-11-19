# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.alignment import CurriculumAnchor, CurriculumVectors
from taskcompendium.pipeline.curriculum.models import EmbeddingSpec

from experiments.post_training.task_curation.harrier_parity import accepted_cuda_parity, measure_parity


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


@pytest.mark.parametrize("changed", [None, "accepted", "embedding_spec_sha256", "gpu", "torch", "cuda"])
def test_cuda_acceptance_requires_the_measured_specification_and_runtime(tmp_path, monkeypatch, changed):
    spec = EmbeddingSpec(
        model="fixture", revision="1", tokenizer_revision="1", dimensions=2, max_tokens=16, compute_dtype="bfloat16"
    )
    receipt = {
        "accepted": True,
        "embedding_spec_sha256": canonical_sha256(spec.model_dump(mode="json")),
        "thresholds": {"min_vector_cosine": 0.999, "min_area_agreement": 0.995},
        "measurement": {
            "rows": 2048,
            "min_cosine": 1.0,
            "p01_cosine": 1.0,
            "mean_cosine": 1.0,
            "area_agreement": 1.0,
            "facet_agreement": {"subject_domain": 1.0},
        },
        "gpu": "accepted GPU",
        "torch": "accepted Torch",
        "cuda": "accepted CUDA",
    }
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="accepted Torch",
            version=SimpleNamespace(cuda="accepted CUDA"),
            cuda=SimpleNamespace(get_device_name=lambda: "accepted GPU"),
        ),
    )
    if changed is not None:
        receipt[changed] = False if changed == "accepted" else "different"
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(receipt))
    if changed is not None:
        with pytest.raises(ValueError):
            accepted_cuda_parity(str(path), spec)
        return
    accepted = accepted_cuda_parity(str(path), spec)
    assert accepted.uri == str(path)
    assert accepted.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
