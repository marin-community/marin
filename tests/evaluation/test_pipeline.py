# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.evaluation.model_config import ModelConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.evaluation.pipeline import ProducedEvaluationModel


def test_produced_evaluation_model_drops_source_revision():
    checkpoint = ArtifactStep(
        name="checkpoints/generated-model",
        version="2026.09.22",
        artifact_type=Artifact,
        run=lambda _config: None,
        build_config=lambda _ctx: {},
    )
    source = ModelConfig(
        name="generated-model",
        location="organization/base-model",
        revision="base-model-revision",
        tokenizer="organization/tokenizer",
    )

    resolved = ProducedEvaluationModel(checkpoint, source).resolve(StepContext.for_fingerprint(deps=(checkpoint,)))

    assert resolved.location == "artifact://checkpoints/generated-model@2026.09.22"
    assert resolved.revision is None
    assert resolved.tokenizer == "organization/tokenizer"
