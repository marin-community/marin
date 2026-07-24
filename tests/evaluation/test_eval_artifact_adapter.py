# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy pipeline steps preserve their cache identity."""

import json

from marin.evaluation.evalchemy import EvalchemyRunConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.serving_config import ServeSpec
from marin.execution.lazy import ArtifactStep
from marin.experiment.evaluation import evaluate_evalchemy
from marin.training.training import LevanterCheckpoint


def test_evalchemy_adapter_identity_and_fingerprint_payload():
    model = ArtifactStep(
        name="checkpoints/model",
        version="2026.07.24",
        artifact_type=LevanterCheckpoint,
        run=lambda config: None,
        build_config=lambda ctx: {},
    )

    step = evaluate_evalchemy(
        model_name="model",
        model=model,
        config=EvalchemyRunConfig(
            name="core",
            tasks=(EvalTaskConfig("arc_easy", 0),),
            extra_gen_kwargs={"repetition_penalty": "1.1"},
        ),
        serve=ServeSpec(tpu_type="v6e-4"),
        discover_latest_checkpoint=False,
        version="2026.07.24",
    )
    payload = json.loads(step.fingerprint_payload())

    assert step.name == "evaluation/evalchemy/model/core"
    assert step.version == "2026.07.24"
    assert payload["model"] == "checkpoints/model@2026.07.24"
    assert payload["run"]["extra_gen_kwargs"] == {"repetition_penalty": "1.1"}
