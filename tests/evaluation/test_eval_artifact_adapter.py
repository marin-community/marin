# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reusable Evalchemy adapter preserves its artifact identity contract."""

import json

from marin.evaluation.eval_result import EvalchemyResult
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
        evals=(EvalTaskConfig("arc_easy", 0),),
        task_group_id="core",
        serve=ServeSpec(tpu_type="v6e-4"),
        extra_gen_kwargs={"repetition_penalty": "1.1"},
        discover_latest_checkpoint=False,
        version="2026.07.24",
    )
    payload = json.loads(step.fingerprint_payload())

    assert step.name == "evaluation/evalchemy/model/core"
    assert step.version == "2026.07.24"
    assert step.artifact_type is EvalchemyResult
    assert step.deps == (model,)
    assert payload["model"] == "checkpoints/model@2026.07.24"
    assert payload["extra_gen_kwargs"] == {"repetition_penalty": "1.1"}
    assert payload["tasks"] == [
        {
            "completion_only": False,
            "generation": False,
            "name": "arc_easy",
            "num_fewshot": 0,
            "task_alias": None,
            "task_kwargs": None,
            "unsafe_code": False,
        }
    ]
