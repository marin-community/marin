# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published launch recipes materialize the benchmark settings they claim."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from iris.rpc import job_pb2
from marin.evaluation.eval_policy import SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION
from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import LaunchProvenance

from eval_policy.launch import _harbor_config, launch_policy
from experiments.evaluation.evals import EvalchemyDefinition
from experiments.evaluation.launch import LaunchSpec, build_evaluation_batch


def test_september_16_mini_datasets_use_pinned_registry(tmp_path):
    config = _harbor_config(SEPTEMBER_16_VERSION, "ds-1000-local", None, None, tmp_path / "ds.yaml")
    data = yaml.safe_load(config.read_text())

    assert data["datasets"] == [
        {
            "name": "ds-1000",
            "version": "mini-200",
            "registry_url": (
                "https://raw.githubusercontent.com/marin-community/harbor/"
                "7b18505a56e5624f55887e3b20f4de452f698a7a/registry.json"
            ),
        }
    ]
    assert data["environment"]["force_build"] is True


@pytest.mark.parametrize("version", [SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION])
def test_verified_launch_requires_its_historical_evaluator_checkout(monkeypatch, version):
    path = Path(__file__).resolve().parents[2] / "experiments/evaluation/configs/evalchemy/math500.yaml"
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=ModelConfig(name="test-model", location="org/test-model"),
        evals=(),
        evalchemy_definitions=(EvalchemyDefinition(name="math500", config_path=path),),
        harbor_definitions=(),
        platform=Platform.GPU,
        accelerator="H100x8",
        limit=None,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        version=version,
    )

    with pytest.raises(ValueError, match="requires historical Evalchemy commit"):
        build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="test"), "test")

    ad_hoc = build_evaluation_batch(
        replace(spec, version=None), LaunchProvenance(git_sha="abc", launch_host="test"), "test"
    ).evaluations[0]
    assert isinstance(ad_hoc.executor, EvalchemyExecutor)
    assert ad_hoc.identity.eval_runtime.startswith("evalchemy[")
    assert "vendor/evalchemy#sha256=" in ad_hoc.identity.eval_runtime


@pytest.mark.parametrize("version", [SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION])
@pytest.mark.parametrize(
    ("selection", "evaluator"),
    [
        (None, "Evalchemy"),
        (("math500",), "Evalchemy"),
        (("swebench-recovery", "math500"), "Evalchemy"),
        (("swebench-recovery", "ot-tblite-recovery"), "Harbor"),
    ],
)
def test_policy_launcher_rejects_historical_evaluators_before_writing_or_launching(
    tmp_path, monkeypatch, version, selection, evaluator
):
    output_dir = tmp_path / "eval_policy"
    output_dir.mkdir()
    model_path = tmp_path / "model.yaml"
    model_path.write_text("name: model\nlocation: org/model\nresource_hint:\n  gpu:\n    H100: 1\n")
    launched = []
    monkeypatch.setattr("eval_policy.launch.ROOT", tmp_path)
    monkeypatch.setattr("eval_policy.launch.run", lambda command, **_kwargs: launched.append(command))

    with pytest.raises(ValueError, match=rf"requires historical {evaluator} commit.*checkout"):
        launch_policy(version, model_path, None, None, "cw-rno2a", selection)

    assert not launched
    assert not list(output_dir.iterdir())
