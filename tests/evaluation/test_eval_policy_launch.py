# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published launch recipes materialize the benchmark settings they claim."""


import runpy
from pathlib import Path

import pytest
import yaml
from iris.rpc import job_pb2
from marin.evaluation.eval_policy import SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import EvaluationSelection
from marin.evaluation.runner import LaunchProvenance

from eval_policy.launch import _harbor_config, launch_policy
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
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        evals=(EvaluationSelection(benchmark="math500"),),
        model=ModelConfig(name="test-model", location="org/test-model"),
        platform=Platform.GPU,
        accelerator="H100x8",
        limit=None,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        version=version,
    )

    with pytest.raises(ValueError, match="requires its historical Evalchemy checkout"):
        build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="test"), "test")


@pytest.mark.parametrize("version", [SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION])
@pytest.mark.parametrize(
    "selection",
    [
        None,
        ("math500",),
        ("swebench-recovery", "math500"),
    ],
)
def test_policy_launcher_rejects_historical_evalchemy_before_writing_or_launching(
    tmp_path, monkeypatch, version, selection
):
    output_dir = tmp_path / "eval_policy"
    output_dir.mkdir()
    model_path = tmp_path / "model.yaml"
    model_path.write_text("name: model\nlocation: org/model\nresource_hint:\n  gpu:\n    H100: 1\n")
    launched = []
    monkeypatch.setattr("eval_policy.launch.ROOT", tmp_path)
    monkeypatch.setattr("eval_policy.launch.run", lambda command, **_kwargs: launched.append(command))

    with pytest.raises(ValueError, match=r"requires historical Evalchemy commit.*checkout"):
        launch_policy(version, model_path, None, None, "cw-rno2a", selection)

    assert not launched
    assert not list(output_dir.iterdir())


def test_curated_campaign_metrics_match_the_publication_contract():
    checker = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/check_policy_metrics.py")
    runpy.run_path(str(checker))["main"]()
