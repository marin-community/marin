# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical launch admission and curated campaign policy checks."""

import runpy
from pathlib import Path

import pytest
from evalrecords.eval_policy import SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION
from evalrecords.records import EvaluationSelection
from iris.rpc import job_pb2
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import LaunchProvenance

from experiments.evaluation.launch import LaunchSpec, build_evaluation_batch


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


def test_curated_campaign_metrics_match_the_publication_contract():
    checker = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/check_policy_metrics.py")
    runpy.run_path(str(checker))["main"]()
