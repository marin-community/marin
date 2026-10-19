# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published launch recipes materialize the benchmark settings they claim."""

from dataclasses import replace

import pytest
import yaml
from iris.rpc import job_pb2
from marin.evaluation.eval_policy import SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION
from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import LaunchProvenance

from eval_policy.launch import _evalchemy_config, _harbor_config
from experiments.evaluation.evals import EvalchemyDefinition
from experiments.evaluation.launch import LaunchSpec, build_evaluation_batch


def test_september_16_uses_its_original_shots_and_context(tmp_path):
    mbpp = yaml.safe_load(_evalchemy_config(SEPTEMBER_16_VERSION, "mbppplus", None, tmp_path / "mbpp.yaml").read_text())
    crux = yaml.safe_load(_evalchemy_config(SEPTEMBER_16_VERSION, "cruxeval", None, tmp_path / "crux.yaml").read_text())
    ifbench = yaml.safe_load(_evalchemy_config(SEPTEMBER_16_VERSION, "ifbench", None, tmp_path / "if.yaml").read_text())

    assert mbpp["task_options"]["MBPPPlus"]["num_fewshot"] == 3
    assert crux["task_options"]["CruxEval"]["num_fewshot"] == 1
    assert ifbench["max_tokens"] == 1024


def test_september_24_sets_thinking_per_benchmark(tmp_path):
    math = yaml.safe_load(
        _evalchemy_config(SEPTEMBER_24_VERSION, "math500", tmp_path, tmp_path / "math.yaml").read_text()
    )
    mbpp = yaml.safe_load(
        _evalchemy_config(SEPTEMBER_24_VERSION, "mbppplus", tmp_path, tmp_path / "mbpp.yaml").read_text()
    )
    aime = yaml.safe_load(
        _evalchemy_config(SEPTEMBER_24_VERSION, "aime24", tmp_path, tmp_path / "aime.yaml").read_text()
    )

    assert math["chat_template_kwargs"] == {"enable_thinking": True}
    assert mbpp["chat_template_kwargs"] == {"enable_thinking": False}
    assert "seed" not in aime


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
def test_verified_launch_requires_its_historical_evaluator_checkout(tmp_path, monkeypatch, version):
    path = _evalchemy_config(SEPTEMBER_24_VERSION, "math500", None, tmp_path / "math500.yaml")
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
