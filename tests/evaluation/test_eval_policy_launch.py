# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published launch recipes materialize the benchmark settings they claim."""

import pytest
import yaml
from iris.rpc import job_pb2
from marin.evaluation.eval_policy import SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import LaunchProvenance

from eval_policy.launch import _evalchemy_config, _harbor_config, launch_policy
from experiments.evaluation.cli import cli
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


def test_verified_launch_rejects_changed_source_before_contacting_iris(tmp_path, monkeypatch):
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
        version=SEPTEMBER_24_VERSION,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="test"), "test")
    assert batch.evaluations[0].identity.eval_ref.name == "math500"

    changed = yaml.safe_load(path.read_text())
    changed["batch_size"] = 8
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="source config differs"):
        build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="test"), "test")


def test_subset_requests_separate_nonblocking_h100_launches(tmp_path, monkeypatch):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("name: model\nlocation: org/model\n")
    submissions = []

    def capture(command, **_kwargs):
        context = cli.commands["launch"].make_context("launch", command[6:])
        submissions.append(context.params)

    monkeypatch.setattr("eval_policy.launch.subprocess.run", capture)

    launch_policy(SEPTEMBER_24_VERSION, model_path, None, None, "cw-rno2a", ("math500", "gsm8k-0shot"))

    assert len(submissions) == 2
    assert {submission["evalchemy_config"][0].stem for submission in submissions} == {"math500", "gsm8k-0shot"}
    assert all(submission["no_wait"] for submission in submissions)
    assert all(submission["accelerator"] == "H100x8" for submission in submissions)
    assert all(submission["version"] == SEPTEMBER_24_VERSION for submission in submissions)
