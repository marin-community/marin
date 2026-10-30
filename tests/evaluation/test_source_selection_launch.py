# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit source subsets retain canonical generation settings and provenance."""

import json
from pathlib import Path

import yaml
from marin.evaluation.evalchemy.client import build_command
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _run_config_json
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.evaluation.evals import EvalchemyDefinition, evalchemy_run_config


def test_selected_source_keys_reach_evalchemy_and_durable_task_ref(tmp_path):
    parent = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs/math500.yaml")
    data = yaml.safe_load(parent.read_text())
    task = data["tasks"][0]
    keys = ['{"namespace":"default","source_id":1}', '{"namespace":"default","source_id":4}']
    data.setdefault("task_options", {}).setdefault(task, {})["selected_source_keys"] = keys
    subset = tmp_path / "math500-micro.yaml"
    subset.write_text(yaml.safe_dump(data))
    full = evalchemy_run_config("math500", load_evalchemy_config(parent))
    mini = evalchemy_run_config("math500-micro", load_evalchemy_config(subset))
    model = RunningModel(OpenAIEndpoint("http://model.invalid/v1", "model"), tokenizer="org/model")
    full_json = json.loads(_run_config_json(model, full, "s3://bucket/results"))
    mini_json = json.loads(_run_config_json(model, mini, "s3://bucket/results"))
    full_command = build_command(full_json, full_json["tasks"][0], "/runtime/python", 73664)
    mini_command = build_command(mini_json, mini_json["tasks"][0], "/runtime/python", 73664)
    position = mini_command.index("--selected_source_keys")
    assert json.loads(mini_command[position + 1]) == keys
    assert mini_command[:position] + mini_command[position + 2 :] == full_command
    reference = EvalchemyDefinition("math500-micro", subset).record_ref_for(mini)
    assert list(reference.tasks[0].selected_source_keys) == keys
    assert reference.source_digest != EvalchemyDefinition("math500", parent).record_ref_for(full).source_digest
