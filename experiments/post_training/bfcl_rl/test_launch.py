# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext

from experiments.post_training.bfcl_rl import launch
from experiments.post_training.bfcl_rl.launch import recovered_model


@pytest.mark.parametrize("policy_export_version", [None, "2026.10.04.26"])
def test_recovery_handoff_requires_export_and_resolves_saved_policy(tmp_path, policy_export_version):
    # The first live DPO update saved a native checkpoint but skipped its HF hook.
    # The RL learner must fail rather than silently reverting to the starting model.
    model = recovered_model("2026.10.04.21", policy_export_version)
    root = model.step.path(str(tmp_path))
    ctx = StepContext.for_run(str(tmp_path / "rl"), str(tmp_path), deps=model.deps())
    with pytest.raises(ValueError, match="has no HF export"):
        model.resolve(ctx)

    export = Path(root) / "hf" / "step-57"
    export.mkdir(parents=True)
    (export / "config.json").write_text("{}")
    (export / "tokenizer_config.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"saved-policy")
    resolved = model.resolve(ctx)
    assert resolved.uri == str(export)
    assert Path(resolved.uri, "model.safetensors").read_bytes() == b"saved-policy"


def test_rl_recipe_survives_coordinator_workspace_relocation(tmp_path, monkeypatch):
    languages = ("python", "java", "javascript")
    images = tuple(f"registry/{language}@sha256:{index:064x}" for index, language in enumerate(languages))
    expected = yaml.safe_load(launch.rl_recipe(images, 2))
    source = Path(launch.__file__)
    copied = tmp_path / "launch.py"
    shutil.copyfile(source, copied)
    shutil.copyfile(source.with_name("v125_async.yaml"), copied.with_name("v125_async.yaml"))
    name = "experiments.post_training.bfcl_rl.packaged_launch"
    spec = importlib.util.spec_from_file_location(name, copied)
    assert spec is not None and spec.loader is not None
    relocated = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, relocated)
    spec.loader.exec_module(relocated)
    read_text = Path.read_text

    def workspace_read(path, *args, **kwargs):
        if not path.is_relative_to(tmp_path):
            raise FileNotFoundError(f"Not in the remote workspace: {path}")
        return read_text(path, *args, **kwargs)

    # The coordinator cannot access the submitting machine's configuration files.
    monkeypatch.setattr(Path, "read_text", workspace_read)
    assert yaml.safe_load(relocated.rl_recipe(images, 2)) == expected
