# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.rl.cli import _coordinator_request

from experiments.post_training.bfcl_rl import launch
from experiments.post_training.bfcl_rl.launch import recovered_model
from experiments.post_training.bfcl_rl.offline_preferences import native_preference_step


def test_native_preference_graph_submits_cpu_coordinator_to_coreweave(tmp_path):
    teachers = (("s3://bucket/teacher/2026.10.04.34", 7), ("s3://bucket/teacher/2026.10.05.52", 11))
    with build_context(BuildContext(VersionCodex("2026.10.05.55"))):
        step = native_preference_step(
            teachers, "s3://bucket/student/2026.10.05.49", "pinned-teacher", 7, "2026.10.04.21", "2026.10.04.26", 57
        )
    module = "experiments.post_training.bfcl_rl.offline_preferences"
    cluster, request = _coordinator_request([step], module, ("--version", "2026.10.05.55", "--run"), 1, tmp_path, {})
    assert cluster == "lib/iris/config/marin.yaml"
    assert request.resources.target_cluster == "cw-rno2a"
    assert request.resources.chip_count() == 0
    assert request.entrypoint.binary_entrypoint is not None
    assert request.entrypoint.binary_entrypoint.args == [
        "-m",
        module,
        "--version",
        "2026.10.05.55",
        "--run",
        "--max-concurrent",
        "1",
    ]


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
    expected = yaml.safe_load(launch.rl_recipe(images, 2, 8, "stream"))
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
    assert yaml.safe_load(relocated.rl_recipe(images, 2, 8, "stream")) == expected
