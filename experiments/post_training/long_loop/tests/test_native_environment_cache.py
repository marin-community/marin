# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from marin.inference.vllm_cache import VllmCompilationCache, VllmCompileIdentity
from marin.rl.skyrl import skyrl_cpu_command

from experiments.post_training.long_loop.native_environment_cache import CACHE_ROOT_ENV


@pytest.mark.timeout(180)
def test_skyrl_cache_entrypoint_delegates_to_marin_and_restores_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "remote"))
    template = tmp_path / "template.jinja"
    template.write_text("{{ messages }}")
    argv = ["serve", str(tmp_path / "model"), "--chat-template", str(template)]
    cache = VllmCompilationCache.prepare(
        launcher_identity="cpu-cache-boundary-test",
        compile_identity=VllmCompileIdentity.from_vllm_args(model_name_or_path=argv[1], extra_cli_args=tuple(argv[2:])),
        environment=dict(os.environ),
    )
    expected = Path(cache.environment()[CACHE_ROOT_ENV])
    try:
        expected.mkdir()
        (expected / "fixture.bin").write_bytes(b"small compiler cache fixture")
        cache.publish()
        assert len(list((tmp_path / "remote").rglob("latest.json"))) == 1
    finally:
        cache.close()
    profile = {
        "venv_alias": str(tmp_path / "native/venv"),
        "native_cli": argv,
        "cache_launcher_identity": "cpu-cache-boundary-test",
        "values": {CACHE_ROOT_ENV: str(expected)},
    }
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(key, None)
    command = skyrl_cpu_command(
        "python",
        "-c",
        "import importlib.util,json,os,runpy,sys; assert importlib.util.find_spec('marin') is None; "
        "from marinskyrl.native_environment_cache import CONTROL_KEY; os.environ[CONTROL_KEY]=sys.argv[1]; "
        "print(json.dumps({'skyrl_python':sys.executable}),flush=True); "
        "runpy.run_module('marinskyrl.native_environment_cache',run_name='__main__')",
        json.dumps(profile),
    )
    try:
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=150, check=True)
        lines = result.stdout.strip().splitlines()
        skyrl_python = json.loads(lines[0])["skyrl_python"]
        receipt = json.loads(lines[-1])
        assert receipt["native_cache_restored"] == str(expected.parent)
        assert receipt["files"] == 1
        assert receipt["python_executable"] != skyrl_python
        assert (expected / "fixture.bin").read_bytes() == b"small compiler cache fixture"
    finally:
        shutil.rmtree(expected.parent.parent, ignore_errors=True)
