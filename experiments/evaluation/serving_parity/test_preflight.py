# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.evaluation.serving_parity import bootstrap
from experiments.evaluation.serving_parity.settings import PROBE


def test_actual_engine_preflight_is_required_and_mismatch_fails(monkeypatch, tmp_path):
    observed = ["sliding_attention", "full_attention"]
    captured = {}

    class EngineArgs:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_engine_config(self):
            return SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=SimpleNamespace(layer_types=observed)),
                cache_config=SimpleNamespace(sliding_window=None),
            )

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.engine": ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": SimpleNamespace(EngineArgs=EngineArgs),
        "vllm.plugins": SimpleNamespace(load_general_plugins=lambda: None),
        "vllm.transformers_utils": ModuleType("vllm.transformers_utils"),
        "vllm.transformers_utils.configs": ModuleType("vllm.transformers_utils.configs"),
        "vllm.transformers_utils.configs.grugmoe": SimpleNamespace(
            grug_moe_layer_types=lambda layers, interval: ["sliding_attention", "full_attention"]
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("experiments.evaluation.serving_parity.config_preflight", None)
    preflight = importlib.import_module("experiments.evaluation.serving_parity.config_preflight")
    config = dict(
        probe=vars(PROBE), weights="local-model", tensor_parallel_size=4, preflight_path=str(tmp_path / "preflight.json")
    )
    assert preflight.check_config(config)["status"] == "PASS"
    assert captured["tensor_parallel_size"] == 4
    assert captured["max_model_len"] == PROBE.max_model_len
    assert captured["max_num_seqs"] == PROBE.max_num_seqs
    assert "hf_overrides" not in captured
    observed.clear()
    with pytest.raises(RuntimeError, match="preflight"):
        preflight.check_config(config)
    assert json.loads(Path(config["preflight_path"]).read_text())["status"] == "FAIL"


def test_bootstrap_uses_explicit_config_and_execs_after_instrumentation(monkeypatch, tmp_path):
    events = []
    config = dict(hardware="v5p-8", diagnostic_v4_tiles=False, runtime_report_path=str(tmp_path / "runtime.json"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("PARITY_TPU_VARIANT", "v4-32")

    def run_path(script):
        if script.endswith("config_preflight.py"):
            return {"check_config": lambda actual: events.append(("preflight", actual)) or {"status": "PASS"}}
        assert script.endswith("trace_bootstrap.py")
        return {"patch_trace": lambda: events.append(("trace", None)) or {"arithmetic_changed": False}}

    class Executed(Exception):
        pass

    def execute(python, command):
        events.append(("exec", command))
        raise Executed

    monkeypatch.setattr(bootstrap.runpy, "run_path", run_path)
    monkeypatch.setattr(bootstrap.os, "execv", execute)
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--config", str(path), "python", "runtime.py", "--flag"])
    with pytest.raises(Executed):
        bootstrap.main()
    assert [event[0] for event in events] == ["preflight", "trace", "exec"]
    assert events[0][1]["hardware"] == "v5p-8"
    assert events[-1][1][1:] == ["runtime.py", "--flag"]
    assert json.loads(Path(config["runtime_report_path"]).read_text())["patches"] == [{"arithmetic_changed": False}]
