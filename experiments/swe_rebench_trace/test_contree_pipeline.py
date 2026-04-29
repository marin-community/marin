# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from experiments.swe_rebench_trace import contree_pipeline


def test_tracer_injection_defaults_to_python(monkeypatch):
    monkeypatch.delenv("CONTREE_TRACER", raising=False)

    env, files = contree_pipeline._tracer_injection()

    assert env == {"PYTHONPATH": "/pytracer"}
    assert files == {"/pytracer/sitecustomize.py": str(contree_pipeline.PYTRACER_DIR / "sitecustomize.py")}


def test_tracer_injection_uploads_rust_module(monkeypatch, tmp_path: Path):
    (tmp_path / "sitecustomize.py").write_text("")
    native = tmp_path / "_contree_rusttracer.cpython-311-x86_64-linux-gnu.so"
    native.write_text("")
    monkeypatch.setenv("CONTREE_TRACER", "rust")
    monkeypatch.setattr(contree_pipeline, "RUSTTRACER_DIR", tmp_path)

    env, files = contree_pipeline._tracer_injection()

    assert env == {"PYTHONPATH": "/pytracer"}
    assert files == {
        "/pytracer/sitecustomize.py": str(tmp_path / "sitecustomize.py"),
        "/pytracer/_contree_rusttracer.so": str(native),
    }


def test_tracer_injection_requires_built_rust_module(monkeypatch, tmp_path: Path):
    (tmp_path / "sitecustomize.py").write_text("")
    monkeypatch.setenv("CONTREE_TRACER", "rust")
    monkeypatch.setattr(contree_pipeline, "RUSTTRACER_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="_contree_rusttracer"):
        contree_pipeline._tracer_injection()
