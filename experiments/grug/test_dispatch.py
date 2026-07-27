# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug import dispatch


def test_training_setup_scripts_are_opt_in(monkeypatch):
    monkeypatch.delenv("SCALE_A2A_HYBRID_EP", raising=False)
    monkeypatch.delenv("SCALE_NCCL_CUDA13", raising=False)

    assert dispatch._training_setup_scripts(["gpu"]) is None


def test_cuda13_nccl_setup_runs_after_environment_sync(monkeypatch):
    monkeypatch.delenv("SCALE_A2A_HYBRID_EP", raising=False)
    monkeypatch.setenv("SCALE_NCCL_CUDA13", "1")
    monkeypatch.setattr(dispatch, "default_setup_script", lambda **_: "default")
    monkeypatch.setattr(dispatch, "cuda_toolchain_setup_script", lambda: "cuda")

    scripts = dispatch._training_setup_scripts(["gpu"])

    assert scripts == ["default", "cuda", dispatch._CUDA13_NCCL_SETUP_SCRIPT]


def test_cuda13_nccl_setup_composes_with_hybridep(monkeypatch):
    monkeypatch.setenv("SCALE_A2A_HYBRID_EP", "1")
    monkeypatch.setenv("SCALE_NCCL_CUDA13", "1")
    monkeypatch.setattr(dispatch, "default_setup_script", lambda **_: "default")
    monkeypatch.setattr(dispatch, "cuda_toolchain_setup_script", lambda: "cuda")

    assert dispatch._training_setup_scripts(["gpu"]) == [
        "default",
        "cuda",
        dispatch._HYBRIDEP_SETUP_SCRIPT,
        dispatch._CUDA13_NCCL_SETUP_SCRIPT,
    ]
