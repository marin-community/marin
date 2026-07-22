# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.moe.standalone import launch_ngc


def test_launch_uses_ngc_image_and_disables_retries(monkeypatch) -> None:
    captured = {}

    def capture(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(launch_ngc, "dispatch_grug_training_run", capture)

    launch_ngc.launch_trial(
        run_id="raa-ep16-t1",
        arguments=("--run-id", "raa-ep16-t1", "--num-gpus", "64"),
        replicas=16,
        gpus_per_node=4,
    )

    assert captured["run_id"] == "raa-ep16-t1"
    assert captured["max_retries_failure"] == 0
    assert captured["resources"].image == "nvcr.io/nvidia/jax:26.06-py3"
    assert captured["resources"].replicas == 16
    assert captured["resources"].device.count == 4
