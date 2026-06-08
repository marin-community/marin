# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from experiments.midtrain_specs.true_midtrain.nemotron_math_only import launcher


def test_verify_final_checkpoint_restores_scheme_for_fsspec_gcs_listing(monkeypatch):
    checked: list[tuple[str, str, int]] = []
    spec = SimpleNamespace(
        run=SimpleNamespace(
            run_id="delphi-true-3e20-p33m67-cooldown20-a010",
            permanent_checkpoints_uri="gs://marin-us-east5/checkpoints/run/checkpoints",
        ),
        model_config={"type": "qwen3"},
        base=SimpleNamespace(num_layers=23),
    )

    monkeypatch.setattr(
        launcher,
        "default_gcs_list",
        lambda _: (
            "marin-us-east5/checkpoints/run/checkpoints/step-28408/",
            "marin-us-east5/checkpoints/run/checkpoints/step-35509/",
        ),
    )
    monkeypatch.setattr(
        launcher,
        "assert_checkpoint_complete_for_model_type",
        lambda checkpoint_dir, *, model_type, num_layers: checked.append((checkpoint_dir, model_type, num_layers)),
    )

    launcher._verify_final_checkpoint(spec, num_train_steps=35510)

    assert checked == [
        ("gs://marin-us-east5/checkpoints/run/checkpoints/step-28408", "qwen3", 23),
        ("gs://marin-us-east5/checkpoints/run/checkpoints/step-35509", "qwen3", 23),
    ]
