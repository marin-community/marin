# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.protein import exp146_sweep as exp146

EXPECTED_MODELS = {
    "1_5b": (2048, 8192, 24, 32, 8, 64),
    "3b": (2560, 10240, 30, 48, 16, 64),
    "6b": (3200, 12800, 37, 64, 32, 64),
}


def test_versions_reuse_exp117_cache_under_fresh_sweep_identity():
    assert exp146.SWEEP_VERSION == "2026.07.23.01"
    assert exp146.CACHE_VERSION == "2026.07.13.1"


def test_models_match_marinfold_issue_146_exactly():
    assert exp146.MODEL_CONFIGS.keys() == EXPECTED_MODELS.keys()
    for model_size, expected in EXPECTED_MODELS.items():
        model = exp146.MODEL_CONFIGS[model_size]
        assert (
            model.hidden_dim,
            model.intermediate_dim,
            model.num_layers,
            model.num_heads,
            model.num_kv_heads,
            model.head_dim,
        ) == expected


@pytest.mark.parametrize("model_size", EXPECTED_MODELS)
def test_model_size_is_in_run_identity(model_size):
    point = exp146.Point(model_size, epochs=2, learning_rate=1e-3, weight_decay=0.1, batch_size=128)

    assert f"-{model_size}-" in exp146.run_id(point, "us-east5")
    assert f"model_size={model_size}" in exp146._tags(point, "us-east5", num_train_steps=10)


def test_grid_matches_marinfold_issue_146_exactly():
    assert exp146.SWEEP_BATCH_SIZES == {64, 128, 256}
    assert exp146.SWEEP_LEARNING_RATES == {3.1623e-4, 1e-3, 3.1623e-3}
    assert exp146.SWEEP_WEIGHT_DECAYS == {0.1, 0.2, 0.4, 0.8, 1.6}
    assert exp146.SWEEP_EPOCHS == {2, 4, 8}


def test_parse_point_canonicalizes_model_alias_and_accepts_grid_point(monkeypatch):
    for key, value in {
        "MODEL_SIZE": "1.5B",
        "EPOCHS": "2",
        "LR": "0.001",
        "WD": "0.1",
        "BATCH_SIZE": "128",
    }.items():
        monkeypatch.setenv(key, value)

    assert exp146.parse_point() == exp146.Point("1_5b", 2, 1e-3, 0.1, 128)


@pytest.mark.parametrize(
    ("key", "value", "expected_error"),
    [
        ("MODEL_SIZE", "1b", "unknown MODEL_SIZE"),
        ("EPOCHS", "1", "EPOCHS=1"),
        ("LR", "0.0001", "LR=0.0001"),
        ("WD", "0.3", "WD=0.3"),
        ("BATCH_SIZE", "32", "BATCH_SIZE=32"),
    ],
)
def test_parse_point_rejects_coordinates_outside_issue_146_grid(monkeypatch, key, value, expected_error):
    point = {
        "MODEL_SIZE": "3b",
        "EPOCHS": "4",
        "LR": "0.001",
        "WD": "0.4",
        "BATCH_SIZE": "128",
    }
    point[key] = value
    for env_key, env_value in point.items():
        monkeypatch.setenv(env_key, env_value)

    with pytest.raises(SystemExit, match=expected_error):
        exp146.parse_point()


def test_smoke_defaults_may_be_outside_production_grid(monkeypatch):
    for key in ("MODEL_SIZE", "EPOCHS", "LR", "WD", "BATCH_SIZE"):
        monkeypatch.delenv(key, raising=False)
    defaults = exp146.Point("1_5b", 1, 1e-3, 0.1, 128)

    assert exp146.parse_point(defaults=defaults) == defaults


def test_calibration_covers_each_issue_146_model_and_supported_tpu_family():
    assert exp146.CORRECTION_FACTORS.keys() == exp146.MODEL_CONFIGS.keys()
    for factors in exp146.CORRECTION_FACTORS.values():
        assert factors.keys() == {"v5e", "v5p", "v6e"}
        assert all(factor > 0 for factor in factors.values())


def test_batch_fit_uses_pr_7380_parallelism_model():
    point = exp146.Point("1_5b", epochs=2, learning_rate=1e-3, weight_decay=0.1, batch_size=128)

    assert exp146.batch_fit(point, "v6e-4", correction_factor=None) == exp146.TpuBatchConfig(
        data_parallelism=4,
        tensor_parallelism=1,
        per_device_parallelism=16,
        gradient_accumulation=2,
    )


def test_wandb_parameter_and_gradient_watch_is_disabled():
    assert not exp146.WANDB_WATCH_CONFIG.is_enabled
