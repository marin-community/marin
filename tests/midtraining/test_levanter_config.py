# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Levanter YAML renderer."""

from unittest.mock import patch

import yaml
from marin.midtraining.levanter_config import (
    render_train_lm_config,
    render_train_lm_yaml,
)
from marin.midtraining.modes import (
    CHECKPOINT_INIT_MODE_FULL_STATE,
    CHECKPOINT_INIT_MODE_MODEL_ONLY,
)
from marin.midtraining.spec import resolve_midtrain_spec

from tests.midtraining._fixtures import (
    FAKE_1E21,
    FAKE_1E22,
    make_cooldown_spec,
    make_cpt_spec,
    make_data_manifest,
)


def _resolve(spec):
    manifest = make_data_manifest(region=spec.run.output_region)
    with patch("marin.midtraining.spec.load_data_manifest", return_value=manifest):
        return resolve_midtrain_spec(spec)


def test_cpt_yaml_has_model_only_init_mode_and_native_path():
    resolved = _resolve(make_cpt_spec())
    rendered = render_train_lm_config(resolved)
    assert rendered["checkpoint_init_mode"] == CHECKPOINT_INIT_MODE_MODEL_ONLY
    assert rendered["initialize_from_checkpoint_path"] == FAKE_1E21.verified_checkpoint_path
    assert rendered["trainer"]["num_train_steps"] == resolved.num_train_steps
    assert rendered["trainer"]["id"] == resolved.spec.run.run_id
    assert rendered["hf_save_path"].endswith("/hf")


def test_cpt_yaml_renders_wandb_tags_with_budget_facts():
    resolved = _resolve(make_cpt_spec())
    rendered = render_train_lm_config(resolved)
    tags = rendered["trainer"]["tracker"]["tags"]
    assert any(t.startswith("budget_kind:pretrain_fraction") for t in tags)
    assert any(t.startswith("requested_tokens:") for t in tags)
    assert any(t.startswith("actual_tokens:") for t in tags)
    assert any(t.startswith("pretrain_fraction_actual:") for t in tags)


def test_cooldown_yaml_has_no_init_path_and_full_state_mode():
    resolved = _resolve(make_cooldown_spec())
    rendered = render_train_lm_config(resolved)
    assert rendered["initialize_from_checkpoint_path"] is None
    assert rendered["checkpoint_init_mode"] == CHECKPOINT_INIT_MODE_FULL_STATE
    assert rendered["trainer"]["num_train_steps"] == FAKE_1E22.num_train_steps


def test_yaml_serializes_safely():
    text = render_train_lm_yaml(_resolve(make_cpt_spec()))
    parsed = yaml.safe_load(text)
    assert parsed["checkpoint_init_mode"] == CHECKPOINT_INIT_MODE_MODEL_ONLY


def test_temp_checkpoint_path_includes_run_id():
    resolved = _resolve(make_cpt_spec())
    rendered = render_train_lm_config(resolved)
    temp = rendered["trainer"]["checkpointer"]["temporary_base_path"]
    assert resolved.spec.run.run_id in temp
    assert "ttl=14d" in temp
