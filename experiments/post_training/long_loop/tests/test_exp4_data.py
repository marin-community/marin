# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import pytest
from taskcompendium.models import Source

from experiments.post_training.long_loop.exp4_data import (
    EXPECTED_COMPOSITION,
    FIVE_FAMILY_COUNTS,
    NEMOTRON_IF_RECIPE_VERSION,
    TIME_INTERVALS_ROWS,
    _cycle_index,
    _old_question,
    _row_family,
    row_identity,
)
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    REASONING_GYM_LABEL,
    TEACHER_ROUTE_LABELS,
)


def test_expected_composition_names_every_pool_label_at_the_adopted_counts():
    assert EXPECTED_COMPOSITION["long-loop-smartie-swesmith"] == 5608
    assert EXPECTED_COMPOSITION[REASONING_GYM_LABEL] == 20_609
    assert EXPECTED_COMPOSITION["nemotron_if"] == 56_083
    assert sum(EXPECTED_COMPOSITION.values()) == 161_500
    # The blend-sourced labels keep the adopted release's exact per-source rows.
    assert EXPECTED_COMPOSITION["open-thoughts/TaskTrove"] == 24_033


def test_the_excluded_generated_family_is_codeio_at_its_adopted_count():
    from experiments.post_training.long_loop.exp4_data import EXCLUDED_GENERATED_FAMILIES, GENERATED_POOL_ROWS

    assert EXCLUDED_GENERATED_FAMILIES == {"codeio": 191}
    assert GENERATED_POOL_ROWS == 20_609


def test_row_identity_derives_the_pipeline_task_id_from_the_source():
    source = Source(
        dataset="https://files.pythonhosted.org/packages/42/21/wheel.whl",
        revision="a" * 64,
        row="wheel.whl:35139",
        importer_revision="1",
    )
    first = row_identity("reasoning_gym_generated", source)
    second = row_identity("reasoning_gym_generated", source)
    assert first == second
    assert first.startswith("reasoning_gym_generated-")
    other = row_identity("reasoning_gym_generated", source.model_copy(update={"row": "wheel.whl:35140"}))
    assert other != first


def test_stamp_helpers_read_the_generated_rows_verifier_config():
    task = {
        "id": "reasoning_gym_generated-" + "0" * 64,
        "source": {"dataset": "wheel", "revision": "1", "row": "wheel:35139", "importer_revision": "1"},
        "resources": {
            "verifier": [
                {
                    "path": "config.json",
                    "source": {
                        "kind": "inline_file",
                        "content_base64": (
                            base64.b64encode(
                                json.dumps(
                                    {
                                        "contract": {
                                            "entry": {
                                                "question": "Count the flips.",
                                                "metadata": {"source_dataset": "coin_flip", "source_index": 7},
                                            }
                                        }
                                    }
                                ).encode()
                            ).decode()
                        ),
                    },
                }
            ]
        },
    }
    assert _row_family(task) == "coin_flip"
    assert _cycle_index(task) == 35139
    assert _old_question(task) == "Count the flips."
    plain = {"id": "x", "source": {"dataset": "wheel", "revision": "1", "row": "wheel:1", "importer_revision": "1"}}
    assert _row_family(plain) is None


def test_the_repaired_family_counts_come_from_the_adopted_label():
    assert FIVE_FAMILY_COUNTS == {
        "coin_flip": 215,
        "complex_arithmetic": 216,
        "game_of_life_halting": 206,
        "propositional_logic": 186,
        "puzzle24": 192,
    }
    assert TIME_INTERVALS_ROWS == 201


def test_the_wheel_route_maps_to_the_curriculum_label():
    wheel_url = "https://files.pythonhosted.org/packages/42/21/bb9f6d2f76424f8b516fe83f8026e52e466b0202341bb4ee7a9861e06277/reasoning_gym-0.1.25-py3-none-any.whl"
    assert TEACHER_ROUTE_LABELS[wheel_url] == REASONING_GYM_LABEL


@pytest.mark.parametrize(
    "recipe_version",
    [NEMOTRON_IF_RECIPE_VERSION],
)
def test_nemotron_if_binds_the_verifyit_recipe_version(recipe_version):
    assert recipe_version == "2"
