# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training import rl_gate
from experiments.post_training.rl_tests import fixtures
from experiments.post_training.rl_tests.native_composer import NativeComposer
from infra.marina.tests.rl_progress_fixtures import rl_progress_client as rl_progress_client


@pytest.fixture(scope="session")
def native_composer():
    composer = NativeComposer()
    yield composer
    composer.close()


@pytest.fixture(autouse=True)
def reuse_native_composer(native_composer, monkeypatch):
    """Run native compositions in one SkyRL process instead of one per call."""
    monkeypatch.setattr(rl_gate, "_run_native", native_composer.run_native)


selection = pytest.fixture(fixtures.selection)
source_provenance = pytest.fixture(fixtures.source_provenance)
recorded_runs = pytest.fixture(fixtures.recorded_runs)
preset_table = pytest.fixture(fixtures.preset_table)


@pytest.fixture
def mochi_sft_presets(monkeypatch):
    from experiments.grug.moe_hero_ep import launch_hero_sft, sft_recipe  # noqa: PLC0415
    from experiments.post_training.mochi_sft import MIXTURE_262K_16NODE, MIXTURE_262K_176NODE  # noqa: PLC0415

    monkeypatch.setattr(sft_recipe, "checkpoint_step", lambda _path: 393491)
    monkeypatch.setattr(launch_hero_sft, "special_token_init", lambda _step: None)
    return {"periodic": MIXTURE_262K_176NODE, "bounded": MIXTURE_262K_16NODE}
