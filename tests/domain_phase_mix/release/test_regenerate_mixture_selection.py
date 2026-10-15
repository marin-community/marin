# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""mixture-selection's bundled inputs rebuilt from the dataset build, and its self-test on them."""

import filecmp
import shutil

import numpy as np
import pytest

from experiments.domain_phase_mix.release import paths, tables
from experiments.domain_phase_mix.release import regenerate_mixture_selection_data as regen

pytestmark = pytest.mark.skipif(
    not (paths.BUILD / "MANIFEST.json").exists() or not paths.MIXSEL.exists(), reason="build or mixture-selection absent"
)
BUNDLE = paths.MIXSEL / "data"
# The self-test compares its refits with the reference fits to 1e-8, and the dataset's exact epochs per unit weight move
# them by 4.6e-8 (swarm) and 1.0e-7 (held-out bank) from the bundle's, a relative 5.0e-7 larger: the staged data keeps
# the bundle's buckets.csv, which test_regenerated_buckets_and_anchors_agree_with_the_bundle compares numerically.
SELF_TEST_FILES = ("swarm_weights.csv", "swarm_outcomes.csv", "objectives.csv", "anchors.csv")


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    out = tmp_path_factory.mktemp("regenerated")
    regen.write(paths.BUILD, out)
    return out


def test_regenerated_swarm_and_objectives_are_the_bundle_byte_for_byte(regenerated):
    for name in ("swarm_weights.csv", "swarm_outcomes.csv", "objectives.csv"):
        assert filecmp.cmp(regenerated / name, BUNDLE / name, shallow=False), name


def test_regenerated_buckets_and_anchors_agree_with_the_bundle(regenerated):
    ours, bundle = (tables.read_csv(folder / "buckets.csv") for folder in (regenerated, BUNDLE))
    assert list(ours.bucket) == list(bundle.bucket) and list(ours.pool_tokens) == list(bundle.pool_tokens)
    # The bundle divides a budget of 6,325,186,813,003.79 tokens, a relative 5.0e-7 above D_TGT, by the pool sizes.
    np.testing.assert_allclose(ours.epochs_per_unit_weight, bundle.epochs_per_unit_weight, rtol=1e-6, atol=0)
    ours, bundle = (
        tables.read_csv(folder / "anchors.csv").set_index(["objective", "component"]) for folder in (regenerated, BUNDLE)
    )
    assert sorted(ours.index) == sorted(bundle.index)
    np.testing.assert_allclose(ours.loc[bundle.index].to_numpy(), bundle.to_numpy(), rtol=0, atol=1e-12)


@pytest.mark.timeout(1800)  # the self-test refits both objectives and five policies: about two minutes
def test_self_test_passes_on_the_regenerated_data(regenerated, tmp_path):
    """The self-test refits both objectives on the staged data and must reproduce the reference fits and policies.

    The bucket and fold tables, the reference fits and policies and the held-out banks stay the bundle's.
    """
    shutil.copytree(BUNDLE, tmp_path / "data")
    for name in SELF_TEST_FILES:
        shutil.copyfile(regenerated / name, tmp_path / "data" / name)
    shutil.copyfile(paths.MIXSEL / "mixture_selection.py", tmp_path / "mixture_selection.py")
    result = regen.run_mixture_selection(["mixture_selection.py", "self-test"], cwd=tmp_path)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
