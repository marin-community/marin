# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""mixture-selection's bundled inputs rebuilt from the dataset build, its self-test on them, and the archived swarms'
committed refits."""

import filecmp
import json
import shutil

import numpy as np
import pandas as pd
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


@pytest.mark.timeout(300)
@pytest.mark.parametrize("swarm_table", regen.ARCHIVED_SWARMS)
def test_committed_archived_fits_refit_from_the_build(swarm_table, tmp_path):
    """mixture-selection's fit of each objective's first task, on the swarm's inputs written from the build and with
    the inner folds recomputed, reproduces the committed parameters exactly."""
    swarm = swarm_table.removeprefix("swarm_")
    regen.write_archived(paths.BUILD, swarm_table, tmp_path)
    objectives = tables.read_csv(tmp_path / "objectives.csv")
    objectives.groupby("objective", sort=False).head(1).to_csv(tmp_path / "objectives.csv", index=False)
    committed = tables.read_csv(regen.ARCHIVED_FITS, comment="#")
    for objective in regen.OBJECTIVES.values():
        out = tmp_path / f"fit_{objective}.json"
        result = regen.run_mixture_selection(regen.fit_command(tmp_path, objective, out), cwd=tmp_path)
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        refit = regen.task_parameters(json.loads(out.read_text()), swarm, paths.BUILD)
        expected = committed.merge(refit[["swarm", "objective", "task_id"]], how="inner")
        pd.testing.assert_frame_equal(refit, expected, check_exact=True)
