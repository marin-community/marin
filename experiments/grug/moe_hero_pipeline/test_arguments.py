# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.moe_hero_pipeline.arguments import parse_args


@pytest.mark.parametrize("optimizer_args", [[], ["--optimizer", "adamw"]])
def test_main_recipe_rejects_adamw_before_training(optimizer_args):
    # The main recipe previously accepted AdamW but built and ran MuonH.
    with pytest.raises(SystemExit) as error:
        parse_args(["--main-hero-recipe", "--processes-per-task", "4", *optimizer_args])
    assert error.value.code == 2
