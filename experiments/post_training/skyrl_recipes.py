# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from skyrlrecipe import Generator, RecipePatch, Trainer

MARIN_RECIPE_POLICY = RecipePatch(
    trainer=Trainer(strategy="megatron"),
    generator=Generator(run_engines_locally=True),
)
