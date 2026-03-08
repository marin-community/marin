# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initial training surface for JPEG tokenizer trials.

This layer stays local to the JPEG tokenizer experiment tree even while it
reuses the grug base trainer implementation. That keeps the eventual fork
boundary narrow once tokenizer-specific evaluation and logging diverge.
"""

from experiments.grug.base.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    build_tagged_evaluator,
    build_train_dataset,
    build_train_loader,
    initial_state,
    run_grug,
)

JpegEvalConfig = GrugEvalConfig
JpegRunConfig = GrugRunConfig
JpegTrainerConfig = GrugTrainerConfig
run_jpeg_tokenizer = run_grug

__all__ = [
    "JpegEvalConfig",
    "JpegRunConfig",
    "JpegTrainerConfig",
    "build_tagged_evaluator",
    "build_train_dataset",
    "build_train_loader",
    "initial_state",
    "run_jpeg_tokenizer",
]
