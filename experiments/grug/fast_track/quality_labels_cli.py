# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the GLM/Harrier labels and fit a ridge candidate on fixed splits."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.fast_track.quality_features import HARRIER_FEATURE_IDENTITY
from experiments.grug.fast_track.quality_labels import (
    GLM_HARRIER_JOIN_PATH,
    GLM_LABELS_PATH,
    GLM_ORACLE,
    FittedRidgeQualityHead,
    QualityLabelSpec,
    build_quality_labels,
    build_ridge_quality_head,
)


@click.command()
@click.option("--labels-path", default=GLM_LABELS_PATH, show_default=True)
@click.option("--joined-path", default=GLM_HARRIER_JOIN_PATH, show_default=True)
@click.option("--regularization", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--split-seed", type=click.IntRange(min=0), default=0, show_default=True)
@build_options
def main(
    labels_path: str, joined_path: str, regularization: float, split_seed: int
) -> ArtifactStep[FittedRidgeQualityHead]:
    labels = build_quality_labels(QualityLabelSpec(labels_path, joined_path, GLM_ORACLE, HARRIER_FEATURE_IDENTITY))
    return build_ridge_quality_head(labels, regularization=regularization, split_seed=split_seed)


if __name__ == "__main__":
    main()
