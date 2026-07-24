# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete definitions and submission policy for the shared evaluation framework.

Pick a model and an eval suite from the registries here (:mod:`experiments.evaluation.models`,
:mod:`experiments.evaluation.evals`), and :mod:`experiments.evaluation.launch` sizes the serving slice
under :mod:`experiments.evaluation.hardware`'s Marin fleet policy, submits one CPU orchestrator, and
hands a fully resolved group to :mod:`marin.evaluation.group_runner`. The command-line entry point is
:mod:`experiments.evaluation.cli`.
"""
