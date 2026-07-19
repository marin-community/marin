# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A one-command evaluation launcher over the ``serve_and_eval`` engine.

Pick a model and an eval suite from the registries here (:mod:`experiments.evaluation.models`,
:mod:`experiments.evaluation.evals`), and :mod:`experiments.evaluation.launch` sizes the serving slice
(:mod:`experiments.evaluation.hardware`), submits one CPU orchestrator job per run, and writes a
durable ``record.json`` (:mod:`marin.evaluation.records`) that the Postgres index
(:mod:`marin.evaluation.results_db`) mirrors. The command-line entry point is
:mod:`experiments.evaluation.cli`.
"""
