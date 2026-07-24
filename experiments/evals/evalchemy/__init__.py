# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy eval client: the ``marin-community/evalchemy`` fork in the ``:evalchemy-tpu`` container.

``serve_and_eval.py`` is the composable eval path behind ``experiments.evals.evals``, sharing the
container (``docker/evalchemy-tpu/Dockerfile``) and its pin (``image.py``). A parent job serves a
model with marin-serve, then runs ``run_evalchemy_client.py`` against the served OpenAI URL
(``eval.eval --model local-completions``) and tears the server down. It copies lm-eval's native
``results_*.json`` tree to a Marin artifact path for
``marin.evaluation.eval_result.EvalchemyResult`` to read back.
"""
