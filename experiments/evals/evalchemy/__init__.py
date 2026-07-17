# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy eval clients: the ``marin-community/evalchemy`` fork in the ``:evalchemy-tpu`` container.

Two entry paths share the container (``docker/evalchemy-tpu/Dockerfile``) and its pin (``image.py``):

- ``serve_and_eval.py`` — the composable eval path behind ``experiments.evals.evals``. A parent job
  serves a model with marin-serve, then runs ``run_evalchemy_client.py`` against the served OpenAI URL
  (``eval.eval --model local-completions``) and tears the server down. This is the post-hoc eval path
  for any checkpoint (issue #7267).
- ``marin_evalchemy_tpu.py`` — a standalone launcher that self-serves the model inside the container
  (``eval.eval --model vllm``) on a TPU pod, a seed loop over a ``suite`` preset. It consumes an HF
  export path or HF id (typically an SFT export) rather than importing ``experiments.sft_launcher``.

Both copy lm-eval's native ``results_*.json`` tree to a Marin artifact path for
``marin.evaluation.eval_result.EvalchemyResult`` to read back.
"""
