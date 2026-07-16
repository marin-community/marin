# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone evalchemy-on-TPU downstream eval launcher.

``marin_evalchemy_tpu.py`` drives the ``marin-community/evalchemy`` fork inside a pinned
``:evalchemy-tpu`` container (``docker/evalchemy-tpu/Dockerfile``) on a TPU pod, as a lazy
``ArtifactStep``: a ``suite`` -> tasks preset, a per-model ``run_name``, a chat-template ``stage``
flag, and a seed loop -> ``results_*.json`` copied to the Marin artifact path. It is independent of
``experiments.sft_launcher`` — it consumes an HF export path or HF id (typically produced by the SFT
launcher) rather than importing it.
"""
