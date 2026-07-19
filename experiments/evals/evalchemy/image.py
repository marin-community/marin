# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned evalchemy eval-client container image.

The ``marin-community/evalchemy`` fork is only installable editable from a git clone (its graders
live in the source tree and its declared deps fight the base image's pins), so it ships as a
container rather than a wheel — see ``docker/evalchemy-tpu/Dockerfile``. The eval client hits a
served OpenAI URL (``eval.eval --model local-completions``), so it needs none of the image's
vLLM/TPU stack and runs on a CPU slice; making the fork ``uvx``-installable to drop the container is
tracked in #7270. Both the composable eval path (``serve_and_eval``) and the standalone launcher
(``marin_evalchemy_tpu``) reference the image through these constants so the pin lives in one place.
"""

import os

# The evalchemy-tpu image (FROM ghcr.io/open-thoughts/openthoughts-agent:tpu + the evalchemy fork +
# lm-eval v0.4.12 + graders), published under the fork's org. Referenced by immutable digest;
# overridable via env so the launcher can be pointed at a fresh build without a code edit.
EVALCHEMY_IMAGE = os.environ.get(
    "EVALCHEMY_TPU_IMAGE",
    "ghcr.io/marin-community/evalchemy-tpu@sha256:1b134caeb2ab7967905d5de93dcafd88aba24c62f12d78089b32a9ef61819f7b",
)

# evalchemy/lm-eval are installed into the image's OWN venv (docker/evalchemy-tpu/Dockerfile:
# VIRTUAL_ENV=/opt/openthoughts/.venv), NOT the Iris repo venv (/app/.venv) that the child job syncs +
# activates. So ``eval.eval`` must be invoked with the baked interpreter explicitly.
EVALCHEMY_PYTHON = os.environ.get("EVALCHEMY_PYTHON", "/opt/openthoughts/.venv/bin/python")
