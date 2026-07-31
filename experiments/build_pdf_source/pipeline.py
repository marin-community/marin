# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Fetch a sample of the focus crawl's PDFs, as input for PDF extraction (#7616).

Launch it at the CoreWeave fleet *through the marin hub*. ``cw-us-east-08a``'s controller surface
is IP-locked to the marin egress and has no off-cluster user surface, so connecting to it directly
(``--cluster=cw-us-east-08a``) falls back to a kubectl tunnel and fails on a laptop::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \
        --job-name build-pdf-source \
        -- python -m experiments.build_pdf_source.pipeline

The entrypoint is only the ``StepRunner`` driver, so its default 0.1 CPU / 1 GB is right; every
step submits its own Fray job with its own resources, and the fetch and classify steps each submit
a Zephyr coordinator and worker fleet from there. ``-m`` is safe even though it loads this module as
``__main__``: every callable that crosses to a Fray job lives in :mod:`plan`, :mod:`fetch` or
:mod:`classify` and pickles by reference to its real module path.

``main`` refuses to run without ``MARIN_PREFIX``, which the *target* cluster's ``defaults.task_env``
supplies (08a pins ``s3://marin-us-east-02a/marin``; the marin hub pins nothing, so a run that
forgot ``--target-cluster`` stops here). Off-cluster the variable is unset, ``marin_prefix()``
silently falls back to ``/tmp/marin``, and Fray falls back to a ``LocalClient`` -- so without this
check a launch that never reached a cluster runs the whole DAG on the launching machine instead,
pulling hundreds of GiB from Common Crawl to a laptop.

Sample size, coalescing, and packing are module constants in :mod:`plan`, not CLI flags, so every
run's identity is its code. Changing one re-keys the plan step, which re-keys the fetch step, and
both rebuild -- the old sample stays cached at its own path.
"""

import logging
import os

from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.extract import extract_step
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

MARIN_PREFIX_ENV = "MARIN_PREFIX"


def build_pdf_source_steps() -> list[StepSpec]:
    """Return every step from the fetch plan through to extracted text.

    The two model steps are independent of each other: ``model_step`` stages the OCR router, and
    ``layout_model_step`` builds the quantized layout graph from the fetched corpus. Extraction
    needs the routing table and the layout graph, and runs only the text-extractable route.
    """
    plan = plan_step()
    fetch = fetch_step(plan)
    ocr_router = model_step()
    classify = classify_step(fetch, ocr_router)
    layout_model = layout_model_step(fetch)
    return [plan, fetch, ocr_router, classify, layout_model, extract_step(fetch, classify, layout_model)]


def main() -> None:
    configure_logging(logging.INFO)
    if not os.environ.get(MARIN_PREFIX_ENV):
        raise RuntimeError(
            f"{MARIN_PREFIX_ENV} is unset, which means this is not running on a cluster. The fetch "
            "step pulls hundreds of GiB from data.commoncrawl.org and must not run locally. Launch "
            "it with `iris --cluster=<name> job run -- ...`, which injects the prefix."
        )
    StepRunner().run(build_pdf_source_steps())


if __name__ == "__main__":
    main()
