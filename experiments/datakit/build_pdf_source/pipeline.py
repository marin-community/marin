# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Turn the focus crawl's PDFs into a datakit source (#7616).

Launch it at the CoreWeave fleet *through the marin hub*. ``cw-us-east-08a``'s controller surface
is IP-locked to the marin egress and has no off-cluster user surface, so connecting to it directly
(``--cluster=cw-us-east-08a``) falls back to a kubectl tunnel and fails on a laptop::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \
        --job-name build-pdf-source \
        --cpu 2 --memory 8GB --enable-extra-resources \
        -- python -m experiments.datakit.build_pdf_source.pipeline

The combine, normalize, and LID steps run their Zephyr drivers inside this process, so the
entrypoint needs a couple of CPUs and a few GB rather than the default. ``-m`` is safe even though
it loads this module as ``__main__``: every callable that crosses to a Fray job lives in its own
module and pickles by reference to it.

``main`` refuses to run without ``MARIN_PREFIX``, which the *target* cluster's ``defaults.task_env``
supplies; off-cluster it is unset and the whole DAG would otherwise run on the launching machine.

Sample size, coalescing, and packing are module constants in :mod:`plan`, not CLI flags. Changing one
re-keys the plan step and everything downstream; the old sample stays cached at its own path.
"""

import logging
import os

from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.classify import classify_step, model_step
from experiments.datakit.build_pdf_source.combine_routes import COMBINED_SCHEMA, combine_step
from experiments.datakit.build_pdf_source.dedup import exact_dedup_step
from experiments.datakit.build_pdf_source.extract_inspector import inspector_extract_step
from experiments.datakit.build_pdf_source.extract_ocr import ocr_extract_step
from experiments.datakit.build_pdf_source.fetch import fetch_step
from experiments.datakit.build_pdf_source.language_label import glotlid_model_step, language_label_step
from experiments.datakit.build_pdf_source.plan import plan_step

MARIN_PREFIX_ENV = "MARIN_PREFIX"


def build_pdf_source_steps() -> list[StepSpec]:
    """Return every step from the fetch plan through to the labeled datakit source.

    Extraction comes before routing: ``inspector_extract_step`` reads every fetched PDF's text layer,
    ``classify_step`` routes on what that produced, and ``ocr_extract_step`` re-reads the escalated
    subset with the vision model. The two routes therefore overlap, and ``combine_step`` keeps exactly
    one reading of each document.

    Every step up to the union is a 1:1 map that names its output shards after its input's, so no
    step holds a corpus-wide table. The one shuffle is the exact dedup (:mod:`dedup`); GlotLID
    labeling (:mod:`language_label`) is a 1:1 map over that. Decontamination and fuzzy dedup are the
    reference pipeline's cross-source stages and run there.
    """
    plan = plan_step()
    fetch = fetch_step(plan)
    text_extraction = inspector_extract_step(fetch)
    router = model_step()
    classify = classify_step(text_extraction, router)
    ocr_extraction = ocr_extract_step(fetch, classify)

    combined = combine_step(text_extraction, ocr_extraction, classify)
    normalized = exact_dedup_step(combined, COMBINED_SCHEMA)

    glotlid_model = glotlid_model_step()
    final = language_label_step(normalized, glotlid_model, COMBINED_SCHEMA)
    return [
        plan,
        fetch,
        text_extraction,
        router,
        classify,
        ocr_extraction,
        combined,
        normalized,
        glotlid_model,
        final,
    ]


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
