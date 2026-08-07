# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Turn the focus crawl's PDFs into the final labeled dataset (#7616).

Launch it at the CoreWeave fleet *through the marin hub*. ``cw-us-east-08a``'s controller surface
is IP-locked to the marin egress and has no off-cluster user surface, so connecting to it directly
(``--cluster=cw-us-east-08a``) falls back to a kubectl tunnel and fails on a laptop::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \
        --job-name build-pdf-source \
        --cpu 2 --memory 8GB --enable-extra-resources \
        -- python -m experiments.datakit.build_pdf_source.pipeline

The entrypoint sizing follows the datakit reference pipeline's: the extraction steps submit their
own Fray jobs, but the combine, dedup, quality, fuzzy, and LID steps run their Zephyr drivers --
and, on a fresh prefix, the eval bloom build -- inside this process, so it needs a couple of CPUs
and a few GB rather than the 0.1 CPU / 1 GB default. ``-m`` is safe even though it loads this
module as ``__main__``: every callable that crosses to a Fray job lives in its own real module
(:mod:`plan` through :mod:`language_label`) and pickles by reference to its real module path.

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

from experiments.datakit.build_pdf_source.classify import classify_step, model_step
from experiments.datakit.build_pdf_source.combine_routes import COMBINED_SCHEMA, combine_step
from experiments.datakit.build_pdf_source.dedup import dedup_steps
from experiments.datakit.build_pdf_source.extract_fleet import fleet_extract_step
from experiments.datakit.build_pdf_source.extract_ocr import ocr_extract_step
from experiments.datakit.build_pdf_source.fetch import fetch_step
from experiments.datakit.build_pdf_source.fuzzy_dedup import fuzzy_steps
from experiments.datakit.build_pdf_source.language_label import glotlid_model_step, language_label_step
from experiments.datakit.build_pdf_source.plan import plan_step
from experiments.datakit.build_pdf_source.quality_label import quality_label_step, quality_output_schema, scorer_model_step

MARIN_PREFIX_ENV = "MARIN_PREFIX"


def build_pdf_source_steps() -> list[StepSpec]:
    """Return every step from the fetch plan through to the final labeled dataset.

    The classifier splits the corpus in two and the two extraction steps are independent from
    there on: ``fleet_extract_step`` runs the text-extractable route through the persistent
    docling converter fleet on CPUs, and ``ocr_extract_step`` runs the rest through a vision model
    on GPUs. ``combine_step`` unions the two back into one corpus, tagged with the router's
    decision. (The fleet's FP32 torch arms do not consume the INT8 OpenVINO layout graph;
    :mod:`~experiments.datakit.build_pdf_source.layout_model` remains as the utility that rebuilds it if a
    VNNI backend is reintroduced.)

    From there the chain is single-corpus: exact dedup + decontamination (:mod:`dedup`), quality
    scoring and gating (:mod:`quality_label`), fuzzy dedup with quality-based canonical election
    (:mod:`fuzzy_dedup`), and finally GlotLID language labeling (:mod:`language_label`), whose
    output is the dataset the training run consumes.
    """
    plan = plan_step()
    fetch = fetch_step(plan)
    ocr_router = model_step()
    classify = classify_step(fetch, ocr_router)
    text_extraction = fleet_extract_step(fetch, classify)
    ocr_extraction = ocr_extract_step(fetch, classify)

    combined = combine_step(text_extraction, ocr_extraction)
    dedup = dedup_steps(combined, COMBINED_SCHEMA)
    clean = dedup[-1]

    scorer_model = scorer_model_step()
    quality = quality_label_step(clean, scorer_model, COMBINED_SCHEMA)
    quality_schema = quality_output_schema(COMBINED_SCHEMA)

    fuzzy = fuzzy_steps(quality, quality_schema)
    fuzzy_clean = fuzzy[-1]

    glotlid_model = glotlid_model_step()
    final = language_label_step(fuzzy_clean, glotlid_model, quality_schema)
    return [
        plan,
        fetch,
        ocr_router,
        classify,
        text_extraction,
        ocr_extraction,
        combined,
        *dedup,
        scorer_model,
        quality,
        *fuzzy,
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
