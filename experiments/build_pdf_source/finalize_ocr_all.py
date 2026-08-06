# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finish the all-routes OCR corpus: quality filter, quality-aware fuzzy dedup, language labels.

:mod:`repair_ocr_all` left the corpus exact-deduped and decontaminated but deliberately skipped
fuzzy dedup: electing a canonical per near-duplicate cluster needed the quality signal (#7619)
that did not exist yet. That signal now does -- the pooled fast-transformer trained on the 100k
oracle sample -- so this entrypoint runs the three deferred stages in order:

1. :mod:`quality_label` -- score three 2000-char windows per document, append
   ``edu_begin``/``edu_middle``/``edu_end``/``edu_max`` (0-4 calibrated), and drop documents whose
   best window scores below 1.0.
2. :mod:`fuzzy_ocr_all` -- MinHash + connected components, then re-elect each cluster's canonical
   as the member with the highest ``edu_max`` (the election repair_ocr_all refused to freeze
   arbitrarily), and keep canonicals.
3. :mod:`language_label` -- FinePDFs-style page-level-average GlotLID labeling; appends
   ``language`` and ``language_score`` without dropping anything.

The final artifact, ``data/datakit/final/common_crawl_focus_2026_22_pdf_ocr_all``, is the #7621
training input (:mod:`experiments.grug.moe.launch_pdf_compare`).

Launch through the marin hub with the cw-us-east-02a restriction::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-finalize \\
        --cpu 4 --memory 16GB --enable-extra-resources \\
        -- python -m experiments.build_pdf_source.finalize_ocr_all

Entrypoint sizing follows :mod:`pipeline`: the consolidate steps run their Zephyr drivers in this
process, so it needs a few CPUs and GB rather than the tiny default. The ``MARIN_PREFIX`` guard
below serves the same purpose as pipeline.py's: without it, a launch that never reached a cluster
would run the whole DAG -- including a 6.2 GB corpus read -- on the launching machine.
"""

import logging
import os

from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.fuzzy_ocr_all import fuzzy_steps
from experiments.build_pdf_source.language_label import glotlid_model_step, language_label_step
from experiments.build_pdf_source.quality_label import QUALITY_SCHEMA, quality_label_step, scorer_model_step

MARIN_PREFIX_ENV = "MARIN_PREFIX"


def finalize_ocr_all_steps() -> list[StepSpec]:
    """Return the quality -> fuzzy-dedup -> language-label DAG over the cleaned ocr_all corpus."""
    scorer_model = scorer_model_step()
    quality = quality_label_step(scorer_model)
    fuzzy = fuzzy_steps(quality, QUALITY_SCHEMA)
    glotlid_model = glotlid_model_step()
    final = language_label_step(fuzzy[-1], glotlid_model, QUALITY_SCHEMA)
    return [scorer_model, quality, *fuzzy, glotlid_model, final]


def main() -> None:
    configure_logging(logging.INFO)
    if not os.environ.get(MARIN_PREFIX_ENV):
        raise RuntimeError(
            f"{MARIN_PREFIX_ENV} is unset, which means this is not running on a cluster. The steps "
            "read the 6.2 GB cleaned corpus and stage a 1.7 GB LID model, and must not run locally. "
            "Launch with `iris --cluster=marin job run --target-cluster cw-us-east-02a -- ...`, "
            "which injects the prefix."
        )
    StepRunner().run(finalize_ocr_all_steps())


if __name__ == "__main__":
    main()
