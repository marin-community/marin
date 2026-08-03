# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OCR the *entire* 10% sample -- both routes -- for the OCR-vs-CPU-extraction comparison.

The production pipeline OCRs only the documents the router sent to OCR. This run OCRs all 315,776
classified documents (5.6M pages), including the 214,444 the router judged text-extractable, so
every document the docling route extracted also exists as an OCR extraction and the two can be
compared row for row. It is a one-off corpus, not a pipeline step: nothing downstream consumes it,
and :mod:`~experiments.build_pdf_source.pipeline` does not know it exists.

**The run is partitioned into independent steps, one fleet per step.** Every request body passes
through its fleet's proxy and broker as ~1.9 MB of base64 PNG, and the proxy parks one OS thread
per in-flight request, so per-fleet processes are kept at the size the broker ceiling benchmark
(``_bench_broker_ceiling``) validated rather than asking one driver to hold every partition's
in-flight budget at once. Each partition takes every ``NUM_PARTITIONS``-th source shard, so the
partitions see statistically identical work and finish together.

Each partition is its own entry job, so the partitions run concurrently and one partition's
failure leaves the others' completed output cached under their own step names. Submit all of
them, at batch priority per the budget policy (the whole run is ~44 GB200 nodes, well past the
128k budget's ~18):

    for i in $(seq 0 5); do
      uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
          --priority batch --job-name ocr-all-p$i \\
          -- python -m experiments.build_pdf_source.extract_ocr_all --partition $i
    done

A finished partition's output lands at ``data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p<i>``;
the comparison reads the union of the partition dirs.
"""

import argparse
import logging
from functools import partial
from hashlib import sha256

from fray.types import ResourceConfig
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.extract_ocr import (
    BOILERPLATE_OPTIONS,
    RENDER_OPTIONS,
    ocr_pdf_text,
    sender_fleet_size,
)
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.ocr_extract.client import DEFAULT_MAX_TOKENS, PROMPT_DOC2MD
from experiments.build_pdf_source.ocr_extract.fleet import MODEL
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# Set from the broker ceiling benchmark: instances one broker demonstrably feeds, times the
# partition count, is the fleet. 6 x 29 = 174 GPUs ~= 44 GB200 nodes; at the sweep's 71 pages/s
# per node that is ~30 minutes of steady state over the sample's 5.6M pages.
NUM_PARTITIONS = 6
INSTANCES_PER_PARTITION = 29

# The proxy parks one thread per in-flight request (512 x instances), each holding its ~2 MB body
# until the response lands: ~29 GB of payload at full depth, plus thread stacks, the Zephyr
# coordinator, and the routing-free key set.
_DRIVER_RESOURCES = ResourceConfig(cpu=12, ram="96g", disk="32g")


def ocr_all_partition_step(source: StepSpec, partition_index: int) -> StepSpec:
    """One partition of the all-routes OCR corpus.

    The hash attrs mirror the production OCR step's -- the text is a function of the same model,
    prompt, and render settings -- plus the partition coordinates and the route override, so this
    corpus can never collide with the production OCR route's cache.
    """
    if not 0 <= partition_index < NUM_PARTITIONS:
        raise ValueError(f"partition_index must be in [0, {NUM_PARTITIONS}), got {partition_index}")
    return StepSpec(
        name=f"data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p{partition_index:02d}",
        deps=[source],
        hash_attrs={
            "model": MODEL,
            "prompt_digest": sha256(PROMPT_DOC2MD.encode("utf-8")).hexdigest()[:16],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_visual_tokens": RENDER_OPTIONS.max_visual_tokens,
            "max_render_dpi": RENDER_OPTIONS.max_render_dpi,
            "max_pages": RENDER_OPTIONS.max_pages,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            "routes": "all",
            "partition": [partition_index, NUM_PARTITIONS],
            "instances": INSTANCES_PER_PARTITION,
            "schema_version": 1,
        },
        fn=remote(
            partial(
                ocr_pdf_text,
                source_output_path=source.output_path,
                ocr_route_only=False,
                instances=INSTANCES_PER_PARTITION,
                partition=(partition_index, NUM_PARTITIONS),
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=int, required=True, help=f"partition index in [0, {NUM_PARTITIONS})")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    sender_tasks, max_workers = sender_fleet_size(INSTANCES_PER_PARTITION)
    logger.info(
        "Partition %d/%d: %d instances, %d sender tasks on %d workers",
        args.partition,
        NUM_PARTITIONS,
        INSTANCES_PER_PARTITION,
        sender_tasks,
        max_workers,
    )
    source = fetch_step(plan_step())
    StepRunner().run([ocr_all_partition_step(source, args.partition)])


if __name__ == "__main__":
    main()
