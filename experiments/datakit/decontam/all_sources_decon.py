# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global decontamination across every Datakit source.

DAG shape:

    EVAL_SOURCES (11 GCS dirs, raw paths)
        │
        ▼ one build_eval_bloom_step per eval source (datakit/bloom/<eval>)
        │
        ▼ one merge_eval_blooms_step over all per-eval blooms
        │     (datakit/bloom/_combined)
        │
        ▼ one decon_step per Datakit source, consuming the merged bloom
              (datakit/decon/<source>)

This means:

* Per-eval blooms are independently cacheable. Swapping or adding one eval
  invalidates that eval's bloom + the merge + all 104 corpus marks; the other
  10 per-eval blooms are reused.
* The combined bloom is built ONCE (bf.update merge) and consumed by all 104
  corpus mark steps via ``prebuilt_bloom_dir``. No redundant per-corpus
  bloom builds.

All 11 per-eval blooms must share ``estimated_doc_count`` and
``false_positive_rate`` — ``dupekit.Bloom.update`` requires identical sizing.

Eval reads cross-region (us-east1 → eu-west4) but happen only inside the
per-eval bloom build steps (small, once per eval). Source-corpus scans stay
in-region.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``):

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority production \\
        -- python experiments/decontamination/all_sources_decon.py
"""

import logging

from fray import ResourceConfig
from marin.datakit.decon import build_eval_bloom_step, decon_step, merge_eval_blooms_step
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import marin_temp_bucket
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


# Eleven lm-eval-harness eval sets prepared as decon-format dolma jsonl.gz
# and present on GCS today. The other 7 entries under
# gs://marin-us-east1/decontamination/ (ai2_arc, gsm8k, hellaswag, mmlu_pro,
# piqa, musr, lambada_openai) exist only as `.executor_info` stubs with no
# data files -- the original raw2json job from the (now-deleted)
# experiments/train_test_overlap/eval_datasets_overlap.py never completed for
# those subsets, so they're excluded here. Re-prepare via hf_dataset_to_jsonl
# with output_format="decontamination" to add them back.
EVAL_SOURCES: tuple[str, ...] = (
    "gs://marin-us-east1/decontamination/bbh-dolma-4a45e0",
    "gs://marin-us-east1/decontamination/boolq-5c05e1",
    "gs://marin-us-east1/decontamination/commonsense_qa-b467b9",
    "gs://marin-us-east1/decontamination/gpqa-0d1096",
    "gs://marin-us-east1/decontamination/humaneval-093159",
    "gs://marin-us-east1/decontamination/instruction_following-d07f60",
    "gs://marin-us-east1/decontamination/math-dolma-90f4d8",
    "gs://marin-us-east1/decontamination/mmlu-9fbdd5",
    "gs://marin-us-east1/decontamination/openbookqa-322b52",
    "gs://marin-us-east1/decontamination/truthful_qa-dolma-537c2f",
    "gs://marin-us-east1/decontamination/winograd_wsc-69924e",
)

# Bloom capacity -- unique ngram hashes the filter must hold. Measured by
# experiments/decontamination/count_docs.py: the 11 available eval sets yield
# ~327K unique hashes (~544K total inserts). 2e6 gives ~6x headroom, enough
# slack to add the 7 currently-stubbed evals later without resizing. Filter
# footprint ~30 MB per per-eval bloom at FPR=1e-9; ~330 MB total across the
# 11 per-eval blooms (TTL=7d temp storage).
ESTIMATED_DOC_COUNT = 2_000_000
FALSE_POSITIVE_RATE = 1e-9
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")


def _short_eval_name(path: str) -> str:
    """Strip the gs:// path + hash suffix down to a step-friendly name (e.g. ``mmlu``)."""
    leaf = path.rstrip("/").rsplit("/", 1)[-1]
    # Drop the trailing ``-<hex>`` hash. Two-segment names like ``bbh-dolma``
    # become ``bbh`` (single canonical name per eval family).
    base = leaf.rsplit("-", 1)[0]
    return base.replace("-dolma", "")


def build_decon_steps() -> list[StepSpec]:
    # Per-eval blooms: one StepSpec per eval source, all sized identically so
    # they can be bf.update-merged in the next step.
    per_eval_blooms: list[StepSpec] = [
        build_eval_bloom_step(
            name=f"datakit/bloom/{_short_eval_name(eval_src)}",
            eval_data_sources=[eval_src],
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
        )
        for eval_src in EVAL_SOURCES
    ]

    # Combined bloom (bit-OR merge + concatenated hash-index sidecar).
    combined_bloom = merge_eval_blooms_step(
        name="datakit/bloom/_combined",
        per_eval_bloom_steps=per_eval_blooms,
    )

    # Per-corpus decon steps, all consuming the same combined bloom.
    decon_output_prefix = marin_temp_bucket(ttl_days=7, prefix="rav/decon-all-sources-v0")
    return [
        decon_step(
            name=f"datakit/decon/{name}",
            normalized=src.normalized,
            prebuilt_bloom=combined_bloom,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
            worker_resources=WORKER_RESOURCES,
            output_path_prefix=decon_output_prefix,
        )
        for name, src in all_sources().items()
    ]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_decon_steps())
