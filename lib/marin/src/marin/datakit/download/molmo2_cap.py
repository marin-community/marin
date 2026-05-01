# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""allenai/Molmo2-Cap dataset download and transform.

Long-form video captioning dataset. Each row carries a paragraph-level
``merged_caption`` plus per-frame captions with timestamps. We render each
row as the merged caption followed by timestamp-labeled frame captions.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "allenai/Molmo2-Cap"
HF_REVISION = "b1d59eb"


def row_to_doc(row: dict) -> list[dict]:
    merged = row.get("merged_caption") or ""
    frame_timestamps = row.get("frame_timestamps") or []
    frame_captions = row.get("frame_captions") or []

    if not merged.strip():
        counters.increment("molmo2_cap/dropped")
        return []

    parts = [merged.strip()]
    for ts, cap in zip(frame_timestamps, frame_captions, strict=False):
        cap = (cap or "").strip()
        if not cap:
            continue
        parts.append(f"[Frame at {ts:.2f}s] {cap}")

    text = "\n\n".join(parts)

    counters.increment("molmo2_cap/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": HF_DATASET_ID,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="molmo2-cap-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_molmo2_cap_step() -> StepSpec:
    """Download and transform Molmo2-Cap into JSONL documents."""
    dl = download_hf_step(
        "raw/molmo2-cap",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=["data/train-*.parquet"],
    )

    return StepSpec(
        name="processed/molmo2-cap",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def molmo2_cap_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for molmo2-cap."""
    processed = download_molmo2_cap_step()
    return (
        processed,
        normalize_step(name="normalized/molmo2-cap", download=processed),
    )
