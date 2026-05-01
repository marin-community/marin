# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nyuuzyou/svgfind dataset download and transform.

The HF dataset stores rows as zstd-compressed JSONL, sharded by license. Each
row carries a title, a data-pack name (which encodes icon style — e.g.
``ui-outlines``, ``basic-ui-solid``), a list of search tags, and the raw SVG
markup as a string.

We render each row to a single SFT document where a short description prefix
conditions the model and the raw SVG markup is the target completion::

    Create an SVG which matches the following description.
    Title: messaging
    Data Pack: ui-outlines
    Tags: messaging, messaging app, chat app, chat

    <svg ...>...</svg>

Only the ``CREATIVECOMMONS`` shards are wired up; the ``PUBLICDOMAIN`` shard
can be added later if needed.
"""

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_jsonl

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "nyuuzyou/svgfind"
HF_REVISION = "4d29c79273411f989625fa9f06419b9753ec5e12"

CC_GLOBS = ["svgfind-CREATIVECOMMONS_*.jsonl.zst"]


def svgfind_row_to_doc(row: dict) -> list[dict]:
    title = row.get("title") or ""
    svg = row.get("svg_content") or ""
    if not title or not svg:
        counters.increment("svgfind/cc/dropped")
        return []

    tags = ", ".join(row.get("tags") or [])
    data_pack = row.get("data_pack") or ""
    text = (
        "Create an SVG which matches the following description.\n"
        f"Title: {title}\n"
        f"Data Pack: {data_pack}\n"
        f"Tags: {tags}\n\n"
        f"{svg}"
    )

    counters.increment("svgfind/cc/kept")
    return [
        {
            "id": row["id"],
            "text": text,
            "source": "nyuuzyou/svgfind/creativecommons",
        }
    ]


def transform_svgfind_creativecommons(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/svgfind-CREATIVECOMMONS_*.jsonl.zst")
        .flat_map(load_jsonl)
        .flat_map(svgfind_row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="svgfind-cc-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def download_svgfind_creativecommons_step() -> StepSpec:
    """Download and render svgfind's CREATIVECOMMONS shards as SFT docs."""
    dl = download_hf_step(
        "raw/svgfind-creativecommons",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=CC_GLOBS,
    )
    return StepSpec(
        name="processed/svgfind-creativecommons",
        deps=[dl],
        fn=lambda output_path: transform_svgfind_creativecommons(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def svgfind_creativecommons_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for svgfind CC."""
    processed = download_svgfind_creativecommons_step()
    return (
        processed,
        normalize_step(name="normalized/svgfind-creativecommons", download=processed),
    )
