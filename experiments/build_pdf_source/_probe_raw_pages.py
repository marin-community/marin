# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- capture raw OCR responses to settle two questions about the surviving fences.

DELETE once the answers are recorded. Nothing in the pipeline imports this.

After :func:`~experiments.build_pdf_source.ocr_extract.client.unwrap_markdown_fence` landed, 3.1% of
pages still carried a ```markdown fence. Two different things are hiding in that number and the
stored corpus cannot tell them apart, because what it holds is post-unwrap and post-boilerplate:

* **859 pages had an unbalanced fence** -- an opener with no closer. That is what a page cut off at
  ``max_tokens`` looks like, but it is also what a model that simply never closes the fence looks
  like. Only ``finish_reason`` distinguishes them, and the corpus does not record it.
* **148 pages had a balanced fence** that the unwrapper should have stripped and did not. The
  hypothesis is a preamble line ahead of the fence ("Here is the converted markdown:"), which stops
  the unwrap because the fence is no longer on the first line, and which the boilerplate pass then
  removes -- leaving the fence at line 0 in the stored text and destroying the evidence.

So this stores the **raw** ``message.content`` alongside ``finish_reason`` and the token count, with
no unwrapping and no boilerplate, and reports what fraction of fenced pages fall into each bucket.

Deliberately does not import :mod:`~experiments.build_pdf_source.ocr_extract.fleet`: its ``MODEL``
is switched between comparison runs, and this probe has to pin the model it is characterising.

One GPU, a few thousand pages, a few minutes::

    iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name ocr-raw-page-probe \\
        -- python -m experiments.build_pdf_source._probe_raw_pages
"""

import logging
import time
from collections import Counter
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.inference.config import (
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference.iris import remote_inference
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step, routing_keys
from experiments.build_pdf_source.common import PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.ocr_extract.client import (
    DEFAULT_MAX_TOKENS,
    MIN_PIXELS,
    PROMPT_DOC2MD,
    VISUAL_TOKEN_PIXELS,
    unwrap_markdown_fence,
)
from experiments.build_pdf_source.ocr_extract.render import RenderOptions, iter_rendered_pages, open_pdf
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# Pinned here rather than imported: see the module docstring.
MODEL = "infly/Infinity-Parser2-Flash"

TARGET_PAGES = 3000
_SOURCE_SHARDS = 8
_REQUEST_THREADS = 96
_FLASHINFER_PACKAGES = ("flashinfer-cubin==0.6.13", "flashinfer-jit-cache==0.6.13")
_FLASHINFER_INDEX = "https://flashinfer.ai/whl/cu130/"
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]
_RESOURCES = ResourceConfig(cpu=16, ram="48g", disk="32g")


class RawPageReport(BaseModel):
    version: str = "v1"
    model: str
    raw_path: str
    pages: int
    truncated: int
    fenced_pages: int
    fenced_and_truncated: int
    fenced_not_truncated: int
    unwrap_strips: int
    unwrap_misses: int
    miss_first_lines: list[str]
    finish_reasons: dict[str, int]
    repetition_loop_pages: int


_RAW_SCHEMA = pa.schema(
    [
        pa.field("url", pa.string()),
        pa.field("page_index", pa.int32()),
        pa.field("finish_reason", pa.string()),
        pa.field("completion_tokens", pa.int32()),
        pa.field("raw_text", pa.string()),
    ]
)


def _fenced(text: str) -> bool:
    return "```markdown" in text


def probe(output_path: str, source_output_path: str, classification_output_path: str) -> RawPageReport:
    """OCR a few thousand pages and store what the model actually returned."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    import httpx  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    keys = routing_keys(classification.main_output_dir, needs_ocr=True)

    filesystem, path = url_to_fs(source.main_output_dir)
    shards = sorted(f"s3://{shard}" for shard in filesystem.glob(f"{path}/*.parquet"))[:_SOURCE_SHARDS]

    options = RenderOptions()
    rendered: list[tuple[str, int, str]] = []  # (url, page_index, data_uri)
    for shard in shards:
        if len(rendered) >= TARGET_PAGES:
            break
        shard_fs, shard_path = url_to_fs(shard)
        with shard_fs.open(shard_path, "rb") as stream:
            table = pq.read_table(stream, columns=_SOURCE_COLUMNS)
        for row in table.to_pylist():
            if len(rendered) >= TARGET_PAGES:
                break
            if (row["warc_filename"], row["warc_record_offset"]) not in keys:
                continue
            try:
                with open_pdf(row["pdf"]) as document:
                    for page in iter_rendered_pages(document, options):
                        rendered.append((row["url"], page.page_index, page.data_uri))
                        if len(rendered) >= TARGET_PAGES:
                            break
            except Exception:
                logger.warning("Could not render %s", row["url"], exc_info=True)
    logger.info("Rendered %d pages from %d shards", len(rendered), len(shards))

    config = RemoteInferenceConfig(
        model=ServedModelConfig(weights=MODEL, max_model_len=24_576, tensor_parallel_size=1),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            startup_timeout_seconds=3600,
            max_num_seqs=1024,
            max_num_batched_tokens=131_072,
            extra_args=(
                "--gdn-prefill-backend",
                "flashinfer",
                "--reasoning-parser",
                "qwen3",
                "--mm-processor-cache-type",
                "shm",
                "--api-server-count",
                "2",
            ),
            uv_with_packages=_FLASHINFER_PACKAGES,
            uv_extra_index_urls=(_FLASHINFER_INDEX,),
        ),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu(
                "GB200", count=1, cpu=32, ram="160g", disk="300g", regions=[ANY_REGION]
            ),
            worker_environment=create_environment(),
            endpoint_ready_timeout_seconds=3600.0,
        ),
        instances=1,
    )

    def one(item: tuple[str, int, str]) -> dict:
        url, page_index, data_uri = item
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                            "max_pixels": options.max_visual_tokens * VISUAL_TOKEN_PIXELS,
                            "min_pixels": MIN_PIXELS,
                        },
                        {"type": "text", "text": PROMPT_DOC2MD},
                    ],
                }
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            timeout=900.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = response.choices[0]
        return {
            "url": url,
            "page_index": page_index,
            "finish_reason": choice.finish_reason or "",
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "raw_text": choice.message.content or "",
        }

    with remote_inference(config) as session:
        model_id = session.model.endpoint.model
        limits = httpx.Limits(max_connections=_REQUEST_THREADS, max_keepalive_connections=_REQUEST_THREADS)
        client = OpenAI(
            api_key="EMPTY",
            base_url=session.model.endpoint.base_url,
            timeout=900.0,
            max_retries=2,
            http_client=httpx.Client(limits=limits, timeout=900.0),
        )
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=_REQUEST_THREADS) as pool:
            results = [r for r in pool.map(one, rendered) if r is not None]
        logger.info("OCR'd %d pages in %.0fs", len(results), time.monotonic() - started)

    raw_path = prefix_join(output_path, "raw_pages.parquet")
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(results, schema=_RAW_SCHEMA), buffer)
    StoragePath(raw_path).write_bytes(buffer.getvalue().to_pybytes())

    fenced = [r for r in results if _fenced(r["raw_text"])]
    misses = [r for r in fenced if unwrap_markdown_fence(r["raw_text"]) == r["raw_text"]]
    report = RawPageReport(
        model=MODEL,
        raw_path=raw_path,
        pages=len(results),
        truncated=sum(1 for r in results if r["finish_reason"] == "length"),
        fenced_pages=len(fenced),
        fenced_and_truncated=sum(1 for r in fenced if r["finish_reason"] == "length"),
        fenced_not_truncated=sum(1 for r in fenced if r["finish_reason"] != "length"),
        unwrap_strips=len(fenced) - len(misses),
        unwrap_misses=len(misses),
        # The first line of a page the unwrapper declined is the whole question: if it is prose, a
        # preamble is what blocks the strip.
        miss_first_lines=[r["raw_text"].strip().split("\n")[0][:120] for r in misses[:25]],
        finish_reasons=dict(Counter(r["finish_reason"] for r in results)),
        repetition_loop_pages=sum(1 for r in results if r["raw_text"].count("```markdown") > 20),
    )
    logger.info("RAW PAGE PROBE %s", report.model_dump_json(indent=2))
    print("RAW_PAGE_RESULT " + report.model_dump_json(), flush=True)
    StoragePath(prefix_join(output_path, "raw-page-report.json")).write_bytes(
        report.model_dump_json(indent=2).encode("utf-8")
    )
    return report


def probe_step(source: StepSpec, classification: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/ocr_raw_pages",
        deps=[source, classification],
        hash_attrs={"model": MODEL, "pages": TARGET_PAGES, "attempt": 1},
        fn=remote(
            partial(
                probe,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
            ),
            resources=_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    fetch = fetch_step(plan_step())
    StepRunner().run([probe_step(fetch, classify_step(fetch, model_step()))])


if __name__ == "__main__":
    main()
