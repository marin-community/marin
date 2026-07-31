# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Infinity-Parser2-Flash OCR on GB200s via marin.inference.

Runs as an Iris entrypoint job (CPU only): renders focus-crawl PDF pages with
PyMuPDF, starts a vLLM fleet on GB200s through :func:`marin.inference.iris.
remote_inference`, then drives a closed-loop doc2md load test and prints one
``BENCH_RESULT {json}`` line for scraping from job logs.

Page size is controlled by a *visual-token budget* (``--max-visual-tokens``), not
a DPI target: under DPI, megapixels-per-page — and so throughput — is a function
of the page-size mix that happened to land in the sampled shard, which makes runs
against different shards incomparable. A token budget holds per-page cost roughly
constant instead. Effective DPI becomes the quality diagnostic: it is recorded
per page and summarised as a distribution, since a budget that is fine for Letter
pages can silently render large-format pages below legibility.

Payloads are pre-rendered before the timer starts, so the GPU measurement is
independent of client CPU speed; ``--cpu-bench-only`` measures the render side
(pages/sec/core) for the CPU:GPU ratio without touching a GPU.

Submit (smoke)::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --cpu 8 --memory 16GB --disk 50GB --enable-extra-resources \\
        --extra datakit --priority interactive --job-name ocr-b200-smoke --no-wait \\
        -- python -m experiments.b200_ocr.bench_infinity_parser --smoke
"""

import argparse
import base64
import json
import logging
import math
import multiprocessing
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import pyarrow.parquet as pq
import pymupdf
from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.inference.config import (
    BrokerConfig,
    InferenceProxyConfig,
    InferenceWorkerConfig,
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference.iris import remote_inference
from openai import OpenAI
from rigging.filesystem import url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

MODEL = "infly/Infinity-Parser2-Flash"
DEFAULT_PARQUET = (
    "s3://marin-us-east-02a/marin/data/datakit/raw/common_crawl_focus_2026_22_pdf_e70aa547/"
    "outputs/main/part-00000-of-01773.parquet"
)

# Upstream client defaults (infinity_parser2/utils/image.py): patch multiple 32,
# ~4096x4096 max. The model was validated against smart_resize'd inputs.
RESIZE_FACTOR = 32
MIN_PIXELS = 2048
MAX_PIXELS = 16777216

# One visual token per RESIZE_FACTOR^2 pixels: this arch is patch-16 with 2x2
# merging, so 32x32. NOT the 28 of Qwen2.5-VL (patch-14) that olmOCR-lineage
# pipelines use — reusing 28 here mis-sizes every page by (32/28)^2 ~= 1.31x.
VISUAL_TOKEN_PIXELS = RESIZE_FACTOR * RESIZE_FACTOR
DEFAULT_MAX_VISUAL_TOKENS = 2048

# Optional opt-in to the olmOCR/finepdfs convention: an upscale-only floor on the
# long side, in PDF points (1pt = 1/72in). 1280 is not arbitrary there — the
# 28-alignment of that lineage turns it into a 1288px long side, olmOCR-2's
# documented target_longest_image_dim. It does not transfer to this arch (factor
# 32), so it is off by default and exists for comparison runs only.
LEGACY_LONGEST_SIDE = 1280

# Ceiling on upscaling small pages to fill the budget. Past ~300 DPI there is no
# more glyph detail to recover, only tokens to burn.
DEFAULT_MAX_RENDER_DPI = 300.0

# ~10pt body text at 100 DPI is ~14px/em, about the floor for reliable VLM
# reading. Pages below this are counted, not resized: they want tiling, and
# raising the global budget to rescue them overpays for the ~95% that are fine.
DEFAULT_LEGIBILITY_FLOOR_DPI = 100.0

# Exact doc2md prompt from infinity_parser2/prompts.py (PROMPT_DOC2MD). Do not edit:
# the prompt string is part of the model's validated input distribution.
PROMPT_DOC2MD = """
You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with $ $. For example: This is an inline formula $E = mc^2$
- Enclose block formulas with $$ $$. For example: $$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

3. Table Processing:
- Convert tables to HTML format.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen-VL input sizing (ported from qwen_vl_utils.vision_process.smart_resize).

    Rounds each side to a multiple of ``factor`` and rescales so the pixel count
    lands in ``[min_pixels, max_pixels]``.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be < 200, got {height}x{width}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def target_dims(rect, max_visual_tokens: int, max_render_dpi: float, longest_side: int | None) -> tuple[int, int]:
    """Page dims in points -> final pixel dims under a visual-token budget.

    The page is scaled to *fill* the budget, so the budget is the control that
    actually moves page size. A fixed long-side floor cannot do that job: it pins
    Letter/A4 at one size and leaves the budget slack, so sweeping the budget
    barely moves throughput for the bulk of a crawl corpus.

    ``max_render_dpi`` stops small pages (business cards, cropped scans) from
    being upscaled to absurd resolutions just to spend the budget; they simply
    come in under it. ``longest_side`` opts back into the olmOCR/finepdfs
    convention — an upscale-only floor in points — for comparison runs.

    Returns ``(height, width)``.
    """
    width, height = rect.width, rect.height
    if longest_side is not None:
        scale = longest_side / max(width, height)
        if scale > 1:
            width, height = width * scale, height * scale
    else:
        budget_pixels = max_visual_tokens * VISUAL_TOKEN_PIXELS
        scale = min(math.sqrt(budget_pixels / (width * height)), max_render_dpi / 72.0)
        width, height = width * scale, height * scale
    return smart_resize(
        round(height),
        round(width),
        RESIZE_FACTOR,
        MIN_PIXELS,
        max_visual_tokens * VISUAL_TOKEN_PIXELS,
    )


def effective_dpi(pixels: int, rect) -> float:
    """Geometric-mean DPI actually achieved for a page.

    Per-axis DPI differs slightly because the render matrix is non-uniform (it
    hits the aligned dims exactly rather than letter-boxing), so compare areas.
    """
    points_area = rect.width * rect.height
    return 72.0 * math.sqrt(pixels / points_area) if points_area > 0 else 0.0


@dataclass
class PagePayload:
    """One rendered, resized, base64-encoded PDF page plus its provenance."""

    data_uri: str
    pdf_index: int
    page_index: int
    pixels: int
    encoded_bytes: int
    dpi: float


def load_pdf_blobs(parquet_path: str, count: int) -> list[bytes]:
    """Read the first ``count`` PDF byte blobs from a fetched-sample shard."""
    fs, resolved = url_to_fs(parquet_path)
    blobs: list[bytes] = []
    file = pq.ParquetFile(fs.open(resolved, "rb"))
    for batch in file.iter_batches(batch_size=32, columns=["pdf"]):
        for value in batch.column("pdf"):
            blobs.append(value.as_py())
            if len(blobs) >= count:
                return blobs
    return blobs


def render_encoded_pages(
    blob: bytes, max_visual_tokens: int, max_render_dpi: float, longest_side: int | None, max_pages: int
) -> list[tuple[str, int, int, float]]:
    """Render up to ``max_pages`` pages straight to smart_resize'd base64 PNGs.

    The page is rendered once, at the final resolution: the target comes from the
    token budget, and the PyMuPDF matrix scales directly to it (so there is no
    decode → resize → re-encode round trip, and no PIL dependency). Returns
    ``(data_uri, pixels, encoded_bytes, effective_dpi)`` per page; parse/render
    failures skip the page (or the whole PDF) silently — adversarial crawl PDFs
    fail arbitrarily deep in MuPDF.
    """
    try:
        doc = pymupdf.open(stream=blob, filetype="pdf")
    except Exception:
        return []
    pages: list[tuple[str, int, int, float]] = []
    try:
        for page_number in range(min(len(doc), max_pages)):
            try:
                page = doc[page_number]
                rect = page.rect
                if rect.width < 1 or rect.height < 1:
                    continue
                height, width = target_dims(rect, max_visual_tokens, max_render_dpi, longest_side)
                matrix = pymupdf.Matrix(width / rect.width, height / rect.height)
                png_bytes = page.get_pixmap(matrix=matrix).tobytes("png")
                encoded = base64.b64encode(png_bytes).decode()
                pixels = height * width
                pages.append((f"data:image/png;base64,{encoded}", pixels, len(encoded), effective_dpi(pixels, rect)))
            except Exception:
                continue
    finally:
        doc.close()
    return pages


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already-sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def dpi_distribution(payloads: list[PagePayload], floor_dpi: float) -> dict:
    """Effective-DPI spread for a pool, plus the share below the legibility floor.

    The mean alone hides the failure mode this instrumentation exists for: under a
    token budget every page costs the same, so a large-format page is quietly
    rendered at a fraction of a Letter page's DPI rather than costing more.
    """
    if not payloads:
        return {}
    ordered = sorted(p.dpi for p in payloads)
    below = [p for p in payloads if p.dpi < floor_dpi]
    return {
        "dpi_min": round(ordered[0], 1),
        "dpi_p05": round(_percentile(ordered, 0.05), 1),
        "dpi_p50": round(_percentile(ordered, 0.50), 1),
        "dpi_p95": round(_percentile(ordered, 0.95), 1),
        "dpi_max": round(ordered[-1], 1),
        "floor_dpi": floor_dpi,
        "pages_below_floor": len(below),
        "frac_below_floor": round(len(below) / len(payloads), 4),
        # Worst offenders are the tiling candidates; identify them by provenance.
        "worst_pages": [
            {"pdf": p.pdf_index, "page": p.page_index, "dpi": round(p.dpi, 1)}
            for p in sorted(below, key=lambda p: p.dpi)[:5]
        ],
    }


def prepare_payloads(
    blobs: list[bytes],
    max_visual_tokens: int,
    max_render_dpi: float,
    longest_side: int | None,
    max_pages_per_pdf: int,
    fetch_seconds: float,
    floor_dpi: float,
) -> tuple[list[PagePayload], dict]:
    """Render and encode the payload pool, returning payloads + CPU-side stats."""
    payloads: list[PagePayload] = []
    render_seconds = 0.0
    failed_pdfs = 0
    for pdf_index, blob in enumerate(blobs):
        start = time.monotonic()
        pages = render_encoded_pages(blob, max_visual_tokens, max_render_dpi, longest_side, max_pages_per_pdf)
        render_seconds += time.monotonic() - start
        if not pages:
            failed_pdfs += 1
            continue
        for page_index, (data_uri, pixels, encoded_bytes, dpi) in enumerate(pages):
            payloads.append(
                PagePayload(
                    data_uri=data_uri,
                    pdf_index=pdf_index,
                    page_index=page_index,
                    pixels=pixels,
                    encoded_bytes=encoded_bytes,
                    dpi=dpi,
                )
            )
    pixel_counts = [p.pixels for p in payloads]
    cpu_stats = {
        "num_pdfs": len(blobs),
        "failed_pdfs": failed_pdfs,
        "num_pages": len(payloads),
        "fetch_seconds": round(fetch_seconds, 2),
        "render_seconds": round(render_seconds, 2),
        # Single-threaded driver loop, so wall time here is CPU-core time.
        "cpu_pages_per_core_second": round(len(payloads) / render_seconds, 3) if payloads and render_seconds else 0.0,
        "mean_encoded_mb": round(statistics.mean(p.encoded_bytes for p in payloads) / 1e6, 2) if payloads else 0.0,
        "mean_megapixels": round(statistics.mean(pixel_counts) / 1e6, 2) if payloads else 0.0,
        # The budget is the control variable; this is how close pages land to it.
        "mean_visual_tokens": round(statistics.mean(pixel_counts) / VISUAL_TOKEN_PIXELS) if payloads else 0,
        "max_visual_tokens_observed": round(max(pixel_counts) / VISUAL_TOKEN_PIXELS) if payloads else 0,
        **dpi_distribution(payloads, floor_dpi),
    }
    return payloads, cpu_stats


# Prebuilt FlashInfer kernel artifacts: CoreWeave runtime images have no nvcc, so
# FlashInfer's JIT path cannot compile. flashinfer-cubin (PyPI) ships device
# cubins; flashinfer-jit-cache (flashinfer.ai index, per CUDA version) ships the
# prebuilt host glue. Both MUST match the flashinfer-python version the pinned
# vLLM depends on (0.6.13 for vllm==0.25.1) — flashinfer hard-fails on skew.
FLASHINFER_PACKAGES = ("flashinfer-cubin==0.6.13", "flashinfer-jit-cache==0.6.13")
FLASHINFER_INDEX = "https://flashinfer.ai/whl/cu130/"


def vllm_extra_args(args: argparse.Namespace) -> tuple[str, ...]:
    # --trust-remote-code is already set by the marin vLLM backend; passing it
    # again logs a duplicate-key warning. Qwen3.5 is a gated-delta-net hybrid:
    # its GDN prefill kernel comes from FlashInfer (JIT — needs the prebuilt
    # artifact packages above) or Triton (always available).
    extra = [
        "--reasoning-parser",
        "qwen3",
        "--mm-processor-cache-type",
        "shm",
        "--gdn-prefill-backend",
        args.gdn_backend,
    ]
    if args.prefix_caching:
        # On this hybrid arch, prefix caching forces the experimental 'align'
        # Mamba cache mode, which snapshots recurrent state along the sequence
        # and inflates per-request state memory. Real crawl pages are unique,
        # so prefix caching buys nothing in production anyway.
        extra += ["--enable-prefix-caching"]
    if args.attention_backend:
        # By default vLLM auto-selects FLASHINFER attention on Blackwell, which
        # the prebuilt artifact packages make work without nvcc — and which the
        # backend A/B measured ~4-6% faster than FLASH_ATTN at saturation. The
        # override exists for A/B runs and as a fallback (FLASH_ATTN ships
        # precompiled in the vLLM wheel) if the artifact pin ever breaks.
        extra += ["--attention-backend", args.attention_backend]
    if args.api_server_count > 1:
        # A single API-server process bottlenecks ingest: multimodal
        # preprocessing runs there before requests reach the scheduler.
        extra += ["--api-server-count", str(args.api_server_count)]
    if args.mm_cache_gb is not None:
        # The shm mm-processor cache holds in-flight processed image tensors
        # (~50MB per 8.5MP page); the ~4GiB default caps scheduled multimodal
        # requests near 100.
        extra += ["--mm-processor-cache-gb", str(args.mm_cache_gb)]
    if args.tensor_parallel > 1:
        extra += ["--mm-encoder-tp-mode", "data"]
    return tuple(extra)


def build_inference_config(args: argparse.Namespace) -> RemoteInferenceConfig:
    broker = None
    if args.instances > 1:
        broker = BrokerConfig(
            worker=InferenceWorkerConfig(
                max_in_flight=args.max_in_flight,
                request_timeout_seconds=float(args.request_timeout),
            ),
            request_lease_timeout_seconds=float(args.request_timeout + 120),
            proxy=InferenceProxyConfig(
                request_timeout_seconds=float(args.request_timeout + 240),
                max_pending_requests=max(256, max(args.concurrency) * 2),
            ),
        )
    return RemoteInferenceConfig(
        model=ServedModelConfig(
            weights=MODEL,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel,
        ),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            startup_timeout_seconds=3600,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            extra_args=vllm_extra_args(args),
            # Always ship the prebuilt FlashInfer artifacts: both defaults need
            # them (FLASHINFER attention on Blackwell and FlashInfer GDN
            # prefill), and they are harmless under the fallback backends.
            uv_with_packages=FLASHINFER_PACKAGES,
            uv_extra_index_urls=(FLASHINFER_INDEX,),
        ),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu(
                "GB200",
                count=args.tensor_parallel,
                cpu=args.gpu_worker_cpu * args.tensor_parallel,
                ram=f"{args.gpu_worker_ram_gb * args.tensor_parallel}g",
                disk="300g",
                regions=[ANY_REGION],
            ),
            worker_environment=create_environment(),
            endpoint_ready_timeout_seconds=3600.0,
        ),
        instances=args.instances,
        broker=broker,
    )


# Payload pool shared with forked client shards via copy-on-write (Linux fork);
# pickling ~370MB of data URIs to every shard would dwarf the test itself.
_FORK_POOL: list[PagePayload] = []


def _one_request(
    client: OpenAI, model_id: str, payload: PagePayload, args: argparse.Namespace, max_visual_tokens: int
) -> dict:
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": payload.data_uri},
                        # Restate the client-side budget so the server's own
                        # smart_resize cannot silently re-size the page: the
                        # model's preprocessor_config defaults are unrelated to
                        # the swept value, and a lower server cap would make the
                        # recorded sweep axis a fiction.
                        "max_pixels": max_visual_tokens * VISUAL_TOKEN_PIXELS,
                        "min_pixels": MIN_PIXELS,
                    },
                    {"type": "text", "text": PROMPT_DOC2MD},
                ],
            }
        ],
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        timeout=args.request_timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.monotonic() - start
    usage = response.usage
    return {
        "seconds": elapsed,
        "completion_tokens": usage.completion_tokens if usage else None,
        "text": response.choices[0].message.content or "",
    }


def _make_client(base_url: str, timeout: float, concurrency: int) -> OpenAI:
    """OpenAI client with an httpx pool sized for the load test.

    httpx defaults to 100 connections per client, which silently caps in-flight
    requests below the test's concurrency (it pinned every single-process run's
    ``num_requests_running`` at ~99).
    """
    limits = httpx.Limits(max_connections=max(256, concurrency * 2), max_keepalive_connections=256)
    return OpenAI(
        api_key="EMPTY",
        base_url=base_url,
        timeout=timeout,
        http_client=httpx.Client(limits=limits, timeout=timeout),
    )


def _client_shard(
    base_url: str,
    model_id: str,
    offset: int,
    num_requests: int,
    concurrency: int,
    args: argparse.Namespace,
    max_visual_tokens: int,
) -> dict:
    """One client process's share of the load: its own OpenAI client + thread pool."""
    client = _make_client(base_url, args.request_timeout, concurrency)
    results: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _one_request, client, model_id, _FORK_POOL[(offset + i) % len(_FORK_POOL)], args, max_visual_tokens
            )
            for i in range(num_requests)
        ]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(repr(error)[:300])
    return {
        "seconds": [r["seconds"] for r in results],
        "completion_tokens": [r["completion_tokens"] for r in results if r["completion_tokens"] is not None],
        "errors": errors,
        "sample": results[0]["text"][:600] if results else "",
    }


def run_load_test(
    base_url: str,
    model_id: str,
    payloads: list[PagePayload],
    args: argparse.Namespace,
    concurrency: int,
    max_visual_tokens: int,
) -> dict:
    """Closed-loop load test: ``concurrency`` in-flight requests until ``--num-requests`` done.

    ``--client-processes`` forks that many sender processes, each with
    ``concurrency / N`` threads: a single Python process cannot push enough
    ~1.4MB requests to saturate a tuned server (the engine-stats lines show it
    running 20-40 of 512 seqs with an empty queue).
    """
    client = _make_client(base_url, args.request_timeout, concurrency)
    # Warmup (excluded from the measurement; lets vLLM compile/capture graphs).
    for warmup_index in range(min(args.warmup, len(payloads))):
        _one_request(client, model_id, payloads[warmup_index], args, max_visual_tokens)
    logger.info("warmup done (%d requests)", min(args.warmup, len(payloads)))

    processes = max(1, args.client_processes)
    global _FORK_POOL
    _FORK_POOL = payloads
    shard_requests = [args.num_requests // processes] * processes
    shard_requests[0] += args.num_requests % processes
    per_shard_concurrency = max(1, concurrency // processes)

    start = time.monotonic()
    if processes == 1:
        shards = [_client_shard(base_url, model_id, 0, args.num_requests, concurrency, args, max_visual_tokens)]
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=processes, mp_context=context) as pool:
            futures = [
                pool.submit(
                    _client_shard,
                    base_url,
                    model_id,
                    index * 1000,
                    count,
                    per_shard_concurrency,
                    args,
                    max_visual_tokens,
                )
                for index, count in enumerate(shard_requests)
            ]
            shards = [future.result() for future in futures]
    wall = time.monotonic() - start

    latencies = sorted(second for shard in shards for second in shard["seconds"])
    completion_tokens = [token for shard in shards for token in shard["completion_tokens"]]
    errors = [error for shard in shards for error in shard["errors"]]
    return {
        "requests_ok": len(latencies),
        "requests_failed": len(errors),
        "errors": errors[:5],
        "wall_seconds": round(wall, 2),
        "pages_per_second": round(len(latencies) / wall, 3) if wall > 0 else 0.0,
        "latency_p50": round(latencies[len(latencies) // 2], 2) if latencies else None,
        "latency_p95": round(latencies[int(len(latencies) * 0.95)], 2) if latencies else None,
        "mean_completion_tokens": round(statistics.mean(completion_tokens), 1) if completion_tokens else None,
        "completion_tokens_per_second": round(sum(completion_tokens) / wall, 1) if completion_tokens else None,
        "sample_output": shards[0]["sample"] if shards else "",
    }


def _int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def write_results(url: str, results: list[dict]) -> None:
    """Rewrite the full results file after every point.

    Job log retention is shorter than a long sweep, so BENCH_RESULT lines
    scraped from logs get lost; this file is the durable record. Object stores
    cannot append, and the file is tiny, so rewriting it wholesale is fine.
    """
    fs, path = url_to_fs(url)
    with fs.open(path, "w") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--instances", type=int, default=1, help="vLLM instances (1 GPU each unless --tensor-parallel)")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-in-flight", type=int, default=64, help="broker per-instance in-flight cap")
    parser.add_argument(
        "--max-visual-tokens",
        type=_int_list,
        default=[DEFAULT_MAX_VISUAL_TOKENS],
        help=f"comma list; per-page budget at {VISUAL_TOKEN_PIXELS} px/token. One payload pool per value",
    )
    parser.add_argument(
        "--max-render-dpi",
        type=float,
        default=DEFAULT_MAX_RENDER_DPI,
        help="ceiling on upscaling small pages to fill the budget",
    )
    parser.add_argument(
        "--legacy-longest-side",
        type=int,
        nargs="?",
        const=LEGACY_LONGEST_SIDE,
        default=None,
        help=f"opt into the olmOCR/finepdfs long-side floor in points (bare flag = {LEGACY_LONGEST_SIDE}); "
        "pins page size and makes the budget sweep near-flat, so it is for comparison runs only",
    )
    parser.add_argument(
        "--legibility-floor-dpi",
        type=float,
        default=DEFAULT_LEGIBILITY_FLOOR_DPI,
        help="pages rendered below this effective DPI are counted as tiling candidates",
    )
    parser.add_argument("--num-pdfs", type=int, default=48)
    parser.add_argument("--max-pages-per-pdf", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--concurrency", type=_int_list, default=[32], help="comma list; one load test per value")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument(
        "--gdn-backend",
        choices=("triton", "flashinfer"),
        default="flashinfer",
        help="GDN prefill kernel backend; flashinfer (default) pulls prebuilt cubin/jit-cache packages",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="override vLLM's auto-selected attention backend (FLASHINFER on Blackwell, the measured-fastest "
        "default); FLASH_ATTN is the no-prebuilt-artifacts fallback",
    )
    parser.add_argument(
        "--gpu-worker-cpu",
        type=int,
        default=16,
        help="CPU cores per GPU for the inference worker; the server-side Qwen-VL image preprocessor is CPU-hungry",
    )
    parser.add_argument("--client-processes", type=int, default=1, help="forked sender processes per load test")
    parser.add_argument(
        "--gpu-worker-ram-gb",
        type=int,
        default=160,
        help="pod RAM per GPU; concurrent multimodal preprocessing in the API servers is the peak consumer",
    )
    parser.add_argument("--api-server-count", type=int, default=1, help="vLLM API-server processes per instance")
    parser.add_argument("--mm-cache-gb", type=int, default=None, help="shm mm-processor cache size in GiB")
    parser.add_argument(
        "--prefix-caching", action=argparse.BooleanOptionalAction, default=False, help="enable vLLM prefix caching"
    )
    parser.add_argument(
        "--results-jsonl",
        default=None,
        help="fsspec URL; the accumulated BENCH_RESULT records are rewritten there after every point",
    )
    parser.add_argument("--cpu-bench-only", action="store_true", help="measure render/encode only; no GPUs")
    parser.add_argument("--smoke", action="store_true", help="1 instance, 2 pdfs, 4 requests, 1 page each")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    # No-op on cluster pods (the runtime exports FSSPEC_S3); makes the sample
    # shard and --results-jsonl reachable from a dev box for --cpu-bench-only.
    configure_coreweave_s3()
    over_budget = [t for t in args.max_visual_tokens if t * VISUAL_TOKEN_PIXELS > MAX_PIXELS]
    if over_budget:
        raise SystemExit(
            f"--max-visual-tokens {over_budget} exceeds the upstream client ceiling of "
            f"{MAX_PIXELS // VISUAL_TOKEN_PIXELS} tokens ({MAX_PIXELS} px); the model was not validated above it"
        )
    if args.smoke:
        args.instances = 1
        args.num_pdfs = 2
        args.max_pages_per_pdf = 2
        args.num_requests = 4
        args.concurrency = [2]
        args.warmup = 0

    fetch_start = time.monotonic()
    blobs = load_pdf_blobs(args.parquet, args.num_pdfs)
    fetch_seconds = time.monotonic() - fetch_start
    logger.info("loaded %d pdf blobs (%.1f MB) in %.1fs", len(blobs), sum(map(len, blobs)) / 1e6, fetch_seconds)

    # One payload pool per budget point; rendered up front so the GPU
    # measurements are independent of client CPU speed.
    pools: dict[int, tuple[list[PagePayload], dict]] = {}
    for max_visual_tokens in args.max_visual_tokens:
        payloads, cpu_stats = prepare_payloads(
            blobs,
            max_visual_tokens,
            args.max_render_dpi,
            args.legacy_longest_side,
            args.max_pages_per_pdf,
            fetch_seconds,
            args.legibility_floor_dpi,
        )
        logger.info("payload pool max_visual_tokens=%d: %s", max_visual_tokens, json.dumps(cpu_stats))
        if cpu_stats.get("frac_below_floor"):
            logger.warning(
                "%d/%d pages (%.1f%%) render below %.0f DPI at budget %d — tiling candidates, "
                "not a reason to raise the global budget",
                cpu_stats["pages_below_floor"],
                cpu_stats["num_pages"],
                100 * cpu_stats["frac_below_floor"],
                args.legibility_floor_dpi,
                max_visual_tokens,
            )
        pools[max_visual_tokens] = (payloads, cpu_stats)

    results: list[dict] = []

    def emit(result: dict) -> None:
        results.append(result)
        print("BENCH_RESULT " + json.dumps(result), flush=True)
        if args.results_jsonl:
            write_results(args.results_jsonl, results)

    if args.cpu_bench_only:
        for max_visual_tokens, (_, cpu_stats) in pools.items():
            point = {"max_visual_tokens": max_visual_tokens, "max_render_dpi": args.max_render_dpi}
            emit({"mode": "cpu_only", "config": point, "cpu": cpu_stats})
        return
    if all(not payloads for payloads, _ in pools.values()):
        raise RuntimeError("no renderable pages in the sampled PDFs; widen --num-pdfs")

    server_summary = {
        "instances": args.instances,
        "gdn_backend": args.gdn_backend,
        "attention_backend": args.attention_backend,
        "tensor_parallel": args.tensor_parallel,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_in_flight": args.max_in_flight,
        "api_server_count": args.api_server_count,
        "gpu_worker_cpu": args.gpu_worker_cpu,
        "gpu_worker_ram_gb": args.gpu_worker_ram_gb,
        "client_processes": args.client_processes,
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
    }
    gpus = args.instances * args.tensor_parallel
    startup_start = time.monotonic()
    with remote_inference(build_inference_config(args)) as session:
        startup_seconds = time.monotonic() - startup_start
        base_url = session.model.endpoint.base_url
        logger.info("endpoint ready in %.0fs: %s (backend=%s)", startup_seconds, base_url, session.backend_name)
        for max_visual_tokens, (payloads, cpu_stats) in pools.items():
            if not payloads:
                continue
            for concurrency in args.concurrency:
                load = run_load_test(
                    base_url, session.model.endpoint.model, payloads, args, concurrency, max_visual_tokens
                )
                emit(
                    {
                        "mode": "smoke" if args.smoke else "bench",
                        "config": {
                            **server_summary,
                            "max_visual_tokens": max_visual_tokens,
                            "max_render_dpi": args.max_render_dpi,
                            "legacy_longest_side": args.legacy_longest_side,
                            "concurrency": concurrency,
                        },
                        "cpu": cpu_stats,
                        "startup_seconds": round(startup_seconds, 1),
                        "gpus": gpus,
                        "pages_per_second_per_gpu": round(load["pages_per_second"] / gpus, 3) if gpus else None,
                        **load,
                    }
                )


if __name__ == "__main__":
    main()
