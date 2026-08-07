# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Send one rendered page to the OCR endpoint and get Markdown back.

The senders are deliberately thin. All they do is render a page, post it, and keep the answer; the
model, the batching, and the queueing all live behind the endpoint. What the client is responsible
for is not under-driving the fleet -- every default in this path that bounds concurrency has to be
raised, because each one silently caps throughput rather than failing.
"""

import logging
from dataclasses import dataclass
from functools import cache

import httpx
from openai import OpenAI

from experiments.datakit.build_pdf_source.ocr_extract.render import MIN_PIXELS, VISUAL_TOKEN_PIXELS, RenderedPage

logger = logging.getLogger(__name__)

# The doc2md prompt from ``infinity_parser2/prompts.py`` (PROMPT_DOC2MD), with one added line under
# "Output Format" forbidding stray ```markdown fences -- the model wrapped whole pages in them
# otherwise, which is transcription of nothing and would have to be stripped downstream.
#
# Treat the rest as fixed. It is part of the model's validated input distribution, and the tag
# vocabulary it asks for is what the boilerplate pass and every downstream consumer expect. Any
# change here re-keys the extraction step, by design: ``prompt_digest`` is in its hash_attrs.
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
- The output Markdown document should not contain any markdown code blocks (i.e. ```markdown ```) that are not part of the original PDF content.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""  # noqa: E501 -- the prompt's line breaks are part of the input; rewrapping it changes the input

# Measured mean completion length is ~708 tokens per page across every pod shape and page size in
# the sweep, so this is roughly 6x the mean and exists to bound a runaway repetition loop.
DEFAULT_MAX_TOKENS = 4096
# Well past the measured p50 of 21s at the fleet's operating point. The cost of a timeout that is
# too tight is a lost page; the cost of one that is too loose is a stalled sender thread, and the
# broker's own lease timeout is the backstop that actually matters.
DEFAULT_REQUEST_TIMEOUT = 900.0
DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class OcrEndpoint:
    """Where the OCR fleet is and how to talk to it.

    Frozen and hashable so a sender process can cache one client per endpoint, and cheap to pickle
    so the driver can hand it to every Zephyr map task.
    """

    base_url: str
    model: str
    max_visual_tokens: int
    max_tokens: int = DEFAULT_MAX_TOKENS
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES


@dataclass(frozen=True)
class PageOcr:
    """One page's Markdown, what it cost, and whether the model got to finish."""

    text: str
    completion_tokens: int
    # The model hit ``max_tokens`` and was cut off mid-page. The text is real but incomplete, and
    # nothing else about the response says so -- a truncated page is a normal 200 with a shorter
    # body. Dense pages (formal proofs, packed tables) are where this happens.
    truncated: bool = False


@cache
def _client(endpoint: OcrEndpoint, connections: int) -> OpenAI:
    """One OpenAI client per process, with a connection pool sized for the sender's thread count.

    httpx defaults to 100 connections, which silently caps in-flight requests below the sender's
    concurrency instead of failing -- during the throughput campaign this pinned engine-side
    ``num_requests_running`` at ~99 no matter what the client asked for.
    """
    limits = httpx.Limits(max_connections=connections, max_keepalive_connections=connections)
    return OpenAI(
        api_key="EMPTY",
        base_url=endpoint.base_url,
        timeout=endpoint.request_timeout,
        max_retries=endpoint.max_retries,
        http_client=httpx.Client(limits=limits, timeout=endpoint.request_timeout),
    )


def unwrap_markdown_fence(text: str, *, truncated: bool = False) -> str:
    """Drop a fence wrapping the whole page.

    The model returns the page as one ```markdown block, which is a transcription of nothing -- the
    page is not a code listing. Asking it not to in the prompt has no effect: measured over 3,000
    raw responses, 3,000 of them were wrapped. So the wrapper is removed here instead, which handles
    99.0% of pages.

    Only an explicit ``markdown``/``md`` info string is unwrapped. A bare ``` wrapper is left alone:
    a page really can be a code listing, and on that page the fence is content. Fences *inside* the
    page are untouched either way.

    ``truncated`` is what the remaining 1% needs. A page cut off at ``max_tokens`` has an opening
    fence and no closing one, so the usual both-ends test fails and the marker survives into the
    corpus. Since wrapping is universal, an opener on a page the model never finished can only be a
    wrapper -- there was never going to be a closer. Without this the leftover markers are
    concentrated exactly on the pages that are already damaged.

    Leaving any of this to the boilerplate pass would be wrong even though it often works by
    accident: a fence on every page is a repeated edge pattern, so long documents lose it there,
    and documents under ``BoilerplateOptions.min_pages`` keep it.
    """
    lines = text.strip().split("\n")
    if not lines or lines[0].strip().removeprefix("```").strip().lower() not in ("markdown", "md"):
        return text
    if lines[-1].rstrip() == "```":
        return "\n".join(lines[1:-1])
    return "\n".join(lines[1:]) if truncated else text


def ocr_page(endpoint: OcrEndpoint, connections: int, page: RenderedPage) -> PageOcr:
    """Convert one rendered page to Markdown.

    Exceptions propagate. A page request fails for reasons the caller has to distinguish between --
    a single bad page versus an endpoint that has gone away -- and the extraction step is what holds
    the document-level policy for that.
    """
    response = _client(endpoint, connections).chat.completions.create(
        model=endpoint.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": page.data_uri},
                        # Restate the budget the page was rendered at so the server's own
                        # smart_resize cannot re-size it: the model's ``preprocessor_config``
                        # defaults are unrelated to our budget, and a lower server-side cap would
                        # quietly undo the render decision this pipeline is built around.
                        "max_pixels": endpoint.max_visual_tokens * VISUAL_TOKEN_PIXELS,
                        "min_pixels": MIN_PIXELS,
                    },
                    {"type": "text", "text": PROMPT_DOC2MD},
                ],
            }
        ],
        max_tokens=endpoint.max_tokens,
        temperature=0.0,
        top_p=1.0,
        timeout=endpoint.request_timeout,
        # This is a reasoning-capable hybrid. Thinking tokens would be spent on, and billed for,
        # a transcription task that has nothing to reason about.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage = response.usage
    choice = response.choices[0]
    truncated = choice.finish_reason == "length"
    return PageOcr(
        text=unwrap_markdown_fence(choice.message.content or "", truncated=truncated),
        completion_tokens=usage.completion_tokens if usage else 0,
        truncated=truncated,
    )
