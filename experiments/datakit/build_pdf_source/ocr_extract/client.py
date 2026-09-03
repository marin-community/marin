# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Send one rendered page to the OCR endpoint and get Markdown back.

The senders are thin: render a page, post it, keep the answer. Every default in this path that
bounds concurrency has to be raised, because each one silently caps throughput rather than failing.
"""

import base64
import logging
from dataclasses import dataclass
from functools import cache

import httpx
from openai import OpenAI

from experiments.datakit.build_pdf_source.ocr_extract.render import MIN_PIXELS, VISUAL_TOKEN_PIXELS, RenderedPage

logger = logging.getLogger(__name__)

# The doc2md prompt from ``infinity_parser2/prompts.py`` (PROMPT_DOC2MD), with one added line under
# "Output Format" forbidding stray ```markdown fences. Any change here re-keys the extraction step:
# ``prompt_digest`` is in its hash_attrs.
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

# Roughly 6x the mean completion length per page; bounds a runaway repetition loop.
DEFAULT_MAX_TOKENS = 4096
# Bounds a hung request; the broker's own lease timeout is the backstop.
DEFAULT_REQUEST_TIMEOUT = 900.0
DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class OcrEndpoint:
    """Where the OCR fleet is and how to talk to it.

    Frozen and hashable so a sender process can cache one client per endpoint.
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
    # The model hit ``max_tokens`` and was cut off mid-page; the text is real but incomplete.
    truncated: bool = False


@cache
def _client(endpoint: OcrEndpoint, connections: int) -> OpenAI:
    """One OpenAI client per process, with a connection pool sized for the sender's thread count.

    httpx defaults to 100 connections, which silently caps in-flight requests below the sender's
    concurrency instead of failing.
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
    """Drop a ```markdown fence wrapping the whole page.

    Only an explicit ``markdown``/``md`` info string is unwrapped: a bare ``` wrapper may be a real
    code listing and is left alone, as are fences inside the page. A ``truncated`` page has an
    opening fence and no closing one, so its opener alone is unwrapped.
    """
    lines = text.strip().split("\n")
    if not lines or lines[0].strip().removeprefix("```").strip().lower() not in ("markdown", "md"):
        return text
    if lines[-1].rstrip() == "```":
        return "\n".join(lines[1:-1])
    return "\n".join(lines[1:]) if truncated else text


def ocr_page(endpoint: OcrEndpoint, connections: int, page: RenderedPage) -> PageOcr:
    """Convert one rendered page to Markdown.

    Exceptions propagate: the extraction step holds the document-level policy for a failed page.
    """
    response = _client(endpoint, connections).chat.completions.create(
        model=endpoint.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # The page crossed the pipe as PNG bytes; this is the one place that wants base64.
                        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(page.png).decode()}"},
                        # Restate the render budget so the server's own smart_resize cannot re-size the page.
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
        # A reasoning-capable hybrid; transcription has nothing to reason about.
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
