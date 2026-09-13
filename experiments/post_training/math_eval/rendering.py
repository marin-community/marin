# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the pinned serving engine's incremental response detokenization."""

import logging

import tokenizers
import tokenizers.decoders

INCREMENTAL_RENDERING = "vllm_f0d7_incremental_skip_special_tokens_v1"
DETOKENIZER_SOURCE_SHA256 = "213d71cf6eefcea061b28b656cf8a08af60c9f3513403e724b16876995f3de93"
TOKENIZERS_VERSION = "0.22.2"
logger = logging.getLogger(__name__)


def renderer_provenance():
    """Validate the replay dependency before starting native inference."""
    if tokenizers.__version__ != TOKENIZERS_VERSION:
        raise ValueError("Serving detokenizer library changed; requalify its renderer")
    return {
        "method": INCREMENTAL_RENDERING,
        "detokenizer_source_sha256": DETOKENIZER_SOURCE_SHA256,
        "tokenizers_version": TOKENIZERS_VERSION,
    }


def render_serving_response(decoder, prompt_tokens, response_tokens):
    """Match FastIncrementalDetokenizer with the frozen serving request controls.

    The decode stream is primed by the original prompt. Incomplete UTF-8 bytes
    remain buffered at termination; this does not strip replacement characters
    from already decoded text. Stop strings are absent and special tokens are
    skipped under the separately validated request contract.
    """
    if tokenizers.__version__ != TOKENIZERS_VERSION:
        raise ValueError("Serving detokenizer library changed; requalify its renderer")
    stream = tokenizers.decoders.DecodeStream(ids=prompt_tokens, skip_special_tokens=True)
    output = []
    for token in response_tokens:
        try:
            text = stream.step(decoder, token)
        except (OverflowError, TypeError):
            # Exact pinned engine behavior for a token the decoder cannot accept.
            logger.exception("Pinned serving detokenizer rejected token ID %s", token)
            text = None
        except Exception as error:
            if not str(error).startswith("Invalid prefix encountered"):
                raise
            # The pinned engine resets only this known non-monotonic UTF-8 case.
            logger.warning("Resetting pinned serving decode stream after invalid prefix")
            stream = tokenizers.decoders.DecodeStream(skip_special_tokens=True)
            text = stream.step(decoder, token)
        output.append(text or "")
    return "".join(output)
