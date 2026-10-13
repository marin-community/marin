# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse optional Grug thinking within the current assistant turn."""

from collections.abc import Iterable, Sequence

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.grug_reasoning_parser import GrugReasoningParser


@ReasoningParserManager.register_module("grug_optional")
class GrugOptionalReasoningParser(GrugReasoningParser):
    """Handle continuations that begin with a tool call instead of thinking."""

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # vLLM checks the whole prompt before streaming. Historical closing
        # markers must not disable reasoning for the new assistant response.
        header_id = self.vocab["<|start_header_id|>"]
        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] == header_id:
                return super().is_reasoning_end(input_ids[index + 1 :])
        return super().is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        if self.start_token_id not in input_ids and self.end_token_id not in input_ids:
            return bool(self.model_tokenizer.decode(input_ids).strip())
        return super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.start_token_id not in input_ids and self.end_token_id not in input_ids:
            return input_ids
        return super().extract_content_ids(input_ids)
