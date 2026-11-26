import logging
from pathlib import Path

from upath import UPath
from typing import Any

from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.rollout_detail import RolloutDetail

logger = logging.getLogger(__name__)


def assistant_message_from_response(
    response: LLMResponse, interleaved_thinking: bool
) -> dict[str, Any]:
    """Build an assistant history message using the configured thinking mode."""
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if interleaved_thinking:
        # Some chat templates require this field on every assistant message,
        # including synthetic history and responses without reasoning text.
        message["reasoning_content"] = response.reasoning_content or ""
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    return message


class Chat:
    def __init__(self, model: BaseLLM, interleaved_thinking: bool = False):
        self._model = model
        self._messages = []
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cache_tokens = 0
        self._cumulative_cost = 0.0
        self._prompt_token_ids_list: list[list[int]] = []
        self._completion_token_ids_list: list[list[int]] = []
        self._logprobs_list: list[list[float]] = []
        self._extra_list: list[dict[str, Any]] = []
        self._interleaved_thinking = interleaved_thinking
        self._last_response_id: str | None = None
        self._last_token_response: LLMResponse | None = None
        self._prepared_prompt: str | None = None
        self._prepared_prompt_token_ids: list[int] | None = None

    @property
    def total_input_tokens(self) -> int:
        return self._cumulative_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._cumulative_output_tokens

    @property
    def total_cache_tokens(self) -> int:
        return self._cumulative_cache_tokens

    @property
    def total_cost(self) -> float:
        return self._cumulative_cost

    @property
    def messages(self) -> list:
        return self._messages

    @property
    def rollout_details(self) -> list[RolloutDetail]:
        """Get detailed rollout information from all chat interactions.

        Returns:
            List containing a single RolloutDetail representing the complete
            linear chat history, or an empty list if no rollout data was collected.

            TODO: consider multiple rollout details for non-linear chat histories, e.g.
            subagents, summarization, etc.
        """
        if (
            not self._prompt_token_ids_list
            and not self._completion_token_ids_list
            and not self._logprobs_list
            and not self._extra_list
        ):
            return []

        rollout_detail: RolloutDetail = {}

        if self._prompt_token_ids_list:
            rollout_detail["prompt_token_ids"] = self._prompt_token_ids_list

        if self._completion_token_ids_list:
            rollout_detail["completion_token_ids"] = self._completion_token_ids_list

        if self._logprobs_list:
            rollout_detail["logprobs"] = self._logprobs_list

        if self._extra_list:
            # Pivot per-turn dicts to per-field lists for consistent indexing
            all_keys = {k for d in self._extra_list for k in d}
            rollout_detail["extra"] = {
                key: [turn.get(key) for turn in self._extra_list]
                for key in sorted(all_keys)
            }

        return [rollout_detail]

    async def chat(
        self,
        prompt: str,
        logging_path: Path | UPath | None = None,
        **kwargs,
    ) -> LLMResponse:
        prompt_token_ids = await self.prepare_prompt_token_ids(prompt, **kwargs)

        llm_response: LLMResponse = await self._model.call(
            prompt=prompt,
            message_history=self._messages,
            logging_path=logging_path,
            previous_response_id=self._last_response_id,
            prompt_token_ids=prompt_token_ids,
            **kwargs,
        )

        if (
            prompt_token_ids is not None
            and llm_response.prompt_token_ids != prompt_token_ids
        ):
            raise ValueError(
                "The provider did not serve the exact continuation prompt token IDs"
            )

        self.append_response(prompt, llm_response)
        return llm_response

    async def prepare_prompt_token_ids(self, prompt: str, **kwargs) -> list[int] | None:
        """Prepare and cache an exact token prompt for the next continuation."""
        if self._prepared_prompt == prompt:
            return self._prepared_prompt_token_ids

        prompt_token_ids = None
        last_response = self._last_token_response
        if (
            last_response is not None
            and last_response.prompt_token_ids is not None
            and last_response.completion_token_ids is not None
        ):
            prompt_token_ids = await self._model.build_continuation_prompt_token_ids(
                prompt,
                last_response.prompt_token_ids,
                last_response.completion_token_ids,
                **kwargs,
            )
        self._prepared_prompt = prompt
        self._prepared_prompt_token_ids = prompt_token_ids
        return prompt_token_ids

    def append_response(self, prompt: str, llm_response: LLMResponse) -> None:
        """Append one returned model response to the conversation and rollout."""

        self._prepared_prompt = None
        self._prepared_prompt_token_ids = None

        # Track response chain for Responses API
        if llm_response.response_id is not None:
            self._last_response_id = llm_response.response_id

        # Get token usage and cost from the LLM response
        usage = llm_response.usage
        if usage is not None:
            self._cumulative_input_tokens += usage.prompt_tokens
            self._cumulative_output_tokens += usage.completion_tokens
            self._cumulative_cache_tokens += usage.cache_tokens
            self._cumulative_cost += usage.cost_usd

        # Accumulate rollout details from the response
        self._accumulate_rollout_details(llm_response)

        if (
            llm_response.prompt_token_ids is not None
            and llm_response.completion_token_ids is not None
        ):
            self._last_token_response = llm_response
        else:
            self._last_token_response = None

        self._messages.extend(
            [
                {"role": "user", "content": prompt},
                assistant_message_from_response(
                    llm_response, self._interleaved_thinking
                ),
            ]
        )

    def append_tool_results(
        self, tool_calls: list[dict[str, Any]], content: str
    ) -> None:
        """Resolve native tool calls before the conversation's next model turn."""
        self._messages.extend(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": content,
            }
            for tool_call in tool_calls
        )
        # A tool-result message changes the rendered suffix. Re-render through
        # chat completions instead of applying the plain assistant/user TITO
        # continuation template.
        self.reset_response_chain()

    def add_auxiliary_usage(self, response: LLMResponse) -> None:
        """Account for an auxiliary model call outside the main conversation."""
        if response.usage is None:
            return
        self._cumulative_input_tokens += response.usage.prompt_tokens
        self._cumulative_output_tokens += response.usage.completion_tokens
        self._cumulative_cache_tokens += response.usage.cache_tokens
        self._cumulative_cost += response.usage.cost_usd

    def reset_response_chain(self) -> None:
        """Reset the response chain so the next call sends full message history.

        Call this whenever chat._messages is directly modified (e.g., after
        summarization or unwinding) to ensure the next Responses API call
        doesn't use a stale previous_response_id. Directly rewriting the history
        also invalidates the exact-token chain, so ``_last_token_response`` is
        dropped here too and the next request is rendered from the messages.
        """
        self._last_token_response = None
        self.reset_provider_response_chain()

    def reset_provider_response_chain(self) -> None:
        """Drop the provider response ID while retaining exact continuation tokens."""
        self._last_response_id = None
        self._prepared_prompt = None
        self._prepared_prompt_token_ids = None

    def _accumulate_rollout_details(self, llm_response: LLMResponse) -> None:
        """Accumulate rollout details from an LLM response.

        Args:
            llm_response: The LLM response containing token IDs and logprobs
        """
        # Accumulate prompt token IDs per turn
        if llm_response.prompt_token_ids:
            self._prompt_token_ids_list.append(llm_response.prompt_token_ids)

        # Accumulate completion token IDs per turn
        if llm_response.completion_token_ids:
            # Store completion token IDs for this turn
            self._completion_token_ids_list.append(llm_response.completion_token_ids)

            # Store logprobs for this turn (if available).
            #
            # TIS exact-alignment invariant: downstream training (SkyRL) zips these
            # logprobs onto the completion_token_ids by position to recover exact
            # per-token rollout logprobs without re-tokenizing. That zip is only
            # valid when the two arrays are the SAME length for this turn. vLLM
            # returns them parallel by construction, but guard explicitly so a
            # provider quirk degrades to the LCS fallback (logged + metric-counted
            # downstream) rather than silently mis-pairing logprobs to tokens.
            if llm_response.logprobs:
                n_ids = len(llm_response.completion_token_ids)
                n_lp = len(llm_response.logprobs)
                if n_lp == n_ids:
                    self._logprobs_list.append(llm_response.logprobs)
                else:
                    # Append an EMPTY logprob list (not nothing) so this turn stays
                    # index-aligned with completion_token_ids — downstream TIS keys
                    # both lists by the same per-turn index. An empty list is
                    # recorded as a per-message alignment failure downstream
                    # (tis/alignment_fail_count) rather than silently mis-pairing.
                    logger.warning(
                        "Rollout logprob/token-id length mismatch for this turn "
                        "(%d logprobs vs %d completion_token_ids); recording an "
                        "empty logprob list for this turn so downstream TIS flags "
                        "it instead of mis-pairing logprobs to tokens.",
                        n_lp,
                        n_ids,
                    )
                    self._logprobs_list.append([])

        # Accumulate extra provider-specific fields per turn
        if llm_response.extra:
            self._extra_list.append(llm_response.extra)
