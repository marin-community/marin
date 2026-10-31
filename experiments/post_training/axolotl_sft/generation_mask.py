# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl labels from generation spans in the final rendered text conversation.

Native Qwen templates remove historical reasoning. Prefix-by-prefix turn offsets
therefore do not identify spans in the full conversation. Render once and use
the template's generation mask, retaining Axolotl's message normalization and
batched dataset handling. This strategy is for text-only, assistant-only SFT.
"""

from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy, StrategyLoader
from axolotl.prompters import IGNORE_TOKEN_ID


class GenerationMaskStrategy(ChatTemplateStrategy):
    def _tokenize_single_prompt(self, prompt: dict) -> dict:
        encoded = self.tokenizer.apply_chat_template(
            self.get_conversation_thread(prompt),
            tools=self._get_tools(prompt),
            chat_template=self.prompter.chat_template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            **self.prompter.chat_template_kwargs,
        )
        mask = encoded.pop("assistant_masks")
        encoded["labels"] = [
            token if assistant else IGNORE_TOKEN_ID for token, assistant in zip(encoded["input_ids"], mask, strict=True)
        ]
        return dict(encoded)


class GenerationMaskLoader(StrategyLoader):
    def _get_strategy_cls(self, cfg):
        if cfg.train_on_inputs:
            raise ValueError("Generation masks require assistant-only loss")
        return GenerationMaskStrategy


load = GenerationMaskLoader()
