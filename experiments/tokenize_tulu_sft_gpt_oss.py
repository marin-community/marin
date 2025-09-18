#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Experiment to tokenize the Tulu-3 SFT dataset using GPT-OSS tokenizer for future SFT training.
This downloads from HuggingFace and creates a tokenized cache that can be reused.
"""

from experiments.defaults import default_tokenize
from levanter.data.text import ChatLmDatasetFormat
from marin.execution.executor import executor_main

# Use the GPT-OSS tokenizer from the config
GPT_OSS_TOKENIZER = "openai/gpt-oss-20b"

# Simple chat template for GPT-OSS (since it likely doesn't have a built-in one)
# Based on OpenAI's chat format but with generation tags for Levanter
SIMPLE_CHAT_TEMPLATE = """
{%- for message in messages -%}
  {%- if message['role'] == 'system' -%}
System: {{ message['content'] | trim }}

  {%- elif message['role'] == 'user' -%}
User: {{ message['content'] | trim }}

  {%- elif message['role'] == 'assistant' -%}
Assistant: {% generation %}{{ message['content'] | trim }}{% endgeneration %}

  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
Assistant:
{%- endif -%}""".strip()

# Chat format for SFT data with explicit chat template
chat_format = ChatLmDatasetFormat(
    messages_field="messages",
    single_turn=False,
    chat_template=SIMPLE_CHAT_TEMPLATE,  # Use explicit chat template
    mask_user_turns=True,
)

# Tokenize Tulu-3 SFT dataset using HuggingFace ID and GPT-OSS tokenizer
tulu3_sft_gpt_oss_tokenize_step = default_tokenize(
    name="tulu3_sft_gpt_oss_tokenizer",
    dataset="allenai/tulu-3-sft-mixture",  # HuggingFace dataset ID
    tokenizer=GPT_OSS_TOKENIZER,
    format=chat_format,
)

if __name__ == "__main__":
    # Run only the tokenization step - no training
    executor_main(steps=[tulu3_sft_gpt_oss_tokenize_step])
