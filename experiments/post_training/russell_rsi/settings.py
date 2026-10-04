# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Snowball token limits for the Russell spike."""

CONTEXT_TOKENS = 32768
PROMPT_TOKENS = 16384
RESPONSE_TOKENS = 4096
STOP_TOKEN_IDS = (128001, 128009)
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
ROLLOUT_CONCURRENCY = 8
GLM_TOKEN_ENV = "GLM_API_TOKEN"
