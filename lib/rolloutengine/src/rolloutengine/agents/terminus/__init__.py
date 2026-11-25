# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor's Terminus-2 agent as a RolloutEngine ``AgentHarness``.

The agent, its JSON and XML response parsers, its tmux session and its prompt templates are adapted from
Harbor (https://github.com/laude-institute/harbor, vendored in Marin as ``vendor/harbor`` from
marin-community/harbor@9d099696), licensed under the Apache License 2.0. The prompts, parsers, keystroke
handling, output limits, completion confirmation and context recovery keep Harbor's behavior. The task
machine replaces Harbor's environment, and the harness's ``complete`` replaces Harbor's LLM client.
"""
