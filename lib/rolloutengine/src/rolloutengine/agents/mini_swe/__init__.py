# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""mini-swe-agent 2.1.0, as Harbor runs it, as a RolloutEngine ``AgentHarness``.

The agent loop, its tool-call parsing and the local environment's command handling are adapted from
mini-swe-agent 2.1.0 (https://github.com/SWE-agent/mini-swe-agent), licensed under the MIT License; see
``LICENSE.md`` in this package. ``mini.yaml`` is that release's configuration file, unchanged, and supplies the
prompts, the observation and format-error templates, and the command environment. The context budget is adapted
from Harbor's ``mini_swe_model`` and ``vllm_tokenization`` modules (https://github.com/laude-institute/harbor,
vendored in Marin as ``vendor/harbor`` from marin-community/harbor@9d099696), licensed under the Apache License
2.0. The task machine replaces mini-swe-agent's local environment, and the harness's ``complete`` replaces its
OpenAI client.
"""
