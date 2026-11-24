# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor's installed Pi agent as a RolloutEngine harness.

The agent, its runtime scripts, its MCP configuration script and its model relay are adapted from Harbor
(https://github.com/laude-institute/harbor, vendored in Marin as ``vendor/harbor`` from
marin-community/harbor@9d099696), licensed under the Apache License 2.0. The version check, Pi's provider,
compaction and context-budget configuration, the command line, the staging of endpoint files, the MCP setup and
the classification of Pi's failures keep Harbor's behavior. The task machine replaces Harbor's environment, and
the harness's ``complete`` replaces the host-authenticated model endpoint behind Harbor's relay.
"""
