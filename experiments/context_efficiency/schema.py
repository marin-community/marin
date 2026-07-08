# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The semantic label schema shared by the labeler prompt and the analysis.

One source of truth for the controlled vocabularies so the rubric a labeler is
handed and the buckets the analysis groups on cannot drift apart.
"""

# The substitutes a labeler may name as the best thing that would forestall or
# shrink an episode's lookups. Ordered so "none" (irreducible) is first.
SUBSTITUTES = [
    "none",
    "shared-wiki",
    "semantic-code-index",
    "persistent-memory",
    "better-tool-or-flag",
    "result-compaction",
    "repo-map-or-docs",
]

# Realizability class of each substitute — how its per-episode saving is (or is not)
# realizable in aggregate.
#   AUTOMATIC       generated from code/tooling, no per-item authoring -> realizable as-is.
#   AUTHORED_REPO   one artifact per repo, amortized over every navigation episode there.
#   AUTHORED_TOPIC  one artifact per fact -> gated by how often the topic RECURS across sessions.
AUTOMATIC = {"semantic-code-index", "better-tool-or-flag", "result-compaction"}
AUTHORED_REPO = {"repo-map-or-docs"}
AUTHORED_TOPIC = {"shared-wiki", "persistent-memory"}

# What the episode was trying to do, and what shape of answer it sought.
INTENT_CATEGORIES = [
    "code-navigation",
    "code-comprehension",
    "config-lookup",
    "api-usage",
    "debug-error",
    "test-or-build-run",
    "git-history",
    "pr-issue-inspect",
    "infra-or-cluster-state",
    "data-inspect",
    "verify-own-change",
    "read-own-output",
    "external-docs",
    "other",
]
ANSWER_KINDS = [
    "file-location",
    "symbol-definition",
    "code-snippet",
    "config-value",
    "command-output",
    "error-cause",
    "factual-recall",
    "status-or-state",
    "procedure-howto",
    "large-blob",
    "other",
]
SUFFICIENCY = ["yes", "partial", "no"]
