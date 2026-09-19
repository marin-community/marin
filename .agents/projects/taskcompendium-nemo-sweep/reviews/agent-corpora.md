# Agent corpora review

The sampler pinned seven corpora at the revisions recorded in `agent-corpora.json`; Dataset Viewer returned no row count for ReasoningGym-v1 and SysBench-v1, so those two corpora have two unavailable deferred slots with null offsets.

Both Function-Calling Pivot rows and both Conversational Tool-Use Pivot rows are accepted as answer-only native-action prediction tasks. Their public contract is the source conversation plus the explicitly advertised native function schemas; `PredictedActionVerifier` retains the expected action privately, and no action is executed. The rows do not establish provider capabilities or an environment requirement.

The workplace rows are rejected because the existing workplace provider/importer is pinned to the separate id-0 fixture and seed. Calendar rows are rejected because no supported contract preserves their evolving calendar state. CFBench rows are rejected because they combine multi-turn instruction aggregation with private judge criteria. SWE rows are deferred because repository contents and final state are unavailable; treating their shell calls as a standalone prediction would lose the repository-repair task semantics.
