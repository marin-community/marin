# Domain sweep launch plan

Implement the reviewed 28-run target/matched survey without changing frozen StarCoder files or adding training points.

1. Build fresh central1 Wikipedia/FineMath raw, permuted-parent and nested-subset caches using the reviewed object generations, tokenizer, quotas and index draws. Reuse web and held-out caches. Bound parquet reads and reuse completed parts.
2. Compose the frozen StarCoder training recipe with the new focus cache. Preserve internal source names and shuffle keys for exact allocation parity; add all seven Uncheatable validation components. Freeze code, runtime, cache and scoring identities.
3. Check real data-stream behavior, nested membership, evaluator completeness, resumability and exact 28-run selection; lint and type-check the new modules; obtain a scoped CC implementation review.
4. Record the release in Fieldbook, submit a region-pinned non-preemptible coordinator at interactive priority. Prepare and audit data, verify the two matched p=100 endpoints as an execution gate within the 28 runs, then release the remaining 26 concurrently. Successful outputs are reused, never retrained.
5. Verify live progress and required endpoints; keep a thread monitor and Fieldbook/CC handoff current through completion and recover only within the authorized scientific setting.

The user explicitly authorized submission and requested these as the next highest-priority research runs. Iris researcher jobs are limited to the interactive band. No unrelated jobs or cluster state will be changed.
