# Proposed inference-protocol edits

These are small evidence-backed replacement snippets for human approval. The
source protocol remains unchanged.

## 1. Freeze implementation provenance before live evidence

Suggested replacement:

```markdown
Run live evidence only from clean, pushed commits. Each Marin manifest must
pin the exact vLLM commit, immutable image digest, dependency lock, model
configuration, workload hash and seed, Iris job, task count, and required
topology. Do not rewrite or force-push an evidence-linked commit.
```

## 2. Make exact parity a fail-fast gate

Suggested replacement:

```markdown
Before a reference smoke, load one frozen downscaled exact checkpoint through
the training oracle and vLLM. Enable every custom block and use two distinct,
nonzero half-width shared experts. Compare selected experts, normalized
routing weights, the summed shared-expert output, and next-token probabilities
under the repository's existing cross-framework tolerance.

On the live server, require cold full prefill and prefix reuse to return
identical tokens, logprobs, and routed experts across both a physical KV-block
boundary and the 512-token local-window boundary. Mutating one prefix token
must report zero reuse.
```

## 3. Separate three KV quantities

Suggested replacement:

```markdown
With prefix caching enabled, hold active request count and DP rank fixed at two
or more context lengths above the local window, including 65,536 tokens.
Record separately:

1. semantic bytes predicted by local, global, and recurrent state;
2. runtime bytes reserved for the reusable cache pool;
3. physical pages active for the live request.

Require local active pages to plateau and global active pages to grow. Explain
any physical-active versus semantic-payload gap above 10%. Do not treat the
whole reserved pool as per-request occupancy, and do not disable prefix caching
to hide a hybrid-cache defect.
```

## 4. Use one unattended path for the topology ladder

Suggested replacement:

```markdown
The final launcher is one zero-retry Iris gang entrypoint. Each task requests
all four GPUs on one `gb200-4x` node; Kueue hard-coschedules every task in one
`nvlink.domain`. Prove the same submit, rendezvous, health, correctness,
aggregation, upload, and readback path at two nodes before four-node
acceptance. Workstation-controlled holder pods are diagnostic only.

The aggregate result passes only when placement, every rank's health,
correctness, duration, token count, repeatability, and artifact readback all
pass.
```

## 5. Measure long requests with live counters

Suggested replacement:

```markdown
Warm the allocation and populate every prefix cohort. In each measured arm,
sample `vllm:generation_tokens` at fixed wall-clock boundaries while requests
remain active. Compute stable interval throughput from adjacent counter deltas
and their measured elapsed time. Keep request totals, branch coverage,
latencies, and cohort totals completion-based.

Do not assign all response tokens to the minute when a long request returns;
that can produce false zero minutes when request latency exceeds the sampling
interval.
```

## 6. Preserve the acceptance and rerun rule

Suggested replacement:

```markdown
Use six roots in each 10,240/30,720/62,464-token history cohort and eight
branches per root. Append 1,024 tokens and generate 2,048, ending at
13,312/33,792/65,536. On one warmed PP1/TP1/DP16/EP16 allocation, run two
identical arms. Each arm must cover all 144 branches, contain ten stable
live-counter intervals, and generate at least 250,000 tokens. Require no more
than 2% difference between the two stable means.

Plan one acceptance attempt. Repeat only after preserving the failed bundle,
identifying a concrete implementation or measurement defect, fixing it
without changing the workload or threshold, and adding a focused regression
test. Never rerun merely to obtain a favorable result.
```

## 7. Keep the Snowball check bounded

Suggested replacement:

```markdown
Run the pinned Snowball request path at most once during preflight. It is a
storage, request-construction, prefix, logprob, metrics, and artifact check;
it is not target-throughput evidence. If it fails before model load and would
need another integration, object copy, or retry, record the failure and stop.
Do not let this check expand into model-loader or object-store development.
```

## 8. Require independent artifact closure

Suggested replacement:

```markdown
Upload configuration, workload, commands, commits, image digest, placement,
metrics, routes, rank receipts, logs, aggregate result, and a byte-hash
manifest under the run's final S3 prefix. A separate authorized task must read
every claimed object, verify byte identity and the aggregate result, and reach
a successful terminal state.
```
