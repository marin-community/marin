# Proposed architecture-experiment edits

These are small replacement snippets for human approval. The source
architecture draft remains unchanged.

## 1. Mark unsupported rows as blocked

Suggested replacement:

```markdown
At vLLM `afb26719464d5957e695bde478ae93a160b11d14`, heterogeneous
12/6 and 12/2 KV, global attention every six layers, and sconv-on are
implementation-blocked. They are not pending benchmark rows. Exact support and
same-tensor correctness must land before their throughput can update the
architecture ranking.
```

## 2. Add one non-ranking launcher reference

Suggested replacement:

```markdown
Non-ranking control: d6144, 48 layers, 48 query heads, uniform 12 KV heads,
global attention every four layers, window 512, top-4-of-128 routed experts,
i3072, sconv off, PP1/TP1/DP8/EP8. Use it only to diagnose loading, fabric,
prefix reuse, metrics, and artifacts.
```

The control's semantic KV estimate is 4.605 GiB per 65K sequence. The live
cache pool implies about 18.1 GiB, so neither value may stand in for the exact
model's 1.617 GiB estimate.

## 3. Gate the expert-granularity comparison on orchestration

Suggested replacement:

```markdown
Keep top-4/128/i3072 versus top-8/256/i1536 as the only expert-granularity
comparison. The EP16 row remains confounded until a replicated unattended Iris
entrypoint exists. Once it exists and all smaller correctness gates pass, run
one four-node acceptance attempt; do not substitute manually controlled holder
pods.
```

## 4. Add load-distribution evidence

Suggested replacement:

```markdown
Record logical expert and EP-rank assignment histograms beside throughput.
Use contiguous linear ownership when mapping experts to EP ranks. If imbalance
coincides with a throughput change, mark the comparison confounded. A balanced
synthetic fixture validates instrumentation only.
```

## 5. Keep framework and serving claims distinct

Suggested replacement:

```markdown
“Routed expert IDs returned” and “Levanter/vLLM gate weights agree” are
separate gates. Require same-tensor selected-expert, gate-weight, and
next-token-logprob parity before interpreting an architecture result.

Keep dense MTP outside ordinary serving. Require a frozen-tensor equality test
before treating two half-width shared experts as one fused full-width shared
MLP.
```

## 6. Add a no-go banner for the current matrix

Suggested replacement:

```markdown
Current preflight result: NO-GO. EP8 launch and prefix behavior pass on the
non-ranking approximation, but the exact architecture, cross-framework tensor
parity, reserved-KV model, unattended EP16 launch, and throughput repeatability
gates are not all ready.
```

No additional architecture arm is supported by the live evidence.
