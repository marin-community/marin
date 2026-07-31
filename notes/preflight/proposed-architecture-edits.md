# Proposed architecture-experiment edits

These are small evidence-backed replacement snippets for human approval. The
source architecture draft remains unchanged.

## 1. Replace the preflight banner

Suggested replacement:

```markdown
Preflight result: GO for the exact serving baseline. The pinned Marin and
vLLM branches faithfully run the d6144, 48-layer reference with every-six
global attention, 12-local/6-global KV heads, window 512, SConv4, two distinct
shared experts, and PP1/TP1/DP16/EP16. Frozen-tensor parity, active KV
accounting, every P0 implementation family, unattended two-node launch, and
the four-node acceptance pass.

This is serving-stack evidence from frozen and deterministic dummy weights.
The one bounded trained Snowball attempt failed before model load on
object-store access, so do not present the accepted throughput as trained-model
performance.
```

## 2. Pin the exact reference and evidence

Suggested replacement:

```markdown
Reference serving provenance:

- Marin: `a3320a3043018ee923bc98bf2e6e6eef3f03a6fe`
- vLLM: `2c2bef33dfbd7aef3c9d4433a7e4110f77d56a4a`
- image:
  `ghcr.io/marin-community/iris-task@sha256:5e2a69af91a000cb999e6ff0d92933874bd3142eb45469fc64fc7a3f5db64fbb`
- cluster/topology: four whole `gb200-4x` nodes in one hard
  `nvlink.domain`, PP1/TP1/DP16/EP16
- model: d6144, 48 layers, 48 query heads, 12 local KV heads, 6 global
  KV heads, global attention every 6 layers, local window 512, top-4 of
  128 routed experts at i3072, two distinct i3072 shared experts, SConv4
  at K/V/attention/MLP, and all remaining frozen custom blocks.
```

## 3. Mark every P0 implementation family ready

Suggested replacement:

```markdown
P0 implementation readiness:

| Family | Classification | Status |
|---|---|---|
| uniform KV / every 4 / SConv off | config-only control | ready |
| heterogeneous KV / every 6 / SConv on | custom serving path | ready |
| global KV 2 / window 2,048 | config-only on the custom path | ready |
| top-8 / 256 experts / EP16 | config-only | ready |
| exact reference / EP16 | custom path plus unattended launcher | ready |

“Ready” means a downscaled dummy-weight smoke starts and generates through the
distinct path. It is not comparative throughput evidence.
```

## 4. Replace the KV-capacity warning

Suggested replacement:

```markdown
With prefix caching enabled and one active request, exact-reference local KV
plateaus while global KV grows:

| Final length | Local/global/SConv active blocks | Physical active bytes | Reserved pool |
|---:|---:|---:|---:|
| 6,144 | 33 / 180 / 2 | 299,630,592 | 61,899,276,288 |
| 65,536 | 33 / 2,039 / 2 | 1,761,607,680 | 61,899,276,288 |

Physical active bytes equal the hybrid-group payload. The attention prediction
differs by 1.99% at 6,144 and 0.18% at 65,536. Treat the 61.9 GB reservation
as a reusable runtime pool, not as one sequence's active KV.
```

## 5. Add the routing qualification

Suggested replacement:

```markdown
Record logical expert and contiguous-linear EP-rank histograms beside every
throughput result. The deterministic preflight router is reproducible but not
balanced: 121 of 128 experts were unused and its busiest EP rank carried four
times the mean assignments. The cyclic equal-assignment case validates the
histogram and communication path only; it is not model-routing evidence.
```

## 6. Record the acceptance without ranking candidates

Suggested replacement:

```markdown
The exact four-node acceptance passed on one allocation. After warmup, both
identical arms covered all 144 branches, ran ten stable live-counter
intervals, generated 950,272 tokens, and reported 1,578.25 versus 1,583.50
aggregate generated tokens/s. The difference was 0.3324%, below the 2% gate,
with zero preemptions.

This clears the baseline machinery for the later matrix. It does not rank any
architecture candidate, and no comparison or profile was collected during
preflight.
```

## 7. Keep framework claims precise

Suggested replacement:

```markdown
Cross-framework parity uses one frozen downscaled exact checkpoint. It checks
selected experts, normalized routing weights, shared-expert sum, and
next-token probabilities against the training oracle. It also requires cold
prefill and prefix reuse to return identical tokens, logprobs, and routes
across a physical block boundary and the 512-token sliding-window boundary.

Ordinary serving uses trunk logits and does not execute the dense MTP training
head. Speculative MTP decoding remains a separate experiment.
```
