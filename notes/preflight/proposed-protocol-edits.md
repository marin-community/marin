# Proposed inference protocol edits

These are small, evidence-backed replacements for human approval. The source
protocol remains unchanged.

## 1. Separate architecture validity from launcher diagnosis

The exact reference is blocked by every-six attention, heterogeneous KV heads,
and sconv-on. The every-four, uniform-12, sconv-off model is useful only as a
serving-stack diagnostic.

Suggested replacement:

```markdown
Before any performance gate, classify the exact reference as representable or
blocked. A blocked reference may use the named every-four, uniform-12,
sconv-off approximation to test loading, distributed launch, prefix caching,
fabric, metrics, and artifact persistence. Approximation throughput must not
rank architecture candidates.
```

## 2. Make KV allocation evidence explicit

The live approximation exposes 296,653 KV bytes per token. This is 293% above
its sliding-window semantic estimate and within 0.6% of full-length allocation
for all 48 layers.

Suggested replacement:

```markdown
Record three KV quantities separately: semantic bytes implied by attention
windows, bytes reserved by the runtime cache pool, and peak bytes observed
during requests. A gap above 10% blocks concurrency conclusions until
explained. For hybrid attention with prefix caching, do not assume local-window
layers reduce reserved KV bytes.
```

## 3. Tighten the final-launch gate

The current dev workflow submits replicated holder tasks, then relies on
workstation `kubectl`. The production serving command submits one task.

Suggested replacement:

```markdown
The two-node gate passes for final-launch readiness only when the same
replicated Iris entrypoint can run unattended. A workstation controller may
diagnose holder pods, but it cannot qualify the four-node acceptance path. If
the unattended entrypoint is absent, stop before allocating four nodes.
```

## 4. Keep correctness claims separate

The existing response extension proves selected routed IDs. It does not expose
gate weights. The live repeat proves output repeatability but not throughput
repeatability.

Suggested replacement:

```markdown
Cross-framework parity requires one checkpoint and one token fixture loaded by
both frameworks. Compare selected experts, gate weights, and next-token
logprobs. Routed-ID response capture alone does not satisfy this gate.

Report correctness repeatability and throughput repeatability separately. The
2% threshold applies only to two warmed, ten-minute, 250,000-generated-token
arms.
```

## 5. Define the routing control narrowly

Live seeded dummy routing was deterministic but left most experts unused.

Suggested replacement:

```markdown
Record expert and contiguous-linear EP-rank histograms. If experts remain
unused or a rank exceeds twice the mean, run one cyclic balanced assignment
fixture through the histogram code. Label it an instrumentation control; it is
not model-routing evidence.
```

## 6. Preserve the Snowball stop rule

The single request-path attempt failed before load because the model streamer
used path-style S3 listing. A virtual-hosted config is now patched but not
validated.

Suggested replacement:

```markdown
Run the pinned Snowball request path once. If it fails before model load and
would require another serving integration, object copy, or second attempt,
record the exact error as still uncertain and stop. Do not convert this check
into model-loader development.
```

No other protocol change is supported by this preflight.
