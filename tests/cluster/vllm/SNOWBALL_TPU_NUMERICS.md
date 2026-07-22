# Snowball TPU numerical policy

All Snowball TPU contracts use the production v6e libtpu policy:

```text
LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304
```

Fray applies this default to every `v6e-*` task. It was introduced by
[Marin PR #3053](https://github.com/marin-community/marin/pull/3053). The
Levanter TPU VM guide describes it as important for performance, and
`lib/levanter/AGENTS.md` records the same v6e tuning value.

This setting is numerically relevant. On one physical production v6e-8, with
the checkpoint, prompts, fork revisions, packages, engine arguments, and all
other environment fields held fixed, adding only the scoped-VMEM setting moved
the `specialized-mmmlu-05` maximum probability error from `0.0595553706` to
`0.0747029154`. Both runs selected greedy token `426`, whose canonical gap was
`0.0`; the observations were bit-exact within each policy.

The v2 TPU-vLLM snapshot and bucket bounds were therefore re-derived through
the standing-cluster path with the production setting active. The same flag
also changes native and exported-Levanter rounding: under the production
policy those two paths are bit-identical to each other, but not to the older
snapshot captured without the flag. Their shared exact TPU golden is therefore
the production-policy output. The GPU-relative bound remains `0.2`; it was not
relaxed during this rebaseline.

TPU numerics depend on this compiler policy. TPU-vLLM's runtime fingerprint
covers the final `LIBTPU_INIT_ARGS` value and fails closed if the policy drifts;
the Levanter cluster jobs inherit the same Fray default.

The isolated TPU-vLLM launcher also fixes its transitive package release
universe with:

```text
uvx --exclude-newer 2026-07-20T00:00:00Z
```

The fork SHAs alone do not constrain later PyPI releases. This cutoff makes a
clean worker resolve the same versions as the qualification run, and the
runtime fingerprint still covers every resolved package version and registry
wheel `RECORD`. For packages installed from an immutable VCS commit, the digest
uses the `direct_url` commit instead of the rebuilt wheel's nondeterministic
`RECORD`; the full package inventory remains in each report for diagnosis.

The no-prefix numerical contract and the prefix-cache/concurrency contract
cover different execution graphs. In standing production captures, the
cache/concurrency graph reached `0.0430252316` probability error on
`code-humaneval-04`, beyond the no-prefix 256-token bound of `0.03`, while its
greedy token remained unchanged. The production contract does not widen or
reuse that numerical bound for unchanged reference tokens. It requires the
frozen continuation, observed cache hits, and reference concurrent winners;
if a concurrent near tie selects another token, the response's captured
logprobs must demonstrate that the alternate winner is probability-supported
within the unchanged `0.03` bound. This keeps cache-path numerical drift
visible without claiming that the no-prefix envelope covers a different graph.
