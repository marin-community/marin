# Snowball TPU vLLM numerical policy

Snowball's TPU vLLM contract uses the production v6e libtpu policy:

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

The v2 TPU snapshot and bucket bounds were therefore re-derived through the
standing-cluster path with the production setting active. TPU numerics depend
on this compiler policy: the contract's runtime fingerprint covers the final
`LIBTPU_INIT_ARGS` value and fails closed if the policy drifts.

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
