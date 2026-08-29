# DRAFT upstream report (jax) — NOT POSTED; for user review

Target: jax repo issue (jax._src.lax.parallel, ragged_all_to_all transpose rule). Found during
adversarial review of a Marin change that replicates the rule; confirmed independently by two
reviewers with a concrete counterexample and a CPU diagnostic. Latent at Marin's production
scale (empty groups essentially impossible at ~2048 tokens/group), but real wherever a shard
can hold an empty group: small models, early training, skewed routers.

---

Title: ragged_all_to_all transpose mis-masks the passthrough cotangent when a group is empty

The transpose rule for `ragged_all_to_all` (jax 0.11.1, `jax/_src/lax/parallel.py:1705-1710`)
zeroes the cotangent w.r.t. the `output` operand on the intervals the collective overwrote, by
marking interval starts and ends and prefix-summing:

```python
mask = jnp.zeros(...).at[output_offsets].set(1).at[output_offsets + recv_sizes].add(-1)
written = jnp.cumsum(mask)
```

When a receive size is zero, that group's start offset equals the next group's start offset
(offsets are unclipped running starts). The `.set(1)` at the shared offset is idempotent — one
mark for two group starts — but the empty group's `.add(-1)` at `offset + 0` lands on the same
position and cancels it. The running sum then reads 0 across intervals that genuinely were
written (cotangents leak through rows the collective overwrote) and can go negative, which
feeds `select_n` an out-of-range selector (implementation-defined lowering).

Concrete two-shard example: receive sizes `[0, 3, 2, 1]` with offsets `[0, 0, 3, 5]` produce
the implemented mask `[0, 0, 0, 0, 0, 0, -1]` where the correct written-mask is
`[1, 1, 1, 1, 1, 1, 0]`. A CPU reference-gradient check (differentiable emulation of the
collective vs the rule) shows gradient errors up to O(1) once empty groups exist.

Suggested fix: derive the mask from sizes rather than paired start/end marks, e.g. scatter-add
`+1` at starts and `-1` at `starts + sizes` only for `sizes > 0` (a `where` on the update
values), or build `written` directly from `searchsorted` over the cumulative sizes. Happy to
share the standalone repro script.
