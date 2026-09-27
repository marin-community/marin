# Coupled-WSPU runtime mixtures

Nine frozen policies cover Uncheatable cap 6 and Table-9 caps 6 and 8 at
interaction strengths 0.25, 0.5, and 1, with KL coefficient zero. Every policy
has 39 bucket counts summing to 2048 and exact weights `count / 2048`.

The producer rounds each continuous policy with the existing cap-aware count
allocator, then applies existing one-count exchange refinement using the
coupled objective. It stops when no transfer improves predicted BPB by more
than `1e-13`. This certifies local exchange optimality; the nonseparable
coupled objective does not have the original WSPU dynamic program's global
integer-optimality guarantee.

Run from the repository root:

```bash
UV_CACHE_DIR=/private/tmp/delphi-coupling-uv-cache uv run --offline python -m experiments.domain_phase_mix.exploratory.two_phase_many.materialize_delphi_coupling_runtime_20260906
```

The command checks frozen input and code hashes before regenerating these
artifacts. It reads no bank labels or live results and submits no jobs.

`candidate_weights.csv` is the launcher input. Its SHA256 is
`e72d1f2c9b3acc6ffde470dbd05db7ed2cf50d5a8c28fddfcad0d5d547c0f241`.
`candidate_mapping.json` records each continuous policy's identity, exact
runtime weights/counts and their hashes, prediction penalty, transfer distance,
and epoch usage. CSV metadata carries the same information without nested
weights/counts.

The largest continuous-to-runtime allocation transfer is 0.003040 and the
largest prediction penalty is 0.000110 BPB. All nine runtime mixtures are
distinct. `loader_validation.json` verifies the existing loader and separate
training-runtime epoch accounting; the maximum accounting discrepancy is
0.000003897 epochs, below its existing 0.00001 tolerance.

Applying the same refinement to saved additive κ0 optima reproduces all three
historical WSPU exact-DP runtime mixtures exactly. See
`kappa0_comparator_parity.json`. Those historical policies can therefore serve
as coordinate-matched κ0 comparators when data and trainer seeds also match.
