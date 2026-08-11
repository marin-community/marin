# Debugging log for H100 cache identity

Add bounded evidence to the exact persistent-cache convergence failure without
changing the reviewed equality rule or using private JAX cache APIs.

## Initial status

The v11 H100 job completed the first case's profiling protocol and produced all
nine ordinary-XLA cache-root records. Every paired cold/hit root matched, but
the final exact convergence check observed more than one content identity. The
retained task log did not include the individual identities or cache summaries,
so it cannot distinguish semantic nondeterminism, phase-owned metadata, or an
overbroad hashing scope.

## Hypothesis 1

The existing cache identity hashes every public cache file's relative name and
content digest. The failure may come from different compiled content, different
file sets, or phase/process-owned cache metadata. The v11 artifact contains none
of the per-root values needed to choose among those explanations.

## Changes to make

- Record each root's public cache file count and total bytes alongside the
  existing content identity and final HLO.
- On exact-convergence failure, report the fixed nine phase/index roles, exact
  identities, final-HLO hashes, cache counts, and deterministic equality-class
  labels in a closed 4,096-character diagnostic.
- Keep cache paths, file names, raw HLO, and cache contents out of the error.
- Exercise the production cache-protocol boundary and mutations of its context,
  partitions, summaries, and serialization bound.

## Results

An isolated CPU-only JAX 0.10.1 reproduction used three clean public cache
roots, the runner's cache environment, deterministic NumPy inputs, BF16 device
conversion, and one JIT-compiled `tanh(x @ w)` executable. All three final-HLO
hashes were identical, and every root contained three files, but the three root
identities were distinct and their totals were 7,144, 7,145, and 7,140 bytes.
This shows that whole-root byte identity can vary across CPU compiler processes
without a final-HLO change. It does not establish which field differed on H100
or prove that H100 executable semantics differed.

The source audit also confirms that the current identity covers every file in
the worker's public cache root, including entries created during input setup;
it is not scoped to the target executable's cache entry. That makes hashing
scope a concrete hypothesis, but the v11 evidence still cannot choose among
scope, phase metadata, and compiled-content nondeterminism. No observed value
justifies normalization, so the exact one-class acceptance rule remains
unchanged.

The bounded diagnostic and production-boundary regressions pass locally. They
report nine fixed roots and hashes/summaries only; cache paths, file names, raw
HLO, and cache contents are excluded.

## Future work

- [ ] Use a reviewed v12 diagnostic, if needed, to classify the first mismatch.
