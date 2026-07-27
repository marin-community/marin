# Native package releases: post-publication lock drift

Diagnose the failed compatibility-floor update after the first unified package
release and preserve a uv-managed lockfile for later automated releases.

## Initial status

Run
[30223685542](https://github.com/marin-community/marin/actions/runs/30223685542)
published Dupekit, Finelog, Iris, and the Python library bundle through PyPI
Trusted Publishing. The final bump job failed before opening its auto-merge PR.
`uv lock --upgrade-package` changed 47 unrelated package entries, and the
targeted-change validator rejected the result.

The published native versions were:

- `marin-dupekit-native==0.1.4.dev30223685542`
- `marin-finelog-server==0.2.13.dev30223685542`
- `marin-iris-native==0.1.5.dev30223685542`

## Hypothesis 1: PyPI metadata forced a wider resolution

The old and new Dupekit native distributions both require `pyarrow>=23.0.0`.
Plain `uv lock` after raising the Dupekit floor produced the same CUDA, JAX,
Torch, and VLLM changes as `uv lock --upgrade-package`. The extra
`--refresh-package` flag was not responsible.

## Hypothesis 2: the existing lock contained stale Git metadata

Refreshing VLLM without changing a native floor reproduced the wider update:

```text
uv lock --upgrade-package vllm
Updated vllm v0.20.1rc1.dev1465+gafb267194.tpu (afb26719) -> v1+tpu (afb26719)
```

The commit SHA stayed at `afb26719`, but the distribution version and dependency
graph changed. Resolving that graph added or changed CUDA, JAX, Torch, and their
workspace consumers. `uv lock --check` had accepted the old lock because it
reuses locked Git metadata until that package is refreshed or another project
change requires a new resolution.

The previous targeted native-floor update in
[Keep federated capability URLs under `/proxy/t`](https://github.com/marin-community/marin/pull/7634)
briefly generated and then reverted a full lock refresh before committing only
the native lock entries. That preserved the stale VLLM metadata.

## Changes to make

Generate the pending dependency resolution once with uv and review it in
[Use uv for native dependency floors](https://github.com/marin-community/marin/pull/7666).
Carry the three already-published native floors in the same PR. Release-tooling
changes and dependency-floor changes are excluded from the release workflow's
`push.paths`, so merging this recovery does not publish another package set.

Use uv's workspace command for later automated updates:

```text
uv add --package <owner> <distribution>>=<version> \
  --upgrade-package <distribution>==<version> --no-sync
```

Keep the targeted-change validator before enabling auto-merge. A native release
must not silently turn its floor PR into a CUDA, JAX, Torch, or VLLM upgrade.

## Results

After `uv lock --upgrade-package vllm` normalized the lock, the exact published
versions were applied with `uv add`. Each update changed only the expected
package pair:

- Dupekit: `marin-dupekit`, `marin-dupekit-native`
- Finelog: `marin-iris`, `marin-finelog-server`
- Iris: `marin-iris`, `marin-iris-native`

`uv lock --check` accepted the final 610-package lock. The hand-written lock
block merger was removed from
[Use uv for native dependency floors](https://github.com/marin-community/marin/pull/7666).

## Future work

- [ ] Revisit an exact uv pin if resolver drift recurs across uv upgrades.
