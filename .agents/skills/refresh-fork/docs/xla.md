# The XLA fork: manual rebuild and re-pin

The `marin-community/xla` fork carries Marin's ragged all-to-all device-kernel
patches on the GPU PJRT plugin. It is deliberately NOT in
`config/external/migration.toml` and the weekly refresh never advances it: jax
pins an exact XLA revision by integrity hash, so the fork's base must be the XLA
commit the workspace's pinned `jax[cuda13]` release names, and it moves only
when that pin moves. `docs/dev-guide/forking-policy.md` ("The XLA fork") holds
the policy and rationale; this file is the operator procedure.

The pin is not a descriptor: the promoted wheel's URL lives in the `gpu` extra's
`jax-cuda13-pjrt` source in `lib/marin/pyproject.toml`, and `uv.lock` records
its hash. Every sibling package (`jax`, `jaxlib`, `jax-cuda13-plugin`) installs
stock at the pin; the wheel's `+marin.<sha>` local version still satisfies the
stock `jax-cuda13-pjrt==` requirement.

## Refresh after a jax bump

1. Find the new base: the `XLA_COMMIT` in `third_party/xla/revision.bzl` at the
   jax release tag the workspace pins. Rebase the fork's `main` onto it and set
   `jax_commit`/`jax_version` in `marin/release/config.json` in the same commit;
   the build rechecks the pairing and fails in seconds if they disagree.
2. Build: `marin-pjrt.yaml` builds `main` on push (or
   `gh workflow run marin-pjrt.yaml --repo marin-community/xla --ref main`) and
   publishes an unvalidated `marin-xla-pjrt-candidate-<sha12>` prerelease.
   Budget several hours; the compile dominates.
3. Promote:
   `gh workflow run marin-pjrt-promote.yaml --repo marin-community/xla -f candidate_tag=<tag>`.
   Promotion never recompiles; it proves on a GB200 that the device kernel
   engages, then republishes the same bytes as a pinnable release. Do not pin a
   wheel whose manifest lacks `device_kernel_engaged`: a build that leaves the
   kernel inert still passes ordinary correctness tests, including the e2e
   below.
4. Re-pin: point the `jax-cuda13-pjrt` source URL in `lib/marin/pyproject.toml`
   at the promoted wheel (keep the `+` percent-encoded as `%2B`) and run
   `uv lock`. `tests/test_moe_hero_ep.py::test_the_patched_pjrt_wheel_pairs_with_the_pinned_jax`
   fails until the URL's version prefix matches the `jax[cuda13]==` pin.

## End-to-end validation

`tests/cluster/grug/test_ragged_ep_check.py` is the 4-GPU ragged all-to-all
correctness gate on the standing GB200 cluster, `cluster`-marked; nothing runs
it on a schedule. It proves the transport is correct, not that the fork's kernel
engaged — that is what the promotion gate's `device_kernel_engaged` record is
for, so treat a green run as necessary and not sufficient.

```sh
uv run pytest tests/cluster/grug/test_ragged_ep_check.py \
  -m cluster -o addopts= --import-mode=importlib --timeout=0 -vv -s
```
