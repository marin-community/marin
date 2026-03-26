# Forking Policy

When Marin needs a modified version of an upstream package, we maintain a fork
under the `marin-community` GitHub organization rather than vendoring code into
the Marin monorepo. This page documents the requirements for creating and
maintaining such forks.

## Requirements

1. **Host the fork in `marin-community/<package>`.**
   Do not pull the forked source tree into the Marin repo. Keep it as a
   standalone repository under the organization (e.g.
   `marin-community/harbor`, `marin-community/vllm-tpu`).

2. **Set up CI to autobuild wheels on push to main.**
   Every push to the fork's `main` branch must produce a versioned wheel via
   GitHub Actions (or equivalent CI). There should be no manual "run this
   script to get new wheels" step.

3. **Pin by version hash in Marin's `pyproject.toml`.**
   Marin should depend on the fork via a wheel URL or a git dependency with a
   pinned revision, so that `uv sync` runs quickly and results are cacheable.
   Example from `pyproject.toml`:

   ```toml
   harbor = { git = "https://github.com/marin-community/harbor.git", rev = "354692d..." }
   ```

## Existing Forks

| Package | Repository | How Marin depends on it |
|---------|-----------|------------------------|
| Harbor  | [`marin-community/harbor`](https://github.com/marin-community/harbor) | Git dependency pinned to a commit rev in the root `pyproject.toml` |
| tpu-inference | [`marin-community/tpu-inference`](https://github.com/marin-community/tpu-inference) | Git dependency pinned to a commit on the `marin` branch in `lib/marin/pyproject.toml` |

Note: Marin depends on the bundled `vllm-tpu` wheel from PyPI
(`vllm-tpu==0.13.3`), which packages both vLLM and `tpu-inference`
together. We override `tpu-inference` with our fork via an
`override-dependencies` entry in the root `pyproject.toml` so we can
iterate on TPU-specific fixes (fast bootstrap, KV cache, etc.)
independently of the vllm-tpu release cadence.

## When to Fork

Fork only when:

- You need patches that upstream has not (or will not) accept.
- The upstream release cadence is too slow for a fix you need now.
- You need a custom build (e.g. TPU-specific wheels).

Prefer upstreaming changes when possible. A fork is ongoing maintenance
overhead — even with agents helping, keeping it in sync takes effort.

## Maintaining a Fork

- Keep the fork's `main` branch rebased or merged regularly against upstream.
- Tag releases that correspond to the pinned revisions in Marin so the
  provenance is clear.
- When bumping the pinned revision in Marin, update the rev in
  `pyproject.toml` and run `uv sync` to verify resolution.

## tpu-inference Fork Strategy

The `tpu-inference` fork (`marin-community/tpu-inference`) carries patches
for fast TPU startup, KV cache bootstrap, and model loading that upstream
(`vllm-project/tpu-inference`) does not yet support. The goal is to
upstream all changes over time while keeping Marin unblocked.

### Branch layout

```
upstream main ──────────────────────────────────►
                 \                       \
fork main ────────●────────────────────────●────► (syncs with upstream)
                   \          \
                    \          feat/kv-cache-bootstrap ──► upstream PR
                     \
                      feat/abstract-dummy-bootstrap ──► upstream PR

marin branch ────────●────●─────────────────────► (merges all feature branches)
                     ↑    ↑
               merge feat branches
```

- **`main`** — tracks upstream `main`. Periodically merge
  `upstream/main` into it. Never commit Marin-specific changes here.
  This branch exists so feature branches start from a clean upstream
  base and produce clean upstream PRs.

- **`feat/*` branches** — one per logical change, branched from `main`.
  Each one should be independently PR-able to `vllm-project/tpu-inference`.
  Keep them focused and small.

- **`marin`** — the integration branch. Merges all `feat/*` branches
  together. This is what Marin's `pyproject.toml` pins to. It always
  contains the full set of Marin-needed patches on top of the latest
  upstream.

### Workflow

1. **New fix needed**: branch `feat/my-fix` from `main`, implement,
   push to the fork.
2. **Integrate into Marin**: merge `feat/my-fix` into `marin` branch,
   update Marin's pinned rev.
3. **Upstream the fix**: open a PR from `feat/my-fix` against
   `vllm-project/tpu-inference`. The PR is clean because `main` tracks
   upstream and the feature branch has no unrelated changes.
4. **Upstream accepts**: the fix lands in upstream `main`. Sync fork's
   `main`, and the `marin` branch's delta shrinks naturally. Drop the
   merged feature branch.
5. **Upstream updates**: periodically merge `upstream/main` into fork's
   `main`, then merge `main` into `marin` to pick up upstream
   improvements.

### Why not just commit to fork's `main`?

If the fork's `main` diverges from upstream, every upstream PR includes
all unrelated Marin patches. Keeping `main` clean means each feature
branch produces a minimal, reviewable upstream PR. The `marin` branch
absorbs the integration cost so `main` stays pristine.

### Current feature branches

| Branch | Purpose | Upstream PR |
|--------|---------|-------------|
| `feat/fast-tpu-bootstrap-v0.13.2` | Abstract dummy model bootstrap, bootstrap-safe RNG (rebased on v0.13.2 tag) | Not yet filed |
