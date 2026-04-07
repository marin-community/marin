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

2. **Set up CI to autobuild wheels on push to the canonical Marin branch.**
   Every push to the branch Marin actually builds from must produce a
   versioned wheel via GitHub Actions (or equivalent CI). There should be no
   manual "run this script to get new wheels" step.

   Use one of these branch layouts:

   - **Simple layout**: use `main` as the canonical Marin branch. Build wheels
     from `main`.
   - **Split layout**: keep `main` as the upstream-tracking branch and use a
     dedicated integration branch such as `marin-main` for Marin-only changes.
     Build wheels from `marin-main`.

   If you use the split layout, make the branch roles explicit in the repo's
   README, release workflow, and any Marin dependency pins.

3. **Pin by version hash in Marin's `pyproject.toml`.**
   Marin should depend on the fork via a wheel URL or a git dependency with a
   pinned revision from the canonical Marin branch, so that `uv sync` runs
   quickly and results are cacheable. Example from `pyproject.toml`:

   ```toml
   harbor = { git = "https://github.com/marin-community/harbor.git", rev = "354692d..." }
   ```

## Existing Forks

| Package | Repository | How Marin depends on it |
|---------|-----------|------------------------|
| Harbor  | [`marin-community/harbor`](https://github.com/marin-community/harbor) | Git dependency pinned to a commit rev in the root `pyproject.toml` |
| vllm-tpu | [`marin-community/vllm-tpu`](https://github.com/marin-community/vllm-tpu) | PyPI version pin (`vllm-tpu==0.13.2.post6`) in `lib/marin/pyproject.toml` |

## When to Fork

Fork only when:

- You need patches that upstream has not (or will not) accept.
- The upstream release cadence is too slow for a fix you need now.
- You need a custom build (e.g. TPU-specific wheels).

Prefer upstreaming changes when possible. A fork is ongoing maintenance
overhead — even with agents helping, keeping it in sync takes effort.

## Maintaining a Fork

- Keep the upstream-tracking branch rebased or merged regularly against
  upstream.
- If `main` is the canonical Marin branch, that means keeping `main` close to
  upstream.
- If you use a split layout (`main` + `marin-main`), keep `main` close to
  upstream and regularly rebase or merge `main` into `marin-main`.
- Tag releases that correspond to the pinned revisions in Marin so the
  provenance is clear.
- When bumping the pinned revision in Marin, update the rev in
  `pyproject.toml` and run `uv sync` to verify resolution.
