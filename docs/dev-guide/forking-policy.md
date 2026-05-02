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
| vllm-tpu | [`marin-community/vllm-tpu`](https://github.com/marin-community/vllm-tpu) | PyPI version pin (`vllm-tpu==0.18.0`) in `lib/marin/pyproject.toml` |

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
