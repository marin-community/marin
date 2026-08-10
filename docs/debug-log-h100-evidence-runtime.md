# H100 evidence image runtime closure

## Problem

The first reviewed H100 evidence image contained the pinned CUDA compiler and
profilers, but the `task-h100-evidence` target inherited the source-free `task`
image without a Python environment. The source capsule deliberately carries
tile-lifetime code but not its Shuttle dependency, so the runner could not
import JAX, NumPy, or `shuttle.ir`.

## Evidence

- `task` starts from `python:3.12-slim`, copies `uv`, and does not run a Python
  dependency installation.
- `task-h100-evidence` previously added only the hash-pinned NVIDIA packages.
- `lib/tile_lifetime/pyproject.toml` pins JAX 0.10.1 and depends on
  `marin-shuttle`; Shuttle pins jaxlib 0.10.1.
- The root lock already pins the matching JAX CUDA 13 plugin, PJRT package, and
  NVIDIA runtime wheel closure through the canonical `jax[cuda13]` extra.
- `contract_map_backend.py` imports `shuttle.ir`, while the reviewed capsule
  allowlist intentionally excludes `lib/shuttle`.

## Fix

A build-only stage now performs a frozen, non-editable sync for the
tile-lifetime dependency closure with its CUDA 13 extra. It builds Shuttle from
the exact workspace source and copies only `/opt/h100-evidence-runtime` into the
final task image. Tile-lifetime source remains capsule-owned, and `/opt` avoids
being hidden when Iris mounts the task bundle at `/app`.

The final image runs an import and exact-version smoke with
`JAX_PLATFORMS=cpu`. The smoke imports JAX, jaxlib, NumPy, SciPy, and
`shuttle.ir` without enumerating a device. Parsed policy tests bind the Docker
stage to the workspace manifests, Dockerfile-specific context allowlist, lock
versions, non-editable install, source boundary, and CPU-only smoke.

## Validation boundary

This is source and local policy-test evidence only. No image was built or
pushed, no workflow was dispatched, and no GPU or device API was queried.
