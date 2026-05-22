# Fray Agent Notes

Distributed execution abstraction layer. Start with the shared instructions in `/AGENTS.md`; only Fray-specific conventions are below.

## Key Docs

- Archived: `.agents/projects/20260130_fray_lite_design.md` — original API design (implemented; read code instead)

## Source Layout

- `src/fray/__init__.py` — public API exports (recommended interface)
- `src/fray/client.py` — `Client` protocol, `current_client()`, auto-detection
- `src/fray/types.py` — `JobRequest`, `ResourceConfig`, `DeviceConfig` (CPU/GPU/TPU)
- `src/fray/actor.py` — `ActorHandle`, `ActorGroup`, actor hosting
- `src/fray/iris_backend.py` — Iris backend
- `src/fray/local_backend.py` — Local/thread backend (testing)
- `src/fray/device_flops.py` — TPU/GPU flops calculation
- `src/fray/cluster/` — type re-exports (back-compat shim for `from fray.cluster import ResourceConfig`)

## Conventions

- Always use the `Client` protocol, not concrete backend implementations.
- Actor resources: set `num_cpus=0` on actors to avoid head-node resource contention.
- Testing: use `LocalClient` for unit tests. Only use the Iris backend for integration tests.
