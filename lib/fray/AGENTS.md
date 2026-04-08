# Fray Agent Notes

Distributed execution abstraction layer. Start with the shared instructions in `/AGENTS.md`; only Fray-specific conventions are below.

## Key Docs

- Archived: `.agents/project/20260130_fray_lite_design.md` — v2 API design (implemented; read code instead)

## Source Layout

- `src/fray/__init__.py` — v2 public API exports (recommended interface)
- `src/fray/v2/client.py` — `Client` protocol, `current_client()`, auto-detection
- `src/fray/v2/types.py` — `JobRequest`, `ResourceConfig`, `DeviceConfig` (CPU/GPU/TPU)
- `src/fray/v2/actor.py` — `ActorHandle`, `ActorGroup`, actor hosting
- `src/fray/v2/ray_backend/` — Ray backend (`submit`, `host_actor`)
- `src/fray/v2/iris_backend.py` — Iris backend
- `src/fray/v2/local_backend.py` — Local/thread backend (testing)
- `src/fray/v2/device_flops.py` — TPU/GPU flops calculation
- `src/fray/v1/` — Legacy API (Cluster, JobContext, Queue) — do not extend
- `src/fray/cluster/` — Shared TPU orchestration (`run_on_pod`, gang scheduling)

## Conventions

- **v2 is the production API.** All new code should use `fray.v2`. Do not add features to v1.
- Always use the `Client` protocol, not concrete backend implementations.
- Actor resources: set `num_cpus=0` on actors to avoid head-node resource contention.
- Testing: use `LocalClient` for unit tests. Only use Ray/Iris backends for integration tests.
- No backward compatibility shims between v1 and v2 — migrate callers directly.
