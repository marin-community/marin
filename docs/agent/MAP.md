# Marin Agent Reference

## Libraries

| Library | Purpose |
|---------|---------|
| **marin** | Data processing: classification, dedup, decontamination, tokenization, transforms |
| **zephyr** | Lazy dataset pipelines (`Dataset.map/flat_map/group_by`), executed by `ZephyrContext` |
| **dupekit** | Rust FFI: fast hashing (xxHash3) and MinHash/LSH dedup primitives |
| **iris** | Job orchestration: controller, autoscaling, gRPC workers |
| **fray** | RPC actors and resource configs (`ResourceConfig`) |
| **rigging** | Storage abstraction (GCS, S3, local) via fsspec; distributed locking |
| **levanter** | LLM training (JAX): Llama, Gemma, Qwen with distributed training |
| **haliax** | Named tensors for JAX |

## Package Index

*Run `./scripts/generate_agent_docs.py` to populate package docs.*

To read a package doc: `@docs/agent/packages/<package_name>.md`

## Conventions

- **Config style:** Draccus dataclasses; compose sub-configs via embedding, not inheritance.
- **Artifact paths:** Rooted at `MARIN_PREFIX`; output paths constructed per-step.
- **Execution:** Steps described via `StepSpec`; remote execution via `RemoteCallable`.
- **Dedup output contract:** Dedup functions write attribute sidecars marking duplicates, not filtered corpora.
- **No backward compat shims:** Update all call sites; no `hasattr` guards.
- **Imports:** All at top of file; no local imports except to break cycles.

## Dependency Direction

`{iris, haliax}` → `{levanter, zephyr}` → `marin`

Each layer may only import from layers to its left.
