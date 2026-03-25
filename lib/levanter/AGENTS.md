# Levanter Agent Notes

JAX-based language model training library using Haliax named tensors and Equinox modules. Start with the shared instructions in `../../AGENTS.md`; only Levanter-specific conventions are below.

## Key Docs

- `docs/` — full documentation site (update `mkdocs.yml` when adding sections)
- `docs/` — model porting guides (see `docs/` for architecture-specific guides)
- `docs/design/jit-safety.md` — JIT safety rules (must-read before writing jitted code)

## Development

```bash
# Default test suite
uv run pytest tests -m "not entry and not slow and not ray"
```

- Mark long-running tests with `@pytest.mark.slow`.
- Protect PyTorch-dependent tests with `@skip_if_no_torch`.
- Batch sizes should be a low multiple of `len(jax.devices())` to ensure multi-device correctness.
- Do not relax numerical tolerances without prior agreement from a human. Prefer `assert_allclose` with 1e-4 for complex modules and 1e-5 for simpler ones.

### TPU VMEM flags for kernel tuning

- `v5p`/`v5e`: `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`
- `v6e`: `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304`
- `v4`: no special flag unless user explicitly asks.

## Code Conventions

- **Named tensors:** Use `NamedArray` and explicit `Axis` objects; operate over named axes.
- **Modules:** Use Equinox and Haliax — avoid Flax and Haiku in new code.
- **Configurations:** Dataclasses loaded via `draccus`. Keep them declarative and typed.
- **Datasets:** Prefer `AsyncDataset` over `SyncDataset` unless there is a concrete reason not to.
- **Compile performance:** Prefer `Stacked` with `fold` or `scan` over hand-written loops.
- **Reproducibility:** Aim for deterministic training; use explicit PRNG keys.
- **JIT safety:** Follow `docs/design/jit-safety.md` for all jitted code paths.
- Maintain compatibility with both GPU and TPU backends.
