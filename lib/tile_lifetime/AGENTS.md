# Tile-Lifetime Compiler Agent Notes

Start with the root `AGENTS.md` and `TESTING.md`. This package contains a research compiler prototype and has no dependency on Levanter.

## Boundaries

- Keep the semantic IR independent of JAX, CUTLASS, ThunderKittens, and CUDA types.
- StableHLO import may depend on public JAX/jaxlib APIs, but raw MLIR values must not escape their MLIR context.
- Keep CUDA and CuTe dependencies optional and avoid importing them from CPU-only modules.
- Select among explicit expert skeleton and layout contracts before adding general GPU instruction synthesis.
- Every algebraic rewrite must report finite-precision effects and structured legality failures.

## Testing

- Test the public path from graph or StableHLO fixture to a structured plan.
- Compare numerical rewrites with an independent NumPy or JAX reference and report pointwise deviation metrics.
- Keep latency assertions out of pytest.
- Mark H100 correctness tests `slow` and benchmark/profiler tests `manual`.

```bash
uv run --frozen --package marin-tile-lifetime --group test pytest lib/tile_lifetime/tests
```
