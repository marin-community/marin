# Debugging log for the H100 evidence runtime interpreter

The first contract-map evidence task authenticated its exact capsule and then
failed before GPU preflight because the launcher selected an interpreter
without NumPy. The negative result is pinned at
[`1f099189de`](https://github.com/marin-community/marin/commit/1f099189de71695c07a7a35096ef03f68cc0a199).

## Initial status

The task used `--no-sync`, so Iris created no `/app/.venv`. The exact persisted
entrypoint ran `/bin/sh -ceu` under Iris's `bash -lc` task wrapper and invoked
`python h100_contract_map_source_payload.py run`. The launcher restored the
capsule and passed its own `sys.executable` to the runner. The runner failed in
5.45 seconds at `tile_lifetime.attention`'s first NumPy import. No device query,
GPU preflight, compiler, profiler, or kernel ran, and the retry limit was zero.

## Hypothesis 1

The H100 image contained a coherent frozen environment at
`/opt/h100-evidence-runtime`, but the task command depended on `PATH`. The image
build smoke also used unqualified `python`, so it tested Docker's build
environment rather than Iris's login-shell launch boundary.

## Changes to make

Require an explicit absolute runtime interpreter in the source-capsule `run`
command and use it for the runner exec. Invoke the image's build-time library
and import probes with the same absolute interpreter. Exercise the transported
launcher and a synthetic capsule with a poisoned `PATH`; the selected
interpreter must import NumPy, JAX, and Shuttle and record its executable.

## Results

The transported-launcher regression and image-policy suite pass 26 tests. The
regression starts the launcher with a bootstrap interpreter, sets `PATH` to a
fake `python` that exits, and selects the test environment by absolute path.
The synthetic runner imports JAX 0.10.1, NumPy, and Shuttle 0.1.0 under the
selected interpreter; the fake `python` is not executed.

The full tile-lifetime suite passed 932 tests and hit one unrelated snapshot
failure. `test_generated_scan_snapshot_preserves_mutation_and_backend_boundary`
references
`mutation_per_key_r1_b1_t64_h32_k128_v128_bv32.stdout.log`, which is absent
from the canonical Git tree and ignored by the repository's `*.log` rule. No
image build, workflow dispatch, GPU query, or Iris launch is part of this
repair.

## Future work

- [ ] Build and independently review a new immutable H100 evidence image.
- [ ] Prepare a new exact-source capsule only after the repair is integrated.
- [ ] Request separate authorization before another single H100 launch.
