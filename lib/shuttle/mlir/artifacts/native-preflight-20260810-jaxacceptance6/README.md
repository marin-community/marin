# CPU ordinary-JAX acceptance preflight 6

This artifact records the single CPU-only Iris validation of canonical Marin
revision `6340706df454d699124fc7b676499f5db9cccd4a`. The job ran with 24 CPUs,
96 GiB of memory, 250 GiB of disk, a four-hour timeout, and zero retries. It did
not request or inspect an accelerator.

The run passed 55 focused Python tests, the fixture oracle, both exact-pin patch
stacks, the configured JAX cquery proof, all Shuttle native build gates, the
six-fixture audit through the built and cqueried `shuttle-test-opt`, four Shuttle
native tests, all 17 lit tests, and all four patched XLA tests. It built and
installed a release jaxlib 0.10.1 wheel with sha256
`7a47e8f2277516db57da0912a765877ec08fc4f8ad656bd0166888643281436a`.

The installed wheel passed the private observer Capture context-manager worker
on both normal and exception exits. The cache-disabled worker observed four
distinct ordinary `jax.jit` forward/VJP invocations under SOURCE_ORDERED and
FAST, retained immutable records after Capture close, and matched the disabled
JAX baselines bitwise. The populate worker created exactly four persistent cache
files. A fresh reuse process reported four public cache hits, reused those same
four files, emitted no observer event, and again matched the disabled baselines
bitwise. `acceptance-summary.txt` records the exact cache mapping, output hashes,
and observer evidence.

Iris resolved the sole task as succeeded with exit 0 after 40 minutes 28.74
seconds. There were no failures, retries, or preemptions. This is evidence for
the checked-in CPU `_jax` acceptance fixture only. It is not GPU validation,
performance evidence, or a claim about arbitrary JAX programs.

`raw-attempt.log.gz` is the complete 2,126-line controller-retained task log.
`task-describe.txt`, `task-events.txt`, `job-summary.txt`, and
`controller-terminal-proof.txt` record the terminal and release state.
`source-sha256.txt`, `toolchain.txt`, `bundle-proof.txt`, and `SHA256SUMS`
preserve the source, runner, payload, patch, wheel, client, and artifact hashes.
