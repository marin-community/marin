# CPU ordinary-JAX acceptance preflight 5

This artifact records the single CPU-only Iris validation of canonical Marin
revision `e606e77594614fac33a547189e275c95a804a036`. The job ran with 24 CPUs,
96 GiB of memory, 250 GiB of disk, a four-hour timeout, and zero retries.

The run passed 55 focused Python tests, the fixture oracle, the configured JAX
cquery proof, all seven Shuttle build gates, the full six-fixture audit through
the built and cqueried `shuttle-test-opt`, four Shuttle native tests, 17 lit
tests, and all four patched XLA tests. It built and installed a release jaxlib
wheel with sha256
`9c200a26fed96ea2fb5b1c50b04e3cde50072b9b5653e02284532117dd69c495`.

The first failure occurred in the ordinary-JAX populate worker after its
observer Capture context body completed. Nanobind rejected the normal
`__exit__(None, None, None)` call. The populate report and cache-reuse worker did
not run to completion. This artifact makes no positive ordinary-JAX persistent
cache acceptance claim.

`raw-attempt.log.gz` is the complete 1,441-line task log returned by Iris.
`task-describe.txt`, `task-events.txt`, and `controller-terminal-proof.txt`
record the terminal task and controller state. `source-sha256.txt`,
`toolchain.txt`, and `bundle-proof.txt` preserve source, patch, runner, wheel,
and client hashes.
