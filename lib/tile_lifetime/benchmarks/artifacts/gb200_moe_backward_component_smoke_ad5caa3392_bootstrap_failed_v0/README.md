# Generated expert reverse component smoke: bootstrap failure

This artifact records the sole authorized compile/correctness smoke from
Shuttle `ad5caa3392d4488f02e689f3319728ba1f55a8d4`. No GPU was allocated and no
benchmark or correctness kernel ran.

The first invocation failed command-line validation because the local guide
used the obsolete `--gpus-per-node` spelling. The corrected invocation used
one GB200, batch priority, 8 CPUs, 96 GB memory, and 100 GB disk. It failed
before holder submission because the local workspace bundle was 35.7 MB and
the controller client limits bundles to 25 MB. The fixed no-retry policy
therefore rejects this smoke.

Static and CPU gates remain green: 35 focused tests pass, the generated source
contains no atomics or semantic communication combine, the four-rank reference
matches the natural JAX VJP, and repeated CPU execution is bitwise stable. The
SM100 physical adapter is Torch-bound (`at::Tensor` and pybind), so it also
fails the final Torch-free runtime criterion even though its generated scalar
semantics and CPU reference do not import Torch.

`result.json` contains the exact commands, source hashes, gate states, and
release checks. There is no latency or GPU numerical result to interpret.
