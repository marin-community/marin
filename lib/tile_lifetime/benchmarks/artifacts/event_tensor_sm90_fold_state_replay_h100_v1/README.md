# SM90 Event Tensor replay after Fold-state repair

This artifact records the H100 replay of Shuttle commit
`ad1a0c319245022e759ee7d79d9f2662c1b1940f`. The commit carries the
normalized-exponential register state across each CuTe update before entering
the generated child loop. The former MLIR dominance failure is gone.

The run used one NVIDIA H100 80GB HBM3 (compute capability 9.0), driver
595.71.05, two host CPU cores, and 32 GB of host memory. This was an H100 run,
not a B200 or GB200 run.

The exact former blocker command compiled, passed sampled semantic checks, and
produced bitwise-stable output hashes. A second comparison used the pre-Event
source at `31e673a1a992af9e89dc005c46dcdc4f4120c28f` with the identical repaired
`cute_normalized_exp.py` copied in. Both trees therefore used helper SHA256
`b164e9a3fc87cb35e47be61917f983c14d94406e811d6eb28e1666b8d14095ca`.

Two 10-sample captures reversed process order:

```text
capture 1: canonical -> Event Tensor
capture 2: Event Tensor -> canonical
```

Pooled medians were 0.080272 ms for the canonical source and 0.080352 ms for
the Event Tensor source, a ratio of 1.000997. Both paths had maximum sampled
absolute error 0.015625 and mean sampled absolute error 0.000114395. Their
deterministic output hashes were identical.

`result.json` retains every timing sample, compile time, source hash,
correctness statistic, and backend hash. The comparison establishes that the
derived Event Tensor attachment is performance-neutral at this shape after the
Fold-state repair. It does not benchmark another sequence length or another GPU
architecture.

