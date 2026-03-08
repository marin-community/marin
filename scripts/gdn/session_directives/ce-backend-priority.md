# Session Directive: CE Backend First

Current diagnosis:
- Iteration 64 captured a real GDN train-shell win.
- Iteration 65 showed further standalone GDN-side reformulation was low impact.
- Iteration 66 showed an XLA-first outer-train-shell attempt was still slightly regressive.
- The remaining `while ~31.6 ms` is currently believed to be largely fused cross-entropy running on XLA.

Implications for this session:
- Before spending more mainline budget on standalone GDN-local work, force or compare the fused CE backend on the real training run.
- Treat CE backend selection as a first-class macro move.

Required behavior:
1. Record `CE backend selected: <impl>` in every profiled iteration writeup.
2. If available, record `CE-attributed while: <before> ms -> <after> ms`.
3. If CE backend is still `xla` and residual `while` stays large, the next iteration must be one of:
   - `P` CE backend forcing / A-B benchmark
   - `O` reduced-Pallas / XLA control arm
   - `M` XLA-first outer train path
4. Do not use a fresh mainline iteration on standalone `E/H/G/I/J/L` work while CE remains the unresolved dominant `while` source.

Preferred experiment matrix:
- champion code + CE default
- champion code + CE forced `pallas_tpu`
- current head + CE default
- current head + CE forced `pallas_tpu`

Goal:
- Determine whether the remaining wall is a backend-selection mistake or a deeper structural ceiling.
