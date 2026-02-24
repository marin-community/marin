Triangular inversion focus for this session:

- Target the strict lower-triangular inversion bottleneck in GDN kernels with high-upside redesigns.
- Bold changes are explicitly allowed if they preserve end-to-end model behavior and improve throughput/MFU.
- Equivalent mathematical reformulations are in scope, including approaches that avoid explicit chunk-size-by-chunk-size matrix inversion.

Optimization direction:

- Prefer numerically stable and parallel TPU-friendly formulations over sequential forward-substitution style methods.
- Consider reformulating the recurrence or chunk update so it is solved via associative/blockwise composition or another equivalent decomposition.
- It is acceptable to replace explicit inversion with an equivalent solve/transform pipeline if correctness is preserved.
- Preserve deployability on target training configs; avoid designs that only work at tiny settings or that repeatedly fail with scoped-vmem OOM.

Validation bar:

- Keep correctness tests green (`test_gdn_kernels.py`, `test_gdn_layer.py`).
- Verify end-to-end behavior remains aligned with reference implementation semantics.
- Require a completed profile run with measured throughput/MFU change before claiming success.

Numerics guardrails:

- Be careful with numerically unstable inversion shortcuts (for example naive high-power series without stabilization).
- If using approximation or preconditioning, explicitly validate numerical error and training stability impact.
