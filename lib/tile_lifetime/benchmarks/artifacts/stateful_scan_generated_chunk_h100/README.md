# Generated factored-chunk StatefulScan on H100

This checkpoint measures Shuttle's compiler-owned ordered factored-chunk path.
The environment deliberately did not contain FLA or FlashQLA. A generic
`RecoveredAffineStateUpdate` is converted into masked-triangular chunk factors,
then an ordered Triton skeleton consumes those factors. The numerical contract
is `bounded_reassociation`.

The primary shape is `B=1, T=2048, H=32, K=V=128`, BF16 factors, FP32 state,
scalar diagonal, and update rank one. Every JSON file retains every CUDA-event
sample, environment information, correctness errors, repeat hashes, and the
recovered factor signature.

| Candidate | Preparation | Execution | Combined |
|---|---:|---:|---:|
| C64, eager FP32 preparation, BV32 | 1.824976 ms | 0.156288 ms | 1.960832 ms |
| C64, compiled FP32 preparation, BV32 | 1.203648 ms | 0.156320 ms | 1.338656 ms |
| C32, compiled FP32/TF32 preparation, BV32 | 0.898928 ms | 0.256208 ms | 1.135616 ms |
| C16, compiled FP32/TF32 preparation, BV32 | 0.809824 ms | 0.339456 ms | 1.128192 ms |
| C16, compiled BF16 contraction preparation, BV32 | 0.665568 ms | 0.340032 ms | **0.984496 ms** |
| C16, compiled BF16 contraction preparation, BV64 | 0.661760 ms | 0.346800 ms | 0.988704 ms |

The pinned FLA chunk oracle is 0.510624 ms, so the best complete generated path
is 1.928x the oracle and misses the requested 1.2x target. Its isolated ordered
execution is 0.666x the oracle; summary preparation and its 84,410,368-byte
materialization are the measured gap. Further tile-size tuning is not the next
experiment. A fused producer/preparation backend or a representation that
forwards fewer prepared factors is required.

The best primary run has maximum absolute output/state errors of `4.883e-4`
and `2.840e-4` against Shuttle's compiler-owned source-ordered recurrent
skeleton. Output and final state repeat bitwise. The rank-two, per-key-diagonal
GPU mutation also repeats bitwise and has maximum absolute output/state errors
of `4.883e-4` and `2.668e-4`.

