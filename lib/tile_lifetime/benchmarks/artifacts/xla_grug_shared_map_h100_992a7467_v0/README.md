# Accepted H100 ten-call Grug replay

This artifact records the accepted physical-H100 replay of the natural one-layer
Grug train-step boundary at source revision
`992a7467da14886cbf03f40e27cd21f376f43125`.

The benchmark used four warmups and 30 measured repetitions in
`shared_map_xla_remainder` mode. The 30 measurement pairs used balanced order:
15 baseline-then-transformed and 15 transformed-then-baseline. The run invoked
each of the ten expected custom-call targets once in transformed HLO and each
handler 35 times. It made no retry.

`execution-evidence.json.gz` contains the evidence file written with status
`execution_checks_passed`
after target, handler, correctness, and determinism guards and before
nonessential summary assembly. It contains all raw timings, whole-output hashes,
and 53 per-leaf hashes for every sample.

The generated median was 0.689586497 ms versus 0.585042497 ms for the baseline,
or 1.178694712 times baseline. Outputs matched the ordered-floating-point
policy: maximum absolute error was 9.760260582e-7, mean absolute error was
7.977959460e-11, and 38 of 53 leaves were bitwise equal. Both paths produced one
whole-tree hash and one per-leaf hash set across all 30 repetitions.

The requested CPU count of one was normalized to four by the Kubernetes
resource configuration. The H100 allocation was explicitly released after the
artifact was copied, and a subsequent status check returned no active session.

Build caches, cubins, shared objects, and duplicate handler copies are excluded.
The ten `generated_*.cu` files are the exact generated sources associated with
the ten transformed-HLO targets. The full summary and execution evidence are
compressed losslessly to satisfy the repository's large-file gate.
