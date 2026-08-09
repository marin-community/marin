# Unaccepted H100 twelve-call Grug replay

This artifact records the sole physical-H100 replay of the natural one-layer
Grug train-step boundary at source revision
`2732ef51a98ad36841f116c566b7e0272654b6e2`.

The generated path passed the harness's execution checks. All twelve selected
custom-call targets occur once in transformed HLO and every handler executed 35
times. The generated output is bitwise stable across all 30 measurements and
matches the ordered-floating-point correctness policy. Runtime dependency and
generated-source audits contain no Torch or Triton dependency.

The result is not accepted for two independent reasons:

1. Generated median latency is 0.658288001 ms versus 0.527479500 ms for the XLA
   baseline, a ratio of 1.247987838. This exceeds the 1.20 acceptance ceiling.
2. The XLA baseline produced two whole-tree hashes. One leaf varied:
   `[0].params.blocks[0].mlp_gated_norm.w_down` (leaf 15, float32,
   shape `[32, 128]`). Its usual hash appeared 29 times and its alternate hash
   appeared only in the final measured sample (index 29), whose baseline latency
   was 1.292757006 ms.

`execution-evidence.json.gz` preserves all 30 counterbalanced raw samples and
all 53 per-leaf hashes for each baseline and generated result. The evidence was
checkpointed with `execution_checks_passed` before nonessential summary
assembly. The outer artifact classification adds the performance and baseline
determinism gates above.

The benchmark used four warmups and 30 measured repetitions in
`shared_map_xla_remainder` mode. It made no replay retry. The requested CPU
count of one was normalized to four by Kubernetes. The allocation was
explicitly released after copying the raw artifact and then verified inactive.

Build caches, cubins, shared objects, PTX, Python bytecode, duplicate handler
copies, and attention AOT intermediates are excluded. The twelve
`generated_*.cu` files are the exact generated handler sources associated with
the twelve transformed-HLO targets.
