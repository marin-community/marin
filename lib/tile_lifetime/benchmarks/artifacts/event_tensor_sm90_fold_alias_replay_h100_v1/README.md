# SM90 normalized-Fold alias replay

This artifact records an actual H100 replay of the normalized-exponential
register-alias fix at Shuttle commit
`9fe460e7db2bc563e9c324d421ae5fd6560a1e68`.

The replay used one NVIDIA H100 80GB HBM3 (compute capability 9.0), driver
595.71.05, and the package versions in `environment.txt`. This was an H100
run, not a B200 or GB200 run. Host allocation was limited to two CPU cores and
32 GB of memory.

The exact former blocker command was replayed:

```bash
export PYTHONPATH=/tmp/shuttle/lib/tile_lifetime:/tmp/shuttle/lib/tile_lifetime/src
/tmp/event-env/bin/python \
  /tmp/shuttle/lib/tile_lifetime/benchmarks/h100_cute_generated_streaming_attention.py \
  --sequence-length 2048 \
  --warmups 2 \
  --repeats 3 \
  --json-output /tmp/fix-smoke.json
```

Compilation still fails before correctness or timing samples. Binding
`self.row_sum`, `self.row_max`, and `self.scale_log2` to Python aliases outside
the finalization loop does not change the generated CuTe SSA region. The
`cute.get_layout` use for the row-sum register tensor still observes an operand
defined in a child region and fails MLIR dominance verification.

`stderr.txt` preserves the complete sanitized compiler diagnostic. No private
cluster, pod, task, or device UUID is retained. There is no latency claim from
this artifact.

