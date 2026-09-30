# mfu30 trace-analysis toolkit

Run from the repo root with `uv run python`. `PROF=<dir with *.xplane.pb>`.

1. `tfop_dump.py $PROF/x.xplane.pb autoresearch/loop-260930-mfu30 $PROF/rows.pkl` — per-kernel rows (stream, name, interval, hlo_op) + train-step launches.
2. `hlo_opnames.py $PROF/x.xplane.pb autoresearch/loop-260930-mfu30 $PROF/opnames.pkl` — HLO instruction -> JAX op_name scope, decoded from the xplane's embedded HloProto.
3. `anatomy.py $PROF/rows.pkl $PROF/opnames.pkl` — per-scope compute-stream time (fwd/remat/bwd), collective busy/exposed by scope, exposed memcpy, idle.
4. `hlo_extract.py`-style text dump then `gemm_flops.py <hlo.txt> gemmflops.pkl` + `gemm_eff.py rows opnames gemmflops <dir>` — cuBLAS GEMM PF/s per scope, split by overlap with ragged all-to-all.
5. `fusion_bytes.py <hlo.txt> fusionbytes.pkl` + `membw.py rows opnames fusionbytes <dir>` — approximate HBM TB/s of non-GEMM fusions (operand bytes are upper bounds).
6. `exposed_detail.py rows opnames <dir>` — the u32 all-reduce, exposed memcpys by op, and their position in the step.

The compute stream is `Stream #17` on these traces; the collective stream is `#83`.
