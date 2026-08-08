# Clean SM100 projected Selection

This adapter lowers the generic program

```text
Contract(index Q, index K)
-> positive scale Map
-> affine bottom-right DomainRestriction
-> maximum Fold over 128 right tokens
-> forced-local indexed Map
-> top-k Selection
```

through the fused projected-selection candidate in
`tile_lifetime.sm100_selection_lowering`.

## Source lineage

The physical implementation is derived from MiniMax MSA revision
`80434d7f67877c6570ca19cac444b84bc9855dac`.

The accepted runtime directly renders and compiles one compiler-selected
physical instantiation. It does not call MiniMax's variant manager. Retained
low-level machinery:

- the `OnlyScore` SM100 QK mainloop, TMA pipeline, mask application, and direct
  per-tile maximum store;
- the private shape planner, used only to derive generic CTA/work metadata;
- the generated `run_fmha_fwd` physical template instantiation;
- the prebuilt standalone `sparse_topk_select` histogram/insertion-selection
  primitive, registered as a generic top-k `Selection` rather than an MSA body;
- ascending selected-index output and out-of-range `-1` handling.

Removed semantic interface and dispatch:

- no call to public `fmha_sm100`;
- no call to `get_fmha_variant` in the accepted path;
- no call to `sparse_atten_func`, `sparse_fmha`, or a payload-attention entry;
- no V contraction, softmax state, PV contraction, or attention output;
- no MSA router/model identity in the lowering or physical dispatch.

For matched comparison only,
`benchmarks/backends/sm100_projected_selection_oracle.py` retains the private
`get_fmha_variant(...OnlyScore...)` path. That module is explicitly
oracle-derived/contaminated and is not reachable from the accepted backend.

The positive score-scale Map is moved after the maximum Fold by the generic
identity `max(scale * x) = scale * max(x)` for positive `scale`. This rewrite is
marked `real_algebra_equivalent`, and the adapter applies the scale to the
block-score buffer before Selection. The explicit causal offset is passed to
the physical planner. A generic indexed Map writes positive infinity to the
required local block before the generic Selection.

At the primary specification shape Q=K=16384, Hq/Hkv=64/4, the score-only
mainloop avoids the full token-score materialization and retains only FP32
block maxima between the generated Contract/Fold and generic Selection. The
smaller Q=256, K=16384, Hq/Hkv=32/8 configuration remains available through
CLI flags for fast bring-up.

## GB200 command

From a checkout containing pinned MSA and initialized CUTLASS submodules:

```bash
PYTHONPATH=lib/tile_lifetime/src \
python lib/tile_lifetime/benchmarks/sm100_fused_projected_selection.py \
  --msa-root /path/to/MSA \
  --execution-mode generated_only \
  --warmups 10 \
  --repeats 50 \
  --output /tmp/shuttle-fused-selection-gb200.json
```

For the shorter bring-up shape, add:

```text
--query-count 256 --right-count 16384 --query-heads 32 --key-value-heads 8
```

The generated-only command preserves raw distributions for three separately
reported boundaries:

- projected-selection core from already projected BF16 index Q/K;
- natural index route from BF16 hidden inputs through identical FP32-accumulating
  index-Q/index-K Contracts and BF16 casts on both sides;
- the acceptance path through projection, selection, relation scheduling,
  sparse QK/normalized-exp/PV, and deterministic partial-state merge.

Main QKV and output projections are excluded. The generated side records a
clean direct-source audit and zero external semantic kernels. Compare it with
the separately captured official MSA distribution under the same shape and
hardware conditions.

`--execution-mode paired` retains an in-process diagnostic against the old
private-variant adapter. On the pinned CUDA 13/GB200 stack, loading the direct
instantiation and the identical private variant in one process can produce a
module/template collision in the second runner. That oracle-only loader issue
does not occur in either isolated process, so it must not invalidate or alter
the generated path. Acceptance evidence uses isolated raw distributions and
records that limitation rather than claiming counterbalancing across the
collision.

The default selection policy is `real_algebra_equivalent`: the artifact records
materialized-reference relation mismatches and top-k cutoff margins instead of
requiring index identity after physical accumulation reordering. It still
reports deterministic relation hashes and final-output error against the
semantic reference. `--numerical-policy source_ordered` restores exact relation
acceptance. The absolute latency threshold is derived from the paired oracle
median in each capture rather than copied from an older shape.
