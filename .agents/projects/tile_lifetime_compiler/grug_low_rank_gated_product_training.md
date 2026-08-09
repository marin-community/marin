# Grug low-rank gated-product training recovery

## Result

Shuttle now forms exact typed-FFI replacement boundaries for the repeated
low-rank gated-product forward, rematerialization, and JAX-owned reverse
structure in the pinned natural post-SPMD Grug HLO. Recovery uses only shapes,
contraction dimensions, scalar dataflow, liveness, and shared parameter
origins. Removing all HLO metadata produces the same boundary families and
ABIs.

The generated physical composition now follows the existing thirteen-call
routed, normalized-exponential, attention-reverse, and axis-Fold rewrite. It
normalizes the ten logical boundaries onto one shape/AST-only physical family,
then generates and registers one forward and one reverse typed-FFI handler.
This checkpoint does not claim whole-step GPU execution or latency.

## Generic algebra

The recovered forward family is:

~~~
h0 = Contract(x, w0)                 BF16
h1 = Map(h0)                         source-ordered BF16 scalar AST
g0 = Contract(h1, w1)                BF16
y  = Map(x, g0)                      source-ordered BF16 scalar AST
~~~

In the pinned program, the first Map is SiLU and the output Map is
'x * sigmoid(g0)'. These functions are imported scalar programs, not dispatch
keys. Replacing the hidden Map with tanh regenerates a different CUDA scalar
body through the same generator while preserving both Contract plans.

JAX differentiation exposes the corresponding reverse:

~~~
dg0  = Map(x, g0, dy)
dh1  = Contract(dg0, transpose(w1))
dh0  = Map(h0, dh1)
dx0  = Contract(dh0, transpose(w0))
dx   = Map(dx0, g0, dy)
dw0  = Contract(transpose(x), dh0)
dw1  = Contract(transpose(h1), dg0)
~~~

Shuttle imports all three reverse Maps as generic scalar ASTs. It does not
differentiate the forward program itself. JAX remains responsible for AD.

## Repeated instances

The natural HLO contains six forward realizations:

| Realization | First Contract | Second Contract | Role |
| --- | --- | --- | --- |
| 0 | dot.8 | dot.9 | primal |
| 1 | dot.10 | dot.11 | primal |
| 2 | dot.19 | dot.20 | primal |
| 3 | dot.21 | dot.22 | primal |
| 4 | dot.25 | dot.26 | rematerialized |
| 5 | dot.34 | dot.35 | rematerialized |

The table records physical evidence, not matching keys. The metadata-stripped
test recovers the same six realizations. Shared parameter-origin tracing groups
them into four logical families: two have one forward realization and two have
both a primal and rematerialized realization.

Each logical family has one JAX-owned reverse. The four reverse families contain
eight input-adjoint Contracts and eight weight-adjoint Contracts:

| Primal/rematerialized first Contract | Input-adjoint pair | Weight-adjoint pair |
| --- | --- | --- |
| dot.8 | dot.48, dot.49 | dot.70, dot.71 |
| dot.21 | dot.23, dot.24 | dot.85, dot.86 |
| dot.25 | dot.46, dot.47 | dot.72, dot.73 |
| dot.34 | dot.36, dot.37 | dot.79, dot.80 |

These names are diagnostics. Association is proved by parameter origin,
Contract dimensions, scalar reachability between the two input-adjoint
Contracts, and equality of the layout-stripped scalar values consumed by the
weight adjoints.

## Exact logical replacement boundaries

The maximal bounded decomposition needs ten generic calls:

| Boundary | Calls | Inputs per call | Outputs per call | Contracts per call |
| --- | ---: | ---: | ---: | ---: |
| Forward/rematerialization | 6 | 4 | `5,1,1,5,5,5` | 2 |
| JAX-owned reverse | 4 | 9 | 3 | 4 |

The two forward-only realizations return only their final BF16 value. The four
realizations used by JAX's reverse also return the exact BF16 pre-Map, sigmoid,
and hidden values already live in the source program. This preserves the
existing save/rematerialize policy and cast ordering. It does not silently
replace saved values with recomputation.

Each reverse call consumes those source-order values, the two weights, the
incoming cotangent, and the required physical input views. It returns:

~~~
input adjoint
down-weight adjoint
up-weight adjoint
~~~

The reverse scalar Maps are re-imported from the saved-value cut. JAX continues
to own differentiation; Shuttle does not differentiate the forward AST.

## Live work after current replacements

The accounting starts from the checked-in thirteen-call transformed HLO
artifact. Root reachability leaves:

| Static HLO work | Contract count | Dot FLOPs |
| --- | ---: | ---: |
| All live Contracts | 52 | 2,232,320 |
| Recovered gated-product family | 28 | 1,835,008 |
| Remaining | 24 | 397,312 |

The recovered family is 53.8% of live Contract instructions and 82.2% of live
dot FLOPs in this small padded fixture. Its 28 Contracts split into:

~~~
6 forward/rematerialized plans × 2 Contracts = 12
4 input-adjoint plans × 2 Contracts          =  8
4 weight-adjoint plans × 2 Contracts         =  8
~~~

This is static HLO accounting, not a production-shape throughput estimate.
The ten generated calls replace all 28 Contracts. The transformed HLO has 23
generated calls in total: the existing thirteen plus six forward/rematerialized
and four reverse calls. This is exact structural accounting, not a recommended
physical kernelization.

## Generated physical ABI

The logical boundaries contain redundant rank-3 and rank-2 views of the same
buffers. The generated physical ABI leaves those views under XLA and calls one
rank-2 family:

~~~
forward(x[8,32], w0[32,128], w1[128,32])
  -> y[8,32], h0[8,128], hidden[8,128], g0[8,32]

reverse(x, w0, w1, h0, hidden, g0, dy[8,32])
  -> dx[8,32], dw0[32,128]{0,1}, dw1[128,32]{0,1}
~~~

Six forward/rematerialization calls reuse one generated forward target. Four
JAX-owned reverse calls reuse one generated reverse target. The rewrite restores
the original rank-3 output and input-adjoint views outside typed FFI. It routes
the three generated BF16 save values directly from each relevant forward call
to its reverse call.

The recovered weight-adjoint layouts are dimension-zero-minor `{0,1}`. The
generic Contract program records these layouts, the CUDA generator writes each
logical element to the matching physical offset, and the JAX component wrapper
requests the same output layouts.

## Numerical and placement boundaries

The physical program places BF16 boundaries at:

* the input to each forward Contract;
* both forward Contract outputs;
* the imported hidden and output Maps;
* both input-adjoint Contract outputs;
* the final input adjoint;
* both weight-adjoint Contract outputs.

Weight adjoints are BF16 at the Contract boundary and are converted to FP32 by
the existing optimizer path. The scalar ASTs carry source_ordered; no
real-algebra reassociation is claimed.

All ten placement all-reduces remain byte-for-byte outside the replacement
regions. The audit records the exact cotangent inputs and their upstream
collective paths. Input-adjoint outputs feed the surrounding normalization
reverse. BF16 weight adjoints feed the ordinary FP32 optimizer conversion and
update.

## Physical ownership boundary

A generic physical implementation should compose the existing Contract
skeleton with generated scalar Maps:

~~~
forward:
    Contract -> generated Map -> Contract -> generated product Map

reverse:
    generated Map -> Contract -> generated Map -> Contract
    + generated residual Map
    + two generic weight-gradient Contracts
~~~

The generated composition preserves JAX's save/rematerialization choices and
keeps collectives outside. Every call has an
exact multi-output ABI, every old scalar instruction is absent or dead after
replacement, and all 28 old dot instructions cease to be live Contracts. The
source contains generated scalar ASTs, fixed ordered FP32 accumulation,
explicit BF16 RNE boundaries, no atomics, and no opaque semantic kernel
dependency.

A tanh mutation changes the generated hidden scalar AST while retaining the
same boundary-family digest, call target, input ABI, output ABI, and Contract
shapes. This demonstrates that the boundary follows generic Contract/Map
structure rather than a workload name.

## Reproduction

~~~
uv run --frozen --package marin-tile-lifetime --group test pytest -q \
  lib/tile_lifetime/tests/test_xla_low_rank_gated_product.py
~~~

The focused tests consume the frozen thirteen-call natural Grug HLO, perform
root-reachable accounting, remove frontend metadata, check every numerical and
collective boundary, round-trip all ten generated typed-FFI calls, enforce the
six-forward/four-reverse target multiplicities, audit the 23-call composition,
and exercise a hidden-Map mutation through the same scalar generator.
