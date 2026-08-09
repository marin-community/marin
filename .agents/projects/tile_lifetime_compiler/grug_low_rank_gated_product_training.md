# Grug low-rank gated-product training recovery

## Result

Shuttle now recovers the repeated low-rank gated-product forward and JAX-owned
reverse structure from the pinned natural post-SPMD Grug HLO. Recovery uses
only shapes, contraction dimensions, scalar dataflow, liveness, and shared
parameter origins. Removing all HLO metadata produces the same plans.

This is a semantic-recovery checkpoint. It does not include a generated GPU
kernel, a replacement call, latency measurements, or a launch-count claim.

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

## Live work after current replacements

The accounting starts from the checked-in transformed HLO artifact and applies
the public compact normalized-exponential forward replacement, followed by its
reverse replacement. Root reachability then leaves:

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
There are ten algebraic regions (six forward/rematerialized and four reverse),
but 28 HLO dots do not imply 28 physical GPU launches. XLA may fuse surrounding
work or select library implementations, and a generated reverse may require
several kernels. Measure launch and latency deltas only after an exact generated
replacement executes.

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

Placement all-reduces feeding the incoming cotangents remain outside the
recovered reverse plans. The recovered plans record the nearest upstream
collectives so a future replacement cannot silently absorb or move them.
Input-adjoint outputs feed the surrounding normalization reverse. Weight
adjoints feed the ordinary FP32 optimizer update.

## Proposed physical ownership boundary

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

The first replacement should preserve JAX's existing saved/rematerialized
values and keep collectives outside. A later candidate can choose save versus
recompute and kernel boundaries. An exact HLO replacement is intentionally not
part of this checkpoint because its tuple outputs must cover the forward result,
the input adjoint, both weight adjoints, and any saved values required by the
selected policy.

## Reproduction

~~~
uv run --frozen --package marin-tile-lifetime --group test pytest -q \
  lib/tile_lifetime/tests/test_xla_low_rank_gated_product.py
~~~

The test applies the two normalized-exponential replacements, performs
root-reachable accounting, removes frontend metadata, checks every numerical
boundary, and exercises a hidden-Map mutation through the same scalar
generator.
