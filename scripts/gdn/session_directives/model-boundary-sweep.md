# Session Directive: Model-Boundary Sweep Is Now Reference Evidence, Not The Mainline

Completed evidence already shows:
- with CE fixed, throughput degrades roughly monotonically as `gdn_layers_per_block` rises,
- the fixed `3/4` GDN regime is non-negotiable,
- the sweep's main value is diagnostic: it proves that most of the added cost of GDN-bearing layers lands
  outside the currently tracked train-path budget.

How to use this evidence now:
- treat the existing sweep as proof that the current optimization boundary is wrong,
- do not spend another mainline iteration repeating the same `gdn_layers_per_block` sweep unless you are
  refreshing stale data or validating a major benchmark/config change,
- use the sweep as reference evidence when deciding whether a candidate attacks the right boundary.

Main interpretation:
- each added GDN-bearing layer brings a large decoder-layer shell/scaffolding tax beyond the tracked kernel budget,
- therefore the next mainline boundary is the whole GDN-bearing decoder layer, not the existing chunk-kernel train path.
