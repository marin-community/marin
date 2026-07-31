# FSDP MoE Hero — model-size sweep

Sweep over model width, fixing the attention schedule per size: the local/global KV-head split and
exactly which (1-indexed) layers run full "global" attention (all others use the sliding window). The
global-layer sets are explicit because they don't all reduce to a single `global_every` stride — e.g.
`d1280` is global on 6, 12, and its final layer 14.

## Architecture

| config | hidden | layers | query heads | KV heads (local / global) | global layers |
|--------|-------:|-------:|------------:|:-------------------------:|:--------------|
| d512   |   512  |    6   |      4      |          1 / 1            | 6             |
| d768   |   768  |    8   |      6      |          2 / 1            | 4, 8          |
| d1024  |  1024  |   12   |      8      |          2 / 1            | 6, 12         |
| d1280  |  1280  |   14   |     10      |          2 / 2            | 6, 12, 14     |
| d1536  |  1536  |   16   |     12      |          3 / 2            | 6, 12, 16     |
| d2048  |  2048  |   18   |     16      |          4 / 2            | 6, 12, 18     |

Query heads = `hidden / 128` (head_dim = 128). KV heads are the GQA groups under them; the projection
is stored per layer at its own head count (local layers use the local count, global layers the global).

## Scale & token budget

Chinchilla-style budget: **total training tokens = 60 × active params**, at **sequence length 8192**.
Steps = tokens / (batch × 8192).

| config | total params | active params (excl. embed + lm_head) | batch | tokens | steps |
|--------|-------------:|--------------------------------------:|------:|-------:|------:|
| d512   |   0.44 B     |                18.5 M                 |   32  | 1.11 B |  4230 |
| d768   |   1.13 B     |                55.4 M                 |   64  | 3.33 B |  6345 |
| d1024  |   2.75 B     |               145.8 M                 |  128  | 8.75 B |  8340 |
| d1280  |   4.86 B     |               263.8 M                 |  128  | 15.83 B| 15094 |
| d1536  |   7.85 B     |               436.1 M                 |  256  | 26.16 B| 12476 |
| d2048  |  15.44 B     |               869.8 M                 |  512  | 52.19 B| 12442 |

- **Total params** counts everything: token embedding, attention, router, all 128 routed experts,
  the 2 shared experts, and the (untied) lm_head. The embedding + lm_head dominate the small configs
  (2 × 128256 × hidden), which is why the sweep budgets on active params.
- **Active params** is the per-token cost excluding embedding and lm_head: attention (q/k/v/o with the
  per-layer KV split), router, top-4 routed experts, and the 2 always-on shared experts.

## Assumptions (not given by the sweep spec — adjust as needed)

The spec fixes width, depth, and the attention schedule. Everything else is inherited from the hero
MoE structure and held constant across the sweep:

- MoE: **128 experts, top-4, 2 shared experts**; routed and shared expert intermediate width = `hidden / 2`.
- head_dim = 128; vocab = 128256; untied embeddings (embed and lm_head counted separately).
- Param counts omit RMSNorm scales, GatedNorm, and SConv weights (< ~1% combined).
