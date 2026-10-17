# Copy-paste head on July MoE

This experiment exposes the target-only loss from [Copy-Paste Language
Models](https://arxiv.org/abs/2610.19247v1). The vocabulary-normalized copy gate
mixes the native head with attention over observed tokens in the same packed
document. Repeated tokens accumulate probability; a learned sink sends its
mass back to the vocabulary head. Queries include the current observed token.

`copy_paste_nll` accepts native vocabulary NLL and log partition and returns
unreduced NLL. `fused_copy_paste_nll` computes those inputs using the TPU fused
CE kernel. Pass batch-sharded arrays with a complete context and vocabulary
inside each shard. Context/vocabulary parallelism is not implemented. The query
tile must divide sequence length. Apply native loss weights, reductions and
router auxiliary loss after this function; final padding and document-boundary
labels must use the native training mask. The functions do not implement
generation, cache decoding, z-loss or a trainer/launcher.

The head uses two hidden-by-hidden matrices plus two hidden vectors, adding
`2 * hidden_dim**2 + 2 * hidden_dim` parameters. Initialize these separately
from the backbone PRNG stream. The July experiment uses independent normal
weights with standard deviation 0.02, BF16 compute, FP32 parameter storage,
query tiles of 128 and native optimizer grouping (matrix MuonH; vectors Adam).
The paper does not prescribe this initialization or the July optimizer.

Rematerialized query tiles avoid storing a full sequence-by-sequence score
matrix during autodiff. The copy operation remains quadratic in arithmetic;
the implementation makes no speedup claim. No full copy-vocabulary tensor is
constructed. The log partition stays attached to autodiff.

The [research issue](https://github.com/marin-community/marin/issues/10229)
tracks d512 and d768 comparisons. Historical runs use July recipe commit
`444b5ea66bda75b6dfc7746ff20c9dfec6ba159a` and runtime
`3fd402a926c907d7cb87a04935e56db972652c48`; they do not establish end-to-end
compatibility with the current Marin trainer. The pinned training adapter,
immutable recipes, loss curves and run identities are linked from that issue.
This initial contribution isolates the loss implementation and its independent
dense-mixture numerical tests. It does not change the default MoE model.

Run non-training numerical tests with:

```bash
uv run pytest experiments/grug/moe_cplm/test_head.py
```

The CPU tests cover FP32 values/gradients, repeated and absent targets, document
and causal isolation, disabled copy and zero loss weights. TPU fused CE and
BF16 parity require accelerator checks separately; the historical d512 run's
checks are recorded in the research issue. No optimizer updates occur in these
tests.
