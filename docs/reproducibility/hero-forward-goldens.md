# Hero 535B forward goldens

This reference fixes native Levanter outputs for the permanent Hero checkpoint at step 108,000. It is an
input and observation contract for vLLM and Megatron forward checks. It is not evidence that either backend
is correct, and it does not contain expert weights.

The required bundle is stored at:

```text
s3://marin-us-east-02a/marin/reference/hero-forward/hero-535b-step108000-bf16-v1-dcfe4ced165a
```

The [checked-in manifest](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/testing/inference/resources/hero_535b_step108000_bf16_v1/manifest.json)
records the object checksum, source revisions, runtime, job, checkpoint layout, resolved training and inference
model configurations, tokenizer revision, case definitions, and native repeatability measurements. The permanent
arrays stay in CoreWeave object storage.

The [required Iris job](https://iris.oa.dev/#/job/%2Fhero-goldens%2Fhero-forward-required-23eb6bbdea13)
completed all eight tasks on 32 GB200s in `cw-us-east-08a`. Its stored bundle was then fetched back and checked
with the NumPy-only consumer. The earlier [short smoke job](https://iris.oa.dev/#/job/%2Fhero-goldens%2Fhero-forward-smoke-c10222d78be0)
used the same topology and producer revision.

## Fetch and load

Use CoreWeave credentials and copy the two bundle files into a fresh directory:

```bash
marin-env uv run --frozen fsutil cp -r \
  s3://marin-us-east-02a/marin/reference/hero-forward/hero-535b-step108000-bf16-v1-dcfe4ced165a \
  /tmp/hero-forward-goldens
```

`GoldenBundle.load` verifies the recorded byte length and SHA-256 digest before loading the pickle-free NumPy
archive. The loader and comparator import NumPy and the Python standard library, not JAX or Levanter.

```python
from marin.testing.inference.hero_forward_goldens import (
    ComparisonTolerances,
    GoldenBundle,
    compare_observations,
    load_observations,
)

golden = GoldenBundle.load("/tmp/hero-forward-goldens")
observed = load_observations("/tmp/my-backend-observations.npz")
report = compare_observations(
    golden,
    observed,
    ComparisonTolerances(
        target_logprob=0.05,
        top_logprob=0.05,
        full_logit=0.5,
        route_combine_weight=0.01,
        route_cutoff_gap=0.05,
    ),
)
report.raise_for_errors()
```

Those bounds are an explicit example, not qualified cross-backend tolerances. Each backend owner must record
the provisional bounds used during bring-up. Native repeatability measurements in the manifest do not establish
a valid cross-backend tolerance.

The producer enables `--xla_gpu_deterministic_ops=true`. The native dropless `sonic_cute` path combines expert
contributions with a GPU scatter-add; fixing that reduction order makes the repeat and traced-versus-ordinary
checks measure instrumentation effects instead of atomic scheduling. The exact flag is retained in the manifest.

## Input and score alignment

All indices are zero-based.

- `tokens[case, token_position]` contains the exact padded token rows passed to native forward.
- `segment_ids` is the structured causal-attention mask. Segment `0` is valid and `-1` is padding.
  `token_validity` records the same validity separately.
- `positions` contains absolute token positions. `valid_lengths` records each row's unpadded length.
- `score_mask` is separate from token validity. It selects fixed target tokens; it is not an attention mask or
  training loss mask.
- Score row `i` predicts `target_token_ids[i]` from hidden state
  `[score_case_indices[i], prediction_positions[i]]`. In other words, token position `p` is predicted at
  position `p - 1`.
- `target_logprobs` and `top_logprobs` are float32 full-vocabulary `log_softmax` values. `full_logits` contains
  a small set of float32 pre-softmax vocabulary rows for diagnosis. Each `full_logit_prediction_positions[i]`
  follows the same convention as a score position: that hidden state predicts the token at position `p + 1`.

The backend observation archive must contain every array named by `REQUIRED_OBSERVATIONS`. Missing arrays and
shape or index shifts are errors, not partial passes.

## Route alignment

`route_expert_ids[layer, case, token_position, route_slot]` contains ordered global expert IDs for every valid
token and every MoE layer. Route slots follow descending biased router score, exactly as emitted by JAX `top_k`;
another backend must reorder its routes to that convention before comparison. `route_combine_weights` contains
the BF16 values passed to expert computation, widened to float32 for storage.
`route_cutoff_gaps[layer, case, token_position]` is the biased Kth routing score minus the biased (K+1)th score.

Padding uses expert ID `-1`, combine weight `0`, and gap `0`. A small cutoff gap is reported when a route differs;
it never excuses the mismatch. The expert parameters remain in the checkpoint.

## Reproduce the 4K reference

Hero trained with `MasterParamMode.DEVICE`. This checkpoint therefore stores its authoritative FP32 master copy
directly under `params`; it has no separate `master_params` tree. The producer validates that layout and dtype,
applies the checkpoint's pending query-bias values once with the native restore rule, casts the result to BF16,
and then freezes that effective bias. Its only intentional model override is the existing `sonic_cute` dropless
MoE implementation.

For a reviewed replacement, first update the pinned checkpoint or producer and bump the release constant. Then
commit the exact producer source and submit from a clean tree:

```bash
marin-env uv run --frozen python -m experiments.grug.moe_hero_ep.ops.forward_goldens \
  submit --mode required
```

The job uses interactive priority on `cw-us-east-08a`, eight four-GPU GB200 nodes, and the source revision named
in the request. It refuses to overwrite a complete or partial bundle path. Run `--mode smoke` first after a
producer change; that exercises the same full checkpoint and 32-GPU topology with 64-token rows.

## Longer-context diagnostics

The original 4K manifest says that 8,192- and 16,384-token runs would require a model-semantic change. That
conclusion came from a producer guard against inputs longer than `max_seq_len`; it was not a runtime result. The
manifest remains unchanged as part of the published 4K record. This section supersedes only that skip rationale.

`max_seq_len=4096` fixes the training input length, optimizer accounting, and exported model metadata. Native
`Transformer.__call__` derives its sequence length from the input tensor. Fused RoPE positions, full-causal FA4
bounds, sliding-window bounds, and token validity are built at that runtime length. The longer runs removed the
producer guard and left the checkpoint model dictionary unchanged, including `max_seq_len=4096`, RoPE, attention,
and kernel choices.

| Input length | Iris job | Bundle | `arrays.npz` |
| --- | --- | --- | --- |
| 8,192 | [8K diagnostic](https://iris.oa.dev/#/job/%2Fhero-goldens%2Fhero-forward-diagnostic-8192-0b2ca7688e4b) | `s3://marin-us-east-02a/marin/reference/hero-forward/hero-535b-step108000-bf16-8k-diagnostic-v1-5a7dffedea5c` | 317,548,529 bytes; SHA-256 `906a2bfb14d40fa23596646807692727e59f9006afda885e95995464b93d4632` |
| 16,384 | [16K diagnostic](https://iris.oa.dev/#/job/%2Fhero-goldens%2Fhero-forward-diagnostic-16384-0fa004114827) | `s3://marin-us-east-02a/marin/reference/hero-forward/hero-535b-step108000-bf16-16k-diagnostic-v1-b4f39ffb123b` | 632,080,781 bytes; SHA-256 `3879ce6770f7016a50b35f38334e455e07b7d7a671ff05780c5fae9d9cc7ae42` |

Each job ran one fixed token sequence repeated across the 32 rows required by the eight-node, 32-GB200 topology.
The 8K and 16K archives contain route tensors shaped `[48, 32, 8192, 8]` and `[48, 32, 16384, 8]`, respectively.
For both lengths, the ordinary repeat and traced-versus-ordinary checks had zero maximum numerical change and zero
top-token changes. Both stored bundles were fetched, checksum-validated, loaded without importing JAX or Levanter,
and exactly self-compared through the NumPy-only consumer. Target alignment and global expert-ID bounds also passed.

The [first 16K attempt](https://iris.oa.dev/#/job/%2Fhero-goldens%2Fhero-forward-diagnostic-16384-c9633c7bbc64)
completed both ordinary forwards and the traced forward, then nonzero ranks entered JAX clean-exit shutdown while
process zero materialized the 1.7 GB uncompressed route trace. The coordination service aborted two minutes later.
Producer revision `231c6b08d94e75acfe3806ede58fa8817e75d9f3` keeps all ranks at a global barrier until
process zero finishes validation, compression, and upload. The linked 16K retry completed all eight tasks in
5 minutes 8 seconds.

The successful runs show that this native implementation can execute these fixed inputs. They do not establish a
supported Hero context length: the checkpoint was trained at 4,096 tokens, and these runs did not add context
extension or qualify model quality beyond that length.

To reproduce each published request, check out its recorded producer revision and use a fresh CoreWeave object
storage root because the producer refuses to overwrite an existing bundle:

```bash
# Published 8K request.
git checkout 79670824e907ebcc4292c6d034983c8a0036e51d
marin-env uv run --frozen python -m experiments.grug.moe_hero_ep.ops.forward_goldens \
  submit --mode diagnostic-8192 --store-root s3://marin-us-east-02a/tmp/<fresh-8k-root>

# Published 16K request, including the rank-lifecycle barrier.
git checkout 231c6b08d94e75acfe3806ede58fa8817e75d9f3
marin-env uv run --frozen python -m experiments.grug.moe_hero_ep.ops.forward_goldens \
  submit --mode diagnostic-16384 --store-root s3://marin-us-east-02a/tmp/<fresh-16k-root>
```
