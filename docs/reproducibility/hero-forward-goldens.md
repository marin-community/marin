# Hero 535B forward goldens

This reference fixes native Levanter outputs for the permanent Hero checkpoint at step 108,000. It is an
input and observation contract for vLLM and Megatron forward checks. It is not evidence that either backend
is correct, and it does not contain expert weights.

The required bundle is stored at:

```text
s3://marin-us-east-02a/marin/reference/hero-forward/hero-535b-step108000-bf16-v1-dcfe4ced165a
```

The checked-in copy of its `manifest.json` records the object checksum, source revisions, runtime, job,
checkpoint layout, resolved training and inference model configurations, tokenizer revision, case definitions,
and native repeatability measurements. The permanent arrays stay in CoreWeave object storage.

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
  a small set of float32 pre-softmax vocabulary rows for diagnosis.

The backend observation archive must contain every array named by `REQUIRED_OBSERVATIONS`. Missing arrays and
shape or index shifts are errors, not partial passes.

## Route alignment

`route_expert_ids[layer, case, token_position, route_slot]` contains ordered global expert IDs for every valid
token and every MoE layer. `route_combine_weights` contains the BF16 values passed to expert computation,
widened to float32 for storage. `route_cutoff_gaps[layer, case, token_position]` is the biased Kth routing score
minus the biased (K+1)th score.

Padding uses expert ID `-1`, combine weight `0`, and gap `0`. A small cutoff gap is reported when a route differs;
it never excuses the mismatch. The expert parameters remain in the checkpoint.

## Reproduce a reviewed baseline

The producer restores authoritative FP32 master parameters, applies the checkpoint's pending query-bias values
once with the native restore rule, casts to BF16 compute parameters, and then freezes that effective bias. Its
only intentional model override is the existing `sonic_cute` dropless MoE implementation.

The 8,192- and 16,384-token diagnostics are deliberately skipped. The checkpoint-writing configuration fixes
`max_seq_len=4096`; either run would require changing model semantics rather than exercising this fixed model.
The manifest retains this result explicitly.

For a reviewed replacement, first update the pinned checkpoint or producer and bump the release constant. Then
commit the exact producer source and submit from a clean tree:

```bash
marin-env uv run --frozen python -m experiments.grug.moe_hero_ep.ops.forward_goldens \
  submit --mode required
```

The job uses interactive priority on `cw-us-east-08a`, eight four-GPU GB200 nodes, and the source revision named
in the request. It refuses to overwrite a complete or partial bundle path. Run `--mode smoke` first after a
producer change; that exercises the same full checkpoint and 32-GPU topology with 64-token rows.
