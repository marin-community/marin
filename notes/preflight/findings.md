# GrugMoE inference preflight findings

Status: **no-go for the architecture matrix; the four-node acceptance was not
run**.

## Recommendation

Do not start the proposed architecture performance matrix from the frozen
serving stack.

The exact July 27 reference cannot start at vLLM
`afb26719464d5957e695bde478ae93a160b11d14`. It needs heterogeneous local and
global KV heads, global attention every six layers, and `sconv4`. The frozen
model has one KV-head count, hardcodes global attention every four layers, and
has no sconv path.

The closest loadable approximation did prove useful facts:

- one GB200 node can serve GrugMoE at PP1/TP1/DP4/EP4;
- two colocated GB200 nodes form one PP1/TP1/DP8/EP8 world;
- cold and reused requests agree on token IDs, logprobs, and routed experts;
- prefix hits react correctly at 17 and 513 tokens;
- the two-node correctness result is byte-identical across two launches;
- every recorded live bundle was read back byte-identically from S3.

Those facts do not make the approximation performance-faithful. Its semantic
KV estimate is 4.605 GiB per 65,536-token sequence, versus 1.617 GiB for the
exact model. Worse, the live cache pool exposes about 296,653 bytes per token,
which is within 0.6% of allocating full-length KV for all 48 layers. That is
about 18 GiB per 65,536 tokens and 293% above the approximation's sliding-window
semantic estimate.

The conditional four-node job had to stop before submission. The smaller
frozen-tensor cross-framework gate is still uncertain. More decisively, the
checked-in production Iris serving path submits one task, while the replicated
dev-GPU path submits holder processes and depends on a workstation to reach the
pods with `kubectl`. Turning that into the required unattended, replicated
entrypoint is broad launcher work, which is a stated stop condition.

## Frozen inputs

| Input | Frozen value |
|---|---|
| Marin base | `75bf2437035cf731d1a4bd71266229dfcdda9478` |
| First harness evidence commit | `874015da11814ac162400baec0e04bee0fb4abd9` |
| Gloo address fix / passing EP8 commit | `d043e51266650ee3db2ff041e1c2095fe443f55f` |
| vLLM | `afb26719464d5957e695bde478ae93a160b11d14` |
| Training reference | `fd3e9bc5b428633027f944be7fdf1136567db028` |
| Snowball export | `s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/` |
| Image | `ghcr.io/marin-community/iris-task:41a1ac729` |
| Image digest | `sha256:d90bc25fc778b9d4f5b9395cba4ac2457a12e106c4c2bcb4c0b9c7d70dd57dca` |
| Two-node workload SHA-256 | `fa52d5a3dd5ad0bed15941fee85be8d120efaedbaaec877da5e72ce86528c14e` |
| Dependency lock SHA-256 | `c64a93c8ea08b7a441e831804b5058ce6fc2ee728d01333e2377e95c72ac7082` |

The live config used BF16 weights and KV cache, seed 1234, prefix caching,
chunked prefill, PP1, TP1, and DP=EP. Smoke runs capped the request context at
2,048 tokens while preserving the reference-sized model configuration.

## Assumption status

| Assumption | Status | Evidence and consequence |
|---|---|---|
| Exact frozen reference starts | **rejected** | Every-six attention, heterogeneous 12/6 KV, and sconv-on need custom model/cache work. |
| Ordinary serving omits dense MTP | **confirmed** | No dense MTP head is invoked by the frozen next-token serving path. |
| Every selected top-K contribution is dispatched | **confirmed** | The Grug call and pinned `FusedMoE` path have no capacity clipping or token drop. |
| Same frozen tiny tensors agree across Levanter and vLLM | **still uncertain** | Router formulas match statically, but no run loaded the same checkpoint into both and compared selected experts, gate weights, and next-token logprobs. |
| Prefix reuse preserves results | **confirmed** | Cold and reused token IDs, logprobs, and route-array hashes match at 17 and 513 tokens. |
| Prefix mutation causes a miss | **confirmed** | Mutated requests report zero reused tokens and zero new prefix hits at both boundaries. |
| Seeded dummy routing is reproducible | **confirmed** | Two EP8 launches produced byte-identical compact correctness evidence. |
| Seeded dummy routing is balanced | **rejected** | Only experts 5, 15, 29, 53, and 72 received assignments in EP8 smoke; most experts were unused. The checked-in deterministic control gives equal work to every expert and EP rank, but is only an instrumentation control. |
| RL append request shape works | **confirmed for dummy; still uncertain for Snowball** | Dummy appends produce real hits, sampled-token logprobs, routed IDs, and fixed four-token responses. The single Snowball attempt failed before load on path-style S3 listing. |
| Sliding-window semantic KV predicts allocation | **rejected** | Live bytes per token are 293% above the loadable semantic estimate and track full-length allocation for all layers. |
| EP8 fabric is usable | **confirmed** | Eight ranks joined one NCCL world across two tray pods, served requests twice, and showed no communication error or hang after the Gloo address fix. |
| Identical throughput arms differ by at most 2% | **still uncertain** | The repeated smoke proves output repeatability, not ten-minute/250,000-token throughput repeatability. |
| Unattended EP16 acceptance is launchable | **rejected at this commit** | No replicated unattended serving entrypoint exists. The dev workflow launches holder pods only. |
| S3 artifact round-trip works | **confirmed** | Every success and failure bundle recorded below passed byte-identical readback. |

## P0 capability audit

The required readiness vocabulary is literal: `ready`, `blocked`, or
`confounded`.

| Capability | Implementation | Readiness | Evidence / smallest no-op control | Exact next step |
|---|---|---|---|---|
| KV 12 local / 12 global | `config-only` | `ready` | One `num_key_value_heads=12` applies to every layer; the EP8 approximation loaded. Control: keep 12/12. | Use only for serving-stack diagnosis, not architecture ranking. |
| KV 12 local / 6 global | `custom-code` | `blocked` | Config, model, and cache expose one KV-head count. Control: 12/12. | Add heterogeneous attention specs and cache sizing in a separate vLLM change. |
| KV 12 local / 2 global | `custom-code` | `blocked` | Same single-count restriction. Control: 12/12. | Validate 12/6 first, then reuse that implementation. |
| Global attention every 4 | `config-only` | `ready` | Frozen config generates every-four and the live approximation starts. Control: current schedule. | Keep as a launcher control only. |
| Global attention every 6 | `custom-code` | `blocked` | `_FULL_ATTENTION_INTERVAL = 4` is fixed. Control: every-four. | Make interval explicit and update hybrid-cache grouping separately. |
| Sliding window 512 | `config-only` | `ready` | Live 513-token append crosses the boundary and preserves cold/reuse equality. Control: 512. | Keep the boundary request. |
| Sliding window 2,048 | `config-only` | `ready` | Same HF config field; no serving source change. Control: 512. | Smoke only after exact architecture support exists. |
| Top-4 / 128 / i3072 | `config-only` | `ready` | Reference-sized dummy weights loaded at EP8. Control: same geometry. | Keep as the reference expert geometry. |
| Top-8 / 256 / i1536 | `config-only` | `confounded` | Config and contiguous EP16 rank starts exist, but no compliant unattended EP16 launch exists. Control: top-4/128/i3072. | Add the unattended entrypoint before one EP16 acceptance attempt. |
| `sconv4` off | `config-only` | `ready` | Frozen vLLM has no sconv state or call. Control: off. | Use off only as the loadable launcher reference. |
| `sconv4` on | `custom-code` | `blocked` | No model path, loader mapping, or serving kernel exists. Control: off. | Implement and validate in a separate model/kernel change. |
| EP8 | `config-only` | `ready` | Two live launches formed ranks 0–7 after advertising routable pod IPs to Gloo. Control: EP4. | Preserve the Gloo interface setting. |
| EP16 | `config-only` | `confounded` | vLLM accepts DP16/EP16, but Marin has no unattended replicated entrypoint. Control: EP8. | Land that entrypoint, then run the single frozen acceptance command shown below. |

## Prefix, routing, and request-path evidence

Both passing EP8 runs report:

```text
17-token boundary: 16 reused tokens; mutation 0
513-token boundary: 512 reused tokens; mutation 0
prefix cache hits: 528
prefix cache queries: 1,614
prompt tokens cached: 528
generated tokens: 24
preemptions: 0
```

The result files index full response JSON, NumPy routed-expert arrays, and raw
Prometheus snapshots as separate objects. The compact comparison SHA-256 for
both EP8 runs is:

```text
f177598299cd6cad7501d1a513c4c52d7097ee958d2ee326b9a537a9bd91b373
```

Dummy routing is deterministic but intentionally not treated as representative
load balance. Mapping the 104,448 live assignments by contiguous groups of 16
experts gives this EP-rank histogram:

```text
[52,224, 96, 0, 26,112, 26,016, 0, 0, 0]
```

The checked-in balanced fixture cycles top-K assignments across all experts.
For 128 experts, top-4, and EP8, it produces four assignments per expert and 64
per EP rank. Its assignment SHA-256 is
`f1d3dd8e5223591ade577c7c438ed3ac80e116fde03d17d0e7e662efa8a71210`.

## KV evidence

At 65,536 tokens:

| Model/allocation interpretation | GiB per sequence | Bytes per token |
|---|---:|---:|
| Exact every-six, 12 local / 6 global semantics | 1.6172 | 26,496 |
| Every-six, uniform-12 semantics | 3.1172 | 51,072 |
| Loadable every-four, uniform-12 semantics | 4.6055 | 75,456 |
| Full-length allocation for all 48 layers | 18.0000 | 294,912 |
| Live pool: 68.73 GiB / 248,770 tokens | about 18.11 | 296,653 |

The observed pool is only 0.6% above the full-allocation calculation. This is
strong evidence that semantic sliding-window savings are not reflected in the
allocated KV bytes for this prefix-cached configuration. It is not an observed
single 65K request: the smoke request cap was 2,048. The live run also recorded
94.98 GiB of model weights per rank, 68.73 GiB available KV memory, 248,770 KV
tokens, 121.47× advertised concurrency at 2,048 tokens, and zero preemptions.

## Live run index

All runs used cluster `cw-us-east-08a`, Iris `interactive` priority, the image
digest above, and the checked-in one-command driver after holder allocation.

| Run | Iris holder | Result | Evidence |
|---|---|---|---|
| `20260731T0228Z-one-node-ep4-retry4` | `/romain/dev-gpu-grugmoe-preflight-1n-20260731t0210z` | EP4 smoke passed; holder killed after use | `s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/one-node-ep4/20260731T0228Z-one-node-ep4-retry4/` |
| `20260731T0234Z-reference-ep8-arm1` | `/romain/dev-gpu-grugmoe-preflight-2n-20260731t0233z` | Failed: Gloo advertised loopback; failure bundle preserved | `s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/reference-ep8/20260731T0234Z-reference-ep8-arm1/` |
| `20260731T0241Z-reference-ep8-retry1` | same two-node holder | EP8 smoke passed | `s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/reference-ep8/20260731T0241Z-reference-ep8-retry1/` |
| `20260731T0248Z-reference-ep8-repeat2` | same two-node holder | Independent EP8 smoke passed; compact result identical | `s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/reference-ep8/20260731T0248Z-reference-ep8-repeat2/` |
| `20260731T0253Z-snowball-request-path` | same two-node holder | Still uncertain: path-style `ListObjectsV2` rejected before model load; holder killed after use | `s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/reference-ep8/20260731T0253Z-snowball-request-path/` |
| Four-node EP16 acceptance | none | Not submitted: smaller parity gate open and unattended launcher absent | No bundle |

The Snowball failure exposed two local defects. The branch now rejects zombie
launcher parents and writes a virtual-hosted S3 config for future runs. Per the
goal, the Snowball driver was run once only, so that fix is not claimed as live
Snowball evidence.

## Exact next actions

1. Close frozen-tensor parity without changing model behavior:

   ```sh
   uv run python tests/cluster/vllm/grug_training_oracle.py \
     --output /tmp/grugmoe-training-oracle
   ```

   Add a focused vLLM consumer that loads those exact tensors and token IDs and
   compares selected experts, gate weights, and next-token logprobs. The current
   API returns selected IDs but not gate weights, so this may itself end in a
   small-instrumentation blocker.

2. Add one replicated, coscheduled Iris entrypoint that runs the checked-in
   preflight worker on every task without workstation `kubectl`. Do not add a
   second serving backend or retry layer.

3. Only after both items pass, expose the intended command below and run it
   once:

   ```sh
   PYTHONPATH=lib/iris/src:lib/marin/src \
     uv run scripts/iris/grugmoe_inference_preflight.py submit \
     --config lib/iris/config/cw-us-east-08a.yaml \
     --case granular-ep16 \
     --mode acceptance \
     --run-id <UTC-run-id>
   ```

   `submit` does not exist at this commit. Its absence is the launcher blocker,
   not an invitation to run four interactive holder pods manually.

## Ranked remaining risks

1. **Architecture validity:** every-six attention, heterogeneous KV, and
   sconv-on are blocked. Exact support is required before throughput can rank
   candidates.
2. **KV capacity:** live allocation is about four times the loadable semantic
   estimate and about eleven times the exact target estimate at 65K.
3. **Cross-framework correctness:** same-tensor selected-expert, gate-weight,
   and next-token parity is still uncertain.
4. **Final orchestration:** no unattended replicated serving entrypoint exists.
5. **Repeatability:** correctness is repeatable; the ≤2% throughput gate is
   unmeasured.
6. **Snowball integration:** virtual-hosted S3 configuration is patched but
   unvalidated because the one permitted request-path attempt already ran.

## Local verification

```text
24 focused tests passed in 4.37s
```

The branch contains the compact 18-root/144-branch workload, cold/reuse/mutation
assertions, balanced routing control, two-arm acceptance contract, complete log
collection, exact manifest hashes, S3 upload/readback, and focused tests. No PR,
Gist, issue edit, architecture sweep, or four-node allocation was created.
