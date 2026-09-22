# Multi-latent MoE (per-expert latent projections)

## Goal
Baseline LatentMoE uses ONE shared `w_latent_down (d→lat)` + shared `latent_norm` + shared
`w_latent_up (lat→d)`; all experts operate in the same latent subspace. This variant gives **each
expert its own** latent-down, latent-norm (learnable RMS), and latent-up, so each expert picks its
own subspace. The all-to-all now transports the **full d-dim** activation (not the compressed
latent). Gated behind `per_expert_latent` (default False → byte-identical shared path; hero-safe).

## Config
`GrugModelConfig.per_expert_latent: bool = False` (model.py). Requires `latent_dim is not None`.

## Approach: FSDP (not EP) — decided

Run with `expert_axis_size=1` (FSDP; params shard over data, all experts local, no all-to-all). This
avoids the shared pooled-wave EP kernel entirely. Per-expert latent is implemented **self-contained
in fast_track `model.py`** reusing public helpers (`_prepare_moe_dispatch`, `ragged_dot`,
`split_moe_w13_output`, `_zero_inactive_grouped_rows` from `levanter.grug._moe.common`) — **no shared
kernel edits**. Cost: re-run the baseline under FSDP for a fair comparison (accepted).

Per-expert forward (dispatched, grouped by expert via `group_sizes`):
`x_d → ragged_dot(pe_latent_down) → rms(lat) → ragged_dot(w13 * gain[:,:,None]) → act → ragged_dot(w2) → ragged_dot(pe_latent_up) → scatter-combine`.
Gain-fold: `(rms(x)*g)@W == rms(x)@(g*W)`, so the learnable per-expert RMS gain multiplies `w13`
(no per-token expert gather). No capacity limit under FSDP (all tokens local) → no drops.

## Edit points (FSDP self-contained)
- **model.py config**: `per_expert_latent: bool = False`.
- **model.py MoEMLP**: new per-expert fields `pe_latent_down [E,d,lat]`, `pe_latent_gain [E,lat]`,
  `pe_latent_up [E,lat,d]` (None unless flag); `expert_mlp` built on `latent` dim; `__call__` runs the
  self-contained per-expert forward above (else the existing shared-latent path).
- **optimizer.py**: `pe_latent_gain` → adam (name guard); `pe_latent_down/up` (3-D) → muonh.
- **launch.py**: `--fsdp-moe` (expert_axis=1) and `--per-expert-latent` (implies fsdp).

## Original EP edit points (deferred; kept for reference)
1. **model.py `MoEMLP.init`** — when `per_expert_latent`: set shared `w_latent_down/latent_norm/
   w_latent_up = None`; call `MoEExpertMlp.init` with `hidden_dim=d` (full), `latent_dim=cfg.latent_dim`,
   `per_expert_latent=True`.
2. **model.py `MoEMLP.__call__`** — when `per_expert_latent`: `routed_input = x_flat` (full d, skip
   shared down/norm).
3. **grug_moe.py `MoEExpertMlp`** — new optional fields `w_latent_down [E,d,lat]`,
   `latent_norm_gain [E,lat]`, `w_latent_up [E,lat,d]` (None unless per-expert). `init` builds them
   (down/up ~ matrix init; norm gain = ones). `w_gate/w_up` become `[E,lat,I2]`, `w_down [E,I,lat]`
   (fan-in=lat). `__call__` passes the new weights to `moe_mlp`.
4. **grug_moe.py `moe_mlp`** — thread the 3 optional weights to the shard_map (expert-sharded specs
   `P("expert",None,None)` / gain `P("expert",None)`) and into the local fn.
5. **_moe/ep_fixed_pooled_wave_all_to_all.py `_compute_pooled` + local fn** — additive in the
   `moe_up_down` scope: `x=einsum('erd,edl->erl',compacted_x,latent_down)`, per-expert RMS-norm
   `* latent_norm_gain[:,None,:]`, then existing gate/up (`erl,eli->eri`) / down (`eri,eil->erl`),
   then `einsum('erl,eld->erd', out, latent_up)`. Transport dim stays d in/out; dispatch/capacity/
   combine unchanged.
6. **optimizer.py `create_mask`** — per-expert latent matrices (3-D) → muonh (matrix catch-all, same
   as w_gate/up/down: correct). Per-expert **norm gain** (`latent_norm_gain`, ends up 3-D under scan
   `[layers,E,lat]`) MUST route to **adam** — add an explicit name match on `latent_norm_gain`
   (endswith) BEFORE the ndim→muonh fallback.
7. **launch.py** — `--per-expert-latent` flag → `dataclasses.replace(model, per_expert_latent=True)`.

## Correctness / tests
- CPU: build a tiny MoE with `per_expert_latent=True`, one forward, assert finite; compare shape to
  shared path.
- Optimizer: assert `latent_norm_gain` → adam, `w_latent_down/up` → muonh via create_mask.
- Default (flag off) unchanged: shared path byte-identical.

## Cost note
All-to-all carries d instead of lat → ~d/lat× (≈2×) more dispatch bytes. Inherent to per-expert latents.
