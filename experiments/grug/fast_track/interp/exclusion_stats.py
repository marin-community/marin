# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron exclusion study, statistics pass: are the ReLU^2 units inside one routed expert mutually exclusive?

Over unseen training sequences it accumulates, per layer and routed expert (on the tokens routed to it) and for the
shared expert (on every token): the routed-token count, the co-firing counts ``B^T B`` of the binary activity
``B = pre > 0`` (the diagonal is each unit's firing count), each unit's activation sum and square sum, the histogram
of the number of active units per token, and the first and second moments of the expert's input (the normed routed
latent; the shared expert's ``mlp_in``). The routed assignments are scattered into a per-expert padded
``[E, C, I]`` tensor (``C`` = ``CAPACITY`` rows per batch); overflow rows are dropped from every statistic (``cnt``
counts the kept rows, ``routed`` all routed tokens, ``dropped`` the difference).

A random ``SAMPLE`` rows per expert (bit-packed activity plus each row's flat token index) are kept for the local
analysis, with the tokens and segment ids of the whole pass.

At the end, per layer and floor ``f`` on the firing rates, a pair census of ``lift = P(i,j) / (P(i) P(j))`` over the
unit pairs with ``P(i), P(j) >= f``, against the within-expert token-shuffle null (each unit's column permuted
across the expert's tokens; the co-firing count is then hypergeometric with mean ``n_i n_j / n``), and its relation
to the weights: the cosine of the gain-scaled up-projection columns and the data correlation of the
pre-activations (``w_i^T Cov w_j`` normalized).

Writes ``excl_stats.npz`` (summaries and census), ``excl_cofire_L{l}.npz`` (full matrices, for the skip pass),
``excl_sample.npz`` (bit sample) and ``excl_tokens.npz``.
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

BATCH = 4
L, E, I = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH
R, D = forward.LATENT, forward.HIDDEN
K = forward.TOP_K
CAPACITY = 6144  # padded rows per expert per batch (mean routed load per batch is BATCH * 4096 * K / E = 427)
SAMPLE = 2048  # bit-sample rows kept per expert
SAMPLE_PER_BATCH = 16  # random rows per expert taken from each batch until the sample is full
STATS_SEED = 21
FLOORS = (0.02, 0.05, 0.1)
LIFT_THRESHOLDS = (0.5, 0.2, 0.1, 0.05, 0.02)  # "exclusive" at lift below each (the last ~ never co-fire)
LOG_LIFT_BINS = np.linspace(-3.0, 1.0, 81)  # log10 lift; lift 0 is clipped into the first bin
COS_BINS = np.linspace(-1.0, 1.0, 81)
FINE_BINS = np.linspace(-1.0, 1.0, 401)  # cosine / correlation histograms for exclusive vs other pairs (AUC)
NUM_TOP_PAIRS = 256  # most exclusive pairs kept per layer


def padded_rows(expert_sorted: jax.Array, starts: jax.Array, valid: jax.Array, capacity: int):
    """Row of each expert-sorted assignment in a ``[E * capacity]`` padded layout (``E * capacity`` = dropped: past
    the capacity or an invalid token) and the number of valid assignments dropped."""
    pos = jnp.arange(expert_sorted.shape[0]) - starts[expert_sorted]
    keep = valid & (pos < capacity)
    rows = jnp.where(keep, expert_sorted * capacity + pos, E * capacity)
    return rows, jnp.sum(valid & ~keep)


def scatter_padded(x: jax.Array, rows: jax.Array, capacity: int) -> jax.Array:
    """``[N, F]`` -> ``[E, capacity, F]`` zero padded."""
    out = jnp.zeros((E * capacity, x.shape[-1]), x.dtype).at[rows].set(x, mode="drop")
    return out.reshape(E, capacity, x.shape[-1])


def sample_rows(expert_sorted: jax.Array, valid: jax.Array, key: jax.Array, per: int) -> jax.Array:
    """Indices ``[E, per]`` into the expert-sorted assignments of ``per`` random valid rows per expert (-1 where the
    expert has fewer)."""
    n = expert_sorted.shape[0]
    u = jax.random.uniform(key, (n,))
    # Invalid rows sort after every valid row of their expert.
    order = jnp.argsort(expert_sorted.astype(jnp.float32) + jnp.where(valid, 0.5 * u, 0.5 + 0.5 * u))
    counts = jnp.zeros((E,), jnp.int32).at[expert_sorted].add(valid.astype(jnp.int32))
    starts = jnp.zeros((E,), jnp.int32).at[expert_sorted].add(1)
    starts = jnp.cumsum(starts) - starts
    idx = starts[:, None] + jnp.arange(per)[None, :]
    ok = jnp.arange(per)[None, :] < counts[:, None]
    return jnp.where(ok, order[jnp.clip(idx, 0, n - 1)], -1)


def empty_state() -> dict:
    return dict(
        cnt=jnp.zeros((L, E)),
        routed=jnp.zeros((L, E)),
        cofire=jnp.zeros((L, E, I, I)),
        asum=jnp.zeros((L, E, I)),
        a2sum=jnp.zeros((L, E, I)),
        hist=jnp.zeros((L, E, I + 1)),
        lat_sum=jnp.zeros((L, E, R)),
        lat_m2=jnp.zeros((L, E, R, R)),
        dropped=jnp.zeros((L,)),
        s_cofire=jnp.zeros((L, I, I)),
        s_asum=jnp.zeros((L, I)),
        s_a2sum=jnp.zeros((L, I)),
        s_hist=jnp.zeros((L, I + 1)),
        s_in_sum=jnp.zeros((L, D)),
        s_in_m2=jnp.zeros((L, D, D)),
        tokens=jnp.zeros(()),
    )


def make_step():
    @jax.jit
    def step(state, model, tokens, seg, weight, key):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        valid_tok = reshard(weight.reshape(-1), P(None)) > 0
        new = dict(state)
        samples = []
        for li, lay in enumerate(tr["layers"]):
            experts, toks = lay["expert_sorted"], lay["token_sorted"]
            pre, act = lay["expert_pre"], lay["expert_act"]
            valid = valid_tok[toks]
            rows, dropped = padded_rows(experts, lay["expert_starts"], valid, CAPACITY)
            bits = (pre > 0).astype(jnp.bfloat16)
            pb = scatter_padded(bits, rows, CAPACITY)
            cof = jnp.einsum("eci,ecj->eij", pb, pb, preferred_element_type=jnp.float32)
            px = scatter_padded(lay["routed_input"][toks].astype(jnp.float32), rows, CAPACITY)
            vf = (rows < E * CAPACITY).astype(jnp.float32)[:, None]

            def by_expert(x, experts=experts):
                return jax.ops.segment_sum(x, experts, num_segments=E, indices_are_sorted=True)

            nact = jnp.sum(pre > 0, axis=-1)
            hist = jnp.zeros((E, I + 1)).at[experts, nact].add(vf[:, 0])
            kept = (rows < E * CAPACITY).astype(jnp.float32)
            new["cnt"] = new["cnt"].at[li].add(by_expert(kept))
            new["routed"] = new["routed"].at[li].add(by_expert(valid.astype(jnp.float32)))
            new["cofire"] = new["cofire"].at[li].add(cof)
            new["asum"] = new["asum"].at[li].add(by_expert(act * vf))
            new["a2sum"] = new["a2sum"].at[li].add(by_expert(act * act * vf))
            new["hist"] = new["hist"].at[li].add(hist)
            new["lat_sum"] = new["lat_sum"].at[li].add(jnp.sum(px, axis=1))
            new["lat_m2"] = new["lat_m2"].at[li].add(jnp.einsum("ecr,ecs->ers", px, px))
            new["dropped"] = new["dropped"].at[li].add(dropped)

            s_pre, s_act = lay["shared_pre"], lay["shared_act"]
            svf = valid_tok.astype(jnp.float32)[:, None]
            s_bits = (s_pre > 0).astype(jnp.bfloat16) * svf.astype(jnp.bfloat16)
            new["s_cofire"] = (
                new["s_cofire"].at[li].add(jnp.einsum("ti,tj->ij", s_bits, s_bits, preferred_element_type=jnp.float32))
            )
            new["s_asum"] = new["s_asum"].at[li].add(jnp.sum(s_act * svf, axis=0))
            new["s_a2sum"] = new["s_a2sum"].at[li].add(jnp.sum(s_act * s_act * svf, axis=0))
            s_n = jnp.sum(s_pre > 0, axis=-1)
            new["s_hist"] = new["s_hist"].at[li].add(jnp.zeros((I + 1,)).at[s_n].add(valid_tok.astype(jnp.float32)))
            x_in = reshard(lay["mlp_in"].reshape(-1, D), P(None, None)).astype(jnp.float32) * svf
            new["s_in_sum"] = new["s_in_sum"].at[li].add(jnp.sum(x_in, axis=0))
            new["s_in_m2"] = new["s_in_m2"].at[li].add(jnp.einsum("td,tf->df", x_in, x_in))

            pick = sample_rows(experts, valid, jax.random.fold_in(key, li), SAMPLE_PER_BATCH)  # [E, per]
            safe = jnp.maximum(pick, 0)
            samples.append(
                dict(bits=(pre[safe] > 0) & (pick >= 0)[..., None], token=jnp.where(pick >= 0, toks[safe], -1))
            )
        new["tokens"] = state["tokens"] + jnp.sum(valid_tok)
        return new, samples

    return step


def effective_up(model) -> jax.Array:
    """``[L, E, R, I]`` the up-projection columns as the normed latent sees them (latent gain applied)."""
    m = forward.replicate(model)
    out = []
    for lay in m.layers():
        gain = lay.mlp.latent_norm(jnp.ones((R,), jnp.float32)).astype(jnp.float32)
        out.append(gain[None, :, None] * lay.mlp.expert_mlp.w_up.astype(jnp.float32))
    return jnp.stack(out)


@jax.jit
def layer_census(cofire, cnt, w_eff, lat_sum, lat_m2):
    """Pair census of one layer: ``[E, I, I]`` co-firing counts, ``[E]`` token counts, ``[E, R, I]`` effective
    up-projection, the input moments. Per floor: eligible and exclusive pair counts per expert (i < j), histograms of
    log10 lift, of the shuffle-null z-score, of (cosine, log lift) and (pre-activation correlation, log lift), the
    cosine and correlation histograms of exclusive (lift < 0.1) vs other pairs, and the most exclusive pairs."""
    n = jnp.maximum(cnt, 1.0)[:, None, None]
    diag = jnp.diagonal(cofire, axis1=1, axis2=2)  # [E, I]
    p = diag / n[:, :, 0]
    pij = cofire / n
    expect = p[:, :, None] * p[:, None, :]
    lift = pij / jnp.maximum(expect, 1e-12)
    ni, nj = diag[:, :, None], diag[:, None, :]
    mu = ni * nj / n
    var = ni * nj * (n - ni) * (n - nj) / (n * n * jnp.maximum(n - 1, 1.0))
    z = (cofire - mu) / jnp.sqrt(jnp.maximum(var, 1e-6))
    wn = w_eff / jnp.linalg.norm(w_eff, axis=1, keepdims=True)
    cos = jnp.einsum("eri,erj->eij", wn, wn)
    mean = lat_sum / n[:, :, 0]
    cov = lat_m2 / n - mean[:, :, None] * mean[:, None, :]
    wcw = jnp.einsum("eri,ers,esj->eij", w_eff, cov, w_eff)
    sd = jnp.sqrt(jnp.maximum(jnp.diagonal(wcw, axis1=1, axis2=2), 1e-12))
    corr = wcw / (sd[:, :, None] * sd[:, None, :])
    upper = jnp.triu(jnp.ones((I, I), bool), k=1)[None]
    log_lift = jnp.clip(jnp.log10(jnp.maximum(lift, 1e-3)), LOG_LIFT_BINS[0], LOG_LIFT_BINS[-1] - 1e-6)
    out = {}
    for f in FLOORS:
        tag = f"f{round(f * 100):02d}"
        elig = upper & (p[:, :, None] >= f) & (p[:, None, :] >= f)
        ef = elig.astype(jnp.float32)
        out[f"{tag}_eligible"] = jnp.sum(ef, axis=(1, 2))
        out[f"{tag}_exclusive"] = jnp.stack(
            [jnp.sum(ef * (lift < t), axis=(1, 2)) for t in LIFT_THRESHOLDS], axis=-1
        )  # [E, T]
        out[f"{tag}_exclusive_sig"] = jnp.stack(
            [jnp.sum(ef * (lift < t) * (z < -5.0), axis=(1, 2)) for t in LIFT_THRESHOLDS], axis=-1
        )
        out[f"{tag}_hist_lift"] = jnp.histogram(log_lift, LOG_LIFT_BINS, weights=ef)[0]
        out[f"{tag}_hist_z"] = jnp.histogram(jnp.clip(z, -50, 50), np.linspace(-50, 50, 201), weights=ef)[0]
        out[f"{tag}_hist_cos_lift"] = jnp.histogram2d(
            cos.reshape(-1), log_lift.reshape(-1), (COS_BINS, LOG_LIFT_BINS), weights=ef.reshape(-1)
        )[0]
        out[f"{tag}_hist_corr_lift"] = jnp.histogram2d(
            jnp.clip(corr, -1, 1).reshape(-1), log_lift.reshape(-1), (COS_BINS, LOG_LIFT_BINS), weights=ef.reshape(-1)
        )[0]
        excl = ef * (lift < 0.1)
        other = ef * (lift >= 0.1)
        for name, x in (("cos", cos), ("corr", jnp.clip(corr, -1, 1))):
            out[f"{tag}_{name}_excl"] = jnp.histogram(x, FINE_BINS, weights=excl)[0]
            out[f"{tag}_{name}_other"] = jnp.histogram(x, FINE_BINS, weights=other)[0]
    elig = upper & (p[:, :, None] >= 0.05) & (p[:, None, :] >= 0.05)
    score = jnp.where(elig, lift, jnp.inf).reshape(-1)
    _, top = jax.lax.top_k(-score, NUM_TOP_PAIRS)
    e_idx, rem = top // (I * I), top % (I * I)
    i_idx, j_idx = rem // I, rem % I
    out["top_pairs"] = jnp.stack([e_idx, i_idx, j_idx], axis=-1)
    flat = lambda x: x.reshape(-1)[top]  # noqa: E731
    out["top_lift"], out["top_z"], out["top_cos"], out["top_corr"] = flat(lift), flat(z), flat(cos), flat(corr)
    out["top_pi"], out["top_pj"] = p[e_idx, i_idx], p[e_idx, j_idx]
    out["p"] = p
    out["unit_cos_min"] = jnp.min(jnp.where(jnp.eye(I, dtype=bool)[None], 2.0, cos), axis=-1)  # [E, I]
    return out


@jax.jit
def shared_census(cofire, n, w_up, in_sum, in_m2):
    """The same census for one layer's shared expert (raw ``w_up`` on ``mlp_in``; its gated norm is per token)."""
    res = layer_census(cofire[None], n[None], w_up[None], in_sum[None], in_m2[None])
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-sequences", type=int, default=768)
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    sample_bits = np.zeros((L, E, SAMPLE, I), bool)
    sample_tok = np.full((L, E, SAMPLE), -1, np.int64)
    filled = np.zeros((L, E), np.int64)
    seq_tokens, seq_segs, seq_valid = [], [], []
    with set_mesh(mesh):
        model = common.compute_model(params)
        state = empty_state()
        step = make_step()
        key = jax.random.PRNGKey(STATS_SEED)
        offset = 0
        for n, batch in enumerate(common.train_batches(BATCH, args.train_sequences, seed=STATS_SEED)):
            tokens, seg, w = common.device_batch(batch, mesh)
            state, samples = step(state, model, tokens, seg, w, jax.random.fold_in(key, n))
            if filled.min() < SAMPLE:
                for li, smp in enumerate(samples):
                    bits, tok = np.asarray(smp["bits"]), np.asarray(smp["token"])
                    for e in np.nonzero(filled[li] < SAMPLE)[0]:
                        ok = tok[e] >= 0
                        take = min(int(ok.sum()), SAMPLE - filled[li, e])
                        sl = slice(filled[li, e], filled[li, e] + take)
                        sample_bits[li, e, sl] = bits[e][ok][:take]
                        sample_tok[li, e, sl] = tok[e][ok][:take] + offset
                        filled[li, e] += take
            offset += batch.tokens.size
            seq_tokens.append(batch.tokens.astype(np.uint16))
            seq_segs.append(batch.segment_ids.astype(np.int32))
            seq_valid.append((batch.loss_weight > 0).astype(np.uint8))
            if n % 25 == 0:
                logger.info("batch %d (%.0fs) sample min fill %d", n, time.time() - t0, filled.min())
        logger.info("pass done (%.0fs), %d tokens, dropped %s", time.time() - t0, state["tokens"], state["dropped"])
        w_eff = effective_up(model)
        census = []
        s_census = []
        m = forward.replicate(model)
        for li, lay in enumerate(m.layers()):
            census.append(
                {
                    k: np.asarray(v)
                    for k, v in layer_census(
                        state["cofire"][li], state["cnt"][li], w_eff[li], state["lat_sum"][li], state["lat_m2"][li]
                    ).items()
                }
            )
            s_census.append(
                {
                    k: np.asarray(v)
                    for k, v in shared_census(
                        state["s_cofire"][li],
                        state["tokens"],
                        lay.shared[0].w_up.astype(jnp.float32),
                        state["s_in_sum"][li],
                        state["s_in_m2"][li],
                    ).items()
                }
            )
        host = {k: np.asarray(v) for k, v in state.items()}
        w_eff = np.asarray(w_eff)
    logger.info("census done (%.0fs)", time.time() - t0)
    summary = {k: v for k, v in host.items() if k not in ("cofire", "lat_m2", "lat_sum", "s_in_m2")}
    for name, rows in (("c", census), ("sc", s_census)):
        for k in rows[0]:
            summary[f"{name}_{k}"] = np.stack([r[k] for r in rows])
    summary.update(
        floors=np.asarray(FLOORS),
        lift_thresholds=np.asarray(LIFT_THRESHOLDS),
        log_lift_bins=LOG_LIFT_BINS,
        cos_bins=COS_BINS,
        fine_bins=FINE_BINS,
        sample_filled=filled,
    )
    common.save_npz(args.tag, "excl_stats", **summary)
    for li in range(L):
        common.save_npz(
            args.tag,
            f"excl_cofire_L{li}",
            cofire=host["cofire"][li],
            cnt=host["cnt"][li],
            lat_sum=host["lat_sum"][li],
            lat_m2=host["lat_m2"][li],
            w_eff=w_eff[li],
        )
    common.save_npz(
        args.tag,
        "excl_sample",
        bits=np.packbits(sample_bits, axis=-1),
        token_index=sample_tok,
        filled=filled,
    )
    common.save_npz(
        args.tag,
        "excl_tokens",
        tokens=np.concatenate(seq_tokens),
        segment_ids=np.concatenate(seq_segs),
        valid=np.concatenate(seq_valid),
    )
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
