# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron exclusion study, output-defined units: do units that push competing next tokens (``'s`` / ``'t``, `` the``
/ `` a``, digits, ...) avoid firing together, anywhere in a layer?

1. Weights only: every routed and shared unit's direct vocabulary effect (``ngram.unit_directions``: the down
   projection, the latent out-projection, the lag-0 MLP SConv tap and the final norm gain, then the unembedding).
   A unit is *selective* when its top promoted token stands out (centred logit z >= ``MIN_Z``) and takes at least
   ``MIN_TAIL_SHARE`` of the vocabulary's mass above ``TAIL_Z`` (``sum relu(z - TAIL_Z)``); its target is that token.
   Per layer the ``MAX_UNITS`` most selective units are kept.
2. Target pairs: hand-listed sets of mutually exclusive continuations (``COMPETING_SETS``, any two different members
   compete) and pairs found in data (both frequent next tokens after the same previous token in the stats draw).
3. One pass over fresh unseen training tokens: per layer, the ``[T, S]`` activity of the selective units (zero where
   their expert is not routed; shared units always computed) gives full ``S x S`` matrices of on/off co-firing,
   strong co-firing (each unit above its own 90th percentile on its routed tokens, from ``excl_mass.npz``), the
   activation product, co-routing of the units' experts, and "unit i on while unit j's expert is routed". Per
   competing target pair, tokens are also binned by the model's ``p(A) / (p(A) + p(B))`` (where ``p(A) + p(B) >=
   RELEVANT_MASS``) and the target groups' on / strong / summed-contribution statistics accumulated per bin.

Writes ``excl_compete.npz``: per-unit selectivity for all units, the selective units, the pair lists by category with
their counts, and the binned group statistics.
"""

import argparse
import collections
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import common, forward
from experiments.grug.fast_track.interp.ngram import NONE, context_arrays, load_npz, unit_directions

logger = logging.getLogger(__name__)

BATCH = 4
L, E, I, V = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.VOCAB
K = forward.TOP_K
SHARED = E  # pseudo-expert index of the shared expert
MIN_Z = 6.0
TAIL_Z = 3.0
MIN_TAIL_SHARE = 0.2  # the median unit's top token holds 0.13 of the tail mass, the 99th percentile 0.3-0.4
MAX_UNITS = 8192
COMPETE_SEED = 23
RELEVANT_MASS = 0.2
NUM_BINS = 10  # p(A) / (p(A) + p(B)) bins; bin NUM_BINS = not relevant
MAX_GROUP_PAIRS = 384
MAX_AUTO_PAIRS = 400
AUTO_MIN_PREV = 50  # previous tokens seen at least this often
AUTO_MIN_SHARE = 0.08  # both targets at least this share of the previous token's continuations
SAMPLE_PAIRS = 60_000  # same-target and unrelated pairs sampled per layer
RSQ = "\u2019"  # typographic apostrophe
COMPETING_SETS = {
    "contraction": [
        "'s",
        "'t",
        "'re",
        "'ll",
        "'ve",
        "'m",
        "'d",
        *(RSQ + x for x in ("s", "t", "re", "ll", "ve", "m", "d")),
    ],
    "suffix": ["ics", "erion", "ically", "ical", "ic", "ism", "ist", "ize", "ized"],
    "article": [" the", " a", " an", " this", " that", " his", " her", " their", " its", " our", " my", " your"],
    "digit": [str(d) for d in range(10)] + [f" {d}" for d in range(10)],
    "preposition": [" of", " in", " to", " for", " on", " with", " at", " by", " from", " as", " into", " about"],
    "punctuation": [".", ",", ";", ":", "!", "?", ".\n", ".\n\n", ",\n"],
    "pronoun": [" he", " she", " it", " they", " we", " I", " you", " He", " She", " It", " They", " We"],
    "be": [" is", " are", " was", " were", " be", " been", " has", " have", " had"],
    "conjunction": [" and", " or", " but", " nor"],
}
FIELDS = ("n_i", "n_j", "on", "s_i", "s_j", "strong", "m_i", "m_j", "mag", "r_e", "r_f", "coroute", "i_fr", "j_er")
GROUP_FIELDS = ("n", "onA", "onB", "onAB", "sA", "sB", "sAB", "mA", "mB", "mAB")


@jax.jit
def selectivity(model) -> dict:
    """Per unit (routed ``[L, E, I]``, shared ``[L, I]``): top promoted token, its centred-logit z, the runner-up z
    and the top token's share of the tail mass ``sum relu(z - TAIL_Z)``."""
    routed, shared = unit_directions(model)
    w_out = forward.replicate(model).output_proj.astype(jnp.float32)

    def one(dirs):  # [N, D]
        logits = dirs @ w_out
        z = (logits - jnp.mean(logits, -1, keepdims=True)) / jnp.std(logits, -1, keepdims=True)
        top, idx = jax.lax.top_k(z, 2)
        tail = jnp.sum(jax.nn.relu(z - TAIL_Z), axis=-1)
        return idx[:, 0], top[:, 0], top[:, 1], jax.nn.relu(top[:, 0] - TAIL_Z) / jnp.maximum(tail, 1e-9)

    out = jax.lax.map(one, routed.reshape(L * E, I, -1))
    s_out = jax.lax.map(one, shared)
    names = ("target", "z1", "z2", "tail_share")
    res = {n: v.reshape(L, E, I) for n, v in zip(names, out, strict=True)}
    res.update({"s_" + n: v for n, v in zip(names, s_out, strict=True)})
    return res


def select_units(sel: dict, priority: set[int]) -> list[dict]:
    """Per layer: the selective units (expert, unit, target, z), those targeting a ``priority`` token first, then by
    z, at most ``MAX_UNITS``."""
    layers = []
    for li in range(L):
        rows = []
        ok = (sel["z1"][li] >= MIN_Z) & (sel["tail_share"][li] >= MIN_TAIL_SHARE)
        for e, i in zip(*np.nonzero(ok), strict=True):
            rows.append((e, i, sel["target"][li, e, i], sel["z1"][li, e, i]))
        s_ok = (sel["s_z1"][li] >= MIN_Z) & (sel["s_tail_share"][li] >= MIN_TAIL_SHARE)
        for i in np.nonzero(s_ok)[0]:
            rows.append((SHARED, i, sel["s_target"][li, i], sel["s_z1"][li, i]))
        rows.sort(key=lambda r: (int(r[2]) not in priority, -r[3]))
        rows = rows[:MAX_UNITS]
        layers.append(
            dict(
                expert=np.asarray([r[0] for r in rows], np.int32),
                unit=np.asarray([r[1] for r in rows], np.int32),
                target=np.asarray([r[2] for r in rows], np.int32),
                z=np.asarray([r[3] for r in rows], np.float32),
            )
        )
    return layers


def hand_sets(vocab: list[str]) -> dict[str, list[int]]:
    index = {s: t for t, s in enumerate(vocab)}
    return {name: [index[s] for s in members if s in index] for name, members in COMPETING_SETS.items()}


def auto_pairs(tokens: np.ndarray, seg: np.ndarray, targets: set[int]) -> list[tuple[int, int, float]]:
    """Target pairs that are both frequent continuations of the same previous token: score = sum over previous
    tokens of ``min(count(prev, A), count(prev, B))``."""
    prev, _, _ = context_arrays(tokens, seg)
    cur = tokens.astype(np.int64).reshape(-1)
    prev = prev.reshape(-1)
    ok = prev != NONE
    counts = collections.Counter(zip(prev[ok].tolist(), cur[ok].tolist(), strict=True))
    per_prev: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    totals: collections.Counter = collections.Counter()
    for (p, c), n in counts.items():
        totals[p] += n
        if c in targets:
            per_prev[p].append((c, n))
    score: collections.Counter = collections.Counter()
    for p, items in per_prev.items():
        if totals[p] < AUTO_MIN_PREV:
            continue
        keep = sorted((x for x in items if x[1] >= AUTO_MIN_SHARE * totals[p]), key=lambda x: -x[1])[:6]
        for (a, na), (b, nb) in ((x, y) for k, x in enumerate(keep) for y in keep[k + 1 :]):
            score[(min(a, b), max(a, b))] += min(na, nb)
    return [(a, b, float(s)) for (a, b), s in score.most_common(MAX_AUTO_PAIRS)]


def layer_pairs(units: dict, sets: dict[str, list[int]], auto: list, rng) -> dict[str, np.ndarray]:
    """Unit-pair index lists ``[N, 2]`` (into the layer's selective units) by category, and the competing target
    pairs used for the binned group statistics."""
    target = units["target"]
    n = len(target)
    set_of: dict[int, set[str]] = collections.defaultdict(set)
    for name, members in sets.items():
        for t in members:
            set_of[t].add(name)
    auto_set = {(a, b) for a, b, _ in auto}
    by_target: dict[int, list[int]] = collections.defaultdict(list)
    for k, t in enumerate(target.tolist()):
        by_target[t].append(k)
    present = sorted(by_target)

    def competing(a: int, b: int) -> str | None:
        if a == b:
            return None
        if set_of[a] & set_of[b]:
            return "hand"
        if (min(a, b), max(a, b)) in auto_set:
            return "auto"
        return None

    cats: dict[str, list] = {"hand": [], "auto": []}
    target_pairs: list[tuple[int, int, str]] = []
    for x, a in enumerate(present):
        for b in present[x + 1 :]:
            kind = competing(a, b)
            if kind is None:
                continue
            target_pairs.append((a, b, kind))
            cats[kind].extend((i, j) for i in by_target[a] for j in by_target[b])
    same = [(i, j) for t in present for x, i in enumerate(by_target[t]) for j in by_target[t][x + 1 :]]
    if len(same) > SAMPLE_PAIRS:
        same = [same[k] for k in rng.choice(len(same), SAMPLE_PAIRS, replace=False)]
    unrelated = []
    while len(unrelated) < SAMPLE_PAIRS and n > 1:
        i, j = rng.integers(0, n, 2)
        a, b = target[i], target[j]
        if i != j and a != b and competing(a, b) is None:
            unrelated.append((min(i, j), max(i, j)))
    out = {k: np.asarray(v, np.int32).reshape(-1, 2) for k, v in cats.items()}
    out["same"] = np.asarray(same, np.int32).reshape(-1, 2)
    out["unrelated"] = np.asarray(unrelated, np.int32).reshape(-1, 2)
    # Group pairs: hand pairs first, then auto, by the number of units they involve.
    target_pairs.sort(key=lambda p: (p[2] != "hand", -len(by_target[p[0]]) * len(by_target[p[1]])))
    target_pairs = target_pairs[:MAX_GROUP_PAIRS]
    out["group_pairs"] = np.asarray([(a, b) for a, b, _ in target_pairs], np.int32).reshape(-1, 2)
    out["group_pair_hand"] = np.asarray([k == "hand" for _, _, k in target_pairs], bool)
    return out


def make_step():
    """``dev[l]``: the layer's selective units (``expert``, ``unit``, ``norm``), ``targets [G]`` (their distinct
    targets), ``member [S, G]``, and ``pa`` / ``pb [NP]`` indices into ``targets`` of the competing group pairs."""

    @jax.jit
    def step(state, model, tokens, seg, weight, q90, s_q90, dev):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        t = tokens.size
        valid = (reshard(weight.reshape(-1), P(None)) > 0).astype(jnp.float32)
        logits = forward.soft_cap(forward._final_logits(model, tr["hidden"])).reshape(t, V)
        logz = jax.nn.logsumexp(logits, axis=-1)
        new = dict(state)
        for li, lay in enumerate(tr["layers"]):
            d = dev[li]
            experts, toks = lay["expert_sorted"], lay["token_sorted"]
            rowmap = jnp.full((t, E), -1, jnp.int32).at[toks, experts].set(jnp.arange(experts.shape[0], dtype=jnp.int32))
            is_shared = d["expert"] == SHARED
            e_safe = jnp.where(is_shared, 0, d["expert"])
            rows = rowmap[:, e_safe]  # [T, S]
            routed = (rows >= 0) | is_shared[None, :]
            act_r = lay["expert_act"].reshape(-1)[jnp.maximum(rows, 0) * I + d["unit"][None, :]]
            act_s = lay["shared_act"][:, d["unit"]]
            act = jnp.where(is_shared[None, :], act_s, jnp.where(rows >= 0, act_r, 0.0)) * valid[:, None]
            thr = jnp.where(is_shared, s_q90[li][d["unit"]], q90[li][e_safe, d["unit"]])
            on = (act > 0).astype(jnp.bfloat16)
            strong = (act >= thr[None, :]) & (act > 0)
            sb = strong.astype(jnp.bfloat16)
            rb = (routed.astype(jnp.float32) * valid[:, None]).astype(jnp.bfloat16)
            ab = act.astype(jnp.bfloat16)
            mm = lambda a, b: jnp.einsum("ts,tu->su", a, b, preferred_element_type=jnp.float32)  # noqa: E731
            new["on"] = new["on"].at[li].add(mm(on, on))
            new["strong"] = new["strong"].at[li].add(mm(sb, sb))
            new["mag"] = new["mag"].at[li].add(mm(ab, ab))
            new["coroute"] = new["coroute"].at[li].add(mm(rb, rb))
            new["on_routed"] = new["on_routed"].at[li].add(mm(on, rb))
            new["msum"] = new["msum"].at[li].add(jnp.sum(act, axis=0))
            # Binned group statistics for the competing target pairs.
            logp = jnp.take(logits, d["targets"], axis=1) - logz[:, None]  # [T, G]
            p = jnp.exp(logp)
            c = act * d["norm"][None, :]
            g_on = (jnp.einsum("ts,sg->tg", on, d["member"]) > 0).astype(jnp.float32)
            g_s = (jnp.einsum("ts,sg->tg", sb, d["member"]) > 0).astype(jnp.float32)
            g_m = jnp.einsum("ts,sg->tg", c, d["member"].astype(jnp.float32))
            pa, pb = p[:, d["pa"]], p[:, d["pb"]]  # [T, NP]
            mass = pa + pb
            frac = pa / jnp.maximum(mass, 1e-30)
            bins = jnp.where(
                mass >= RELEVANT_MASS, jnp.minimum((frac * NUM_BINS).astype(jnp.int32), NUM_BINS - 1), NUM_BINS
            )
            onehot = jax.nn.one_hot(bins, NUM_BINS + 1, dtype=jnp.float32) * valid[:, None, None]  # [T, NP, B]
            oa, ob = g_on[:, d["pa"]], g_on[:, d["pb"]]
            sa, sbb = g_s[:, d["pa"]], g_s[:, d["pb"]]
            ma, mb = g_m[:, d["pa"]], g_m[:, d["pb"]]
            fields = jnp.stack([jnp.ones_like(oa), oa, ob, oa * ob, sa, sbb, sa * sbb, ma, mb, ma * mb], axis=-1)
            new["group"] = new["group"].at[li].add(jnp.einsum("tpb,tpf->pbf", onehot, fields))
        new["tokens"] = state["tokens"] + jnp.sum(valid)
        return new

    return step


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-sequences", type=int, default=1536)
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    vocab = common.vocab_strings()
    mass = load_npz(args.tag, "excl_mass")
    sample = load_npz(args.tag, "excl_tokens")
    rng = np.random.default_rng(COMPETE_SEED)
    with set_mesh(mesh):
        model = common.compute_model(params)
        sel = {k: np.asarray(v) for k, v in selectivity(model).items()}
        logger.info("selectivity (%.0fs)", time.time() - t0)
        sets = hand_sets(vocab)
        everything = select_units(sel, set())
        all_targets = {int(t) for u in everything for t in u["target"]}
        auto = auto_pairs(sample["tokens"], sample["segment_ids"], all_targets)
        priority = {t for members in sets.values() for t in members} | {t for a, b, _ in auto for t in (a, b)}
        units = select_units(sel, priority)
        logger.info(
            "selective units per layer %s; %d auto pairs (%.0fs)",
            [len(u["target"]) for u in units],
            len(auto),
            time.time() - t0,
        )
        pairs = [layer_pairs(u, sets, auto, rng) for u in units]
        routed_norms = np.asarray(
            jnp.stack(
                [
                    jnp.linalg.norm(lay.mlp.expert_mlp.w_down.astype(jnp.float32), axis=-1)
                    for lay in forward.replicate(model).layers()
                ]
            )
        )
        shared_norms = np.asarray(
            jnp.stack(
                [
                    jnp.linalg.norm(lay.shared[0].w_down.astype(jnp.float32), axis=-1)
                    for lay in forward.replicate(model).layers()
                ]
            )
        )
        dev, groups = [], []
        for li, u in enumerate(units):
            targets = np.unique(u["target"])
            member = (u["target"][:, None] == targets[None, :]).astype(np.float32)
            gp = pairs[li]["group_pairs"]
            pos = {t: k for k, t in enumerate(targets.tolist())}
            pa = np.asarray([pos[a] for a in gp[:, 0]], np.int32)
            pb = np.asarray([pos[b] for b in gp[:, 1]], np.int32)
            norm = np.where(
                u["expert"] == SHARED,
                shared_norms[li][u["unit"]],
                routed_norms[li][np.minimum(u["expert"], E - 1), u["unit"]],
            )
            groups.append(dict(targets=targets, pa=pa, pb=pb))
            dev.append(
                dict(
                    expert=jnp.asarray(u["expert"]),
                    unit=jnp.asarray(u["unit"]),
                    targets=jnp.asarray(targets.astype(np.int32)),
                    member=jnp.asarray(member, jnp.bfloat16),
                    pa=jnp.asarray(pa),
                    pb=jnp.asarray(pb),
                    norm=jnp.asarray(norm.astype(np.float32)),
                )
            )
        s = max(len(u["target"]) for u in units)
        npairs = max(len(g["pa"]) for g in groups)
        # Pad every layer to the same S / NP so one compiled step serves all batches (padding units never fire).
        state = dict(
            on=jnp.zeros((L, s, s)),
            strong=jnp.zeros((L, s, s)),
            mag=jnp.zeros((L, s, s)),
            coroute=jnp.zeros((L, s, s)),
            on_routed=jnp.zeros((L, s, s)),
            msum=jnp.zeros((L, s)),
            group=jnp.zeros((L, npairs, NUM_BINS + 1, len(GROUP_FIELDS))),
            tokens=jnp.zeros(()),
        )
        dev = [_pad(d, s, npairs) for d in dev]
        step = make_step()
        q90, s_q90 = jnp.asarray(mass["q90"]), jnp.asarray(mass["s_q90"])
        for n, batch in enumerate(common.train_batches(BATCH, args.train_sequences, seed=COMPETE_SEED)):
            tokens, seg, w = common.device_batch(batch, mesh)
            state = step(state, model, tokens, seg, w, q90, s_q90, dev)
            if n % 50 == 0:
                logger.info("batch %d (%.0fs)", n, time.time() - t0)
        host = {k: np.asarray(v) for k, v in state.items()}
    logger.info("pass done (%.0fs), %d tokens", time.time() - t0, host["tokens"])
    out = {k: v for k, v in sel.items()}
    for li, (u, pr, g) in enumerate(zip(units, pairs, groups, strict=True)):
        n = len(u["target"])
        for k, v in u.items():
            out[f"L{li}_unit_{k}"] = v
        n_on, n_s, r = (np.diagonal(host[k][li])[:n] for k in ("on", "strong", "coroute"))
        m_sum = host["msum"][li][:n]
        for cat in ("hand", "auto", "same", "unrelated"):
            ij = pr[cat]
            i, j = ij[:, 0], ij[:, 1]
            cols = [
                n_on[i],
                n_on[j],
                host["on"][li][i, j],
                n_s[i],
                n_s[j],
                host["strong"][li][i, j],
                m_sum[i],
                m_sum[j],
                host["mag"][li][i, j],
                r[i],
                r[j],
                host["coroute"][li][i, j],
                host["on_routed"][li][i, j],
                host["on_routed"][li][j, i],
            ]
            out[f"L{li}_{cat}_pairs"] = ij
            out[f"L{li}_{cat}_stats"] = np.stack(cols, axis=-1).astype(np.float64)
        out[f"L{li}_group_pairs"] = pr["group_pairs"]
        out[f"L{li}_group_pair_hand"] = pr["group_pair_hand"]
        out[f"L{li}_group"] = host["group"][li][: len(g["pa"])]
        out[f"L{li}_unit_on"] = n_on
        out[f"L{li}_unit_strong"] = n_s
        out[f"L{li}_unit_routed"] = r
    out["auto_pairs"] = np.asarray([(a, b) for a, b, _ in auto], np.int32).reshape(-1, 2)
    out["auto_scores"] = np.asarray([s for _, _, s in auto], np.float32)
    out["tokens"] = host["tokens"]
    out["fields"] = np.asarray(FIELDS)
    out["group_fields"] = np.asarray(GROUP_FIELDS)
    for name, members in sets.items():
        out[f"set_{name}"] = np.asarray(members, np.int32)
    common.save_npz(args.tag, "excl_compete", **out)
    logger.info("done in %.0fs", time.time() - t0)


def _pad(d: dict, s: int, npairs: int) -> dict:
    """Pad a layer's unit arrays to ``s`` units and ``npairs`` group pairs so the layers stack. Padding units (expert
    0, unit 0) do fire; their rows and columns (index >= the layer's count) are dropped on the host, and they belong
    to no target group."""
    pad, npad = s - d["expert"].shape[0], npairs - d["pa"].shape[0]
    zeros = lambda n, dtype: jnp.zeros((n,), dtype)  # noqa: E731
    return dict(
        d,
        expert=jnp.concatenate([d["expert"], zeros(pad, jnp.int32)]),
        unit=jnp.concatenate([d["unit"], zeros(pad, jnp.int32)]),
        norm=jnp.concatenate([d["norm"], zeros(pad, jnp.float32)]),
        member=jnp.concatenate([d["member"], jnp.zeros((pad, d["member"].shape[1]), d["member"].dtype)]),
        pa=jnp.concatenate([d["pa"], zeros(npad, jnp.int32)]),
        pb=jnp.concatenate([d["pb"], zeros(npad, jnp.int32)]),
    )


if __name__ == "__main__":
    main()
