# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Case studies: how the d512 model predicts "ics" after " crit" ("critics") and "'t" after " wasn" ("wasn't"), with
the word-piece pairs " produ" -> "cer" and " Ex" -> "ception" as contrasts (pairs the halting gate stops on).

Queries are the trigger positions in every Paloma set and unseen training-cache sequences, in ``CATEGORIES``: " crit"
followed by "ics" (the document's first visible "crit ics" or a repeat), " crit" followed by anything else, " wasn"
followed by "'t", by "\u2019t" or by anything else, and the two contrast pairs. Phases:

- scan: the next-token histogram of every ``COUNT_TRIGGERS`` position (train / Paloma) and the queries per category;
  the rows holding queries are written to ``critics_rows.npz`` (the halting-model study reads them too);
- per query on the full rows: ``sri.analyse`` with the pair's target (" crit": "ics") as the DLA target (DLA terms,
  gate and tuned lenses, top-DLA units, per-layer expert DLA, the focus units' routing and pre-activation sources);
- per query on its last ``WINDOW`` tokens: the exact attribution of ``log p(target)`` over the position's 25,344
  candidate units (``docs_analyse.attribute``) and interventions at the position with the routing frozen at the clean
  selection: sufficiency (keep the top ``k``, ablate the other candidates) and necessity (ablate the top ``k``), zero
  and mean, by the position's own ranking and by its case's global ranking (attribution summed over the case's
  positives), and single-unit ablations of the global top units;
- ablations everywhere on the full rows (``sri.ablated_logp``: units, experts, sublayers, heads) and context
  truncation.

Writes ``critics.npz``.
"""

import argparse
import dataclasses
import io
import json
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import common, docs_analyse, docs_index, forward, sri

logger = logging.getLogger(__name__)

L, H, E, I, K = forward.NUM_LAYERS, forward.NUM_HEADS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.TOP_K
SW = forward.SHARED_WIDTH
S = common.SEQ_LEN
Q, BATCH, F = sri.Q, sri.BATCH, sri.F
NR, NU = docs_analyse.NR, docs_analyse.NU
NUM_UNITS = docs_index.NUM_UNITS

CATEGORIES = (
    "crit_ics_first",
    "crit_ics_repeat",
    "crit_other",
    "produ_cer",
    "ex_ception",
    "wasn_apos_t",
    "wasn_curly_t",
    "wasn_other",
)
CAT_TARGET = (0, 0, 0, 1, 2, 3, 4, 3)  # index into PAIRS' targets
PAIRS = ((" crit", "ics"), (" produ", "cer"), (" Ex", "ception"), (" wasn", "'t"), (" wasn", "\u2019t"))
# The two case studies: their categories with the pair's continuation (the global unit ranking's queries) and all.
CASES = {"crit": ((0, 1), (0, 1, 2)), "wasn": ((5, 6), (5, 6, 7))}
COUNT_TRIGGERS = (" crit", " critic", " wasn", " produ", " Ex")
ROW_CAPS = {"crit": 700, "wasn": 600, "produ": 120, "ex": 120}
WINDOW = 512
G = docs_analyse.G
R = docs_analyse.R
INTERVENE_CAPS = {0: 10**6, 1: 10**6, 2: 160, 3: 120, 4: 120, 5: 500, 6: 300, 7: 160}
SUFF_K = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 1024, 4096, NU)
NEC_K = (1, 2, 4, 8, 16, 32, 64, 256, 1024)
GLOBAL_K = (1, 2, 4, 8, 16, 32, 64, 128)
SINGLE_TOP = 24
GLOBAL_TOP = 256
TRUNC = (1, 2, 3, 4, 8, 16, 64, 256, 1024)
ABLATE_TOP = 12
CONTEXT_LEFT = 24
TRAIN_SLICE = 16  # scan chunks fetched per dataset call


@dataclasses.dataclass(frozen=True)
class Ids:
    triggers: tuple[int, ...]
    targets: tuple[int, ...]
    count_triggers: tuple[int, ...]
    bos: int


def resolve_ids(vocab: list[str]) -> Ids:
    index: dict[str, int] = {}
    for i, s in enumerate(vocab):
        index.setdefault(s, i)
    return Ids(
        triggers=tuple(index[a] for a, _ in PAIRS),
        targets=tuple(index[b] for _, b in PAIRS),
        count_triggers=tuple(index[s] for s in COUNT_TRIGGERS),
        bos=index["<|begin_of_text|>"],
    )


def next_tokens(tokens: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Each position's next token within its document (-1 at a document's last token)."""
    nxt = np.full(len(tokens), -1, np.int64)
    nxt[:-1] = np.where(seg[1:] == seg[:-1], tokens[1:], -1)
    return nxt


def row_queries(tokens: np.ndarray, seg: np.ndarray, weight: np.ndarray, ids: Ids) -> list[tuple[int, ...]]:
    """``(pos, cat, doc_offset, doc_start_visible, prior_positives)`` of every query in one row."""
    n = len(tokens)
    start = np.concatenate([[True], seg[1:] != seg[:-1]])
    doc_start = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    nxt = next_tokens(tokens, seg)
    ok = (weight > 0) & (nxt >= 0)
    out = []
    seen: dict[int, int] = {}
    for t in np.flatnonzero(ok & (tokens == ids.triggers[0])):
        ds = int(doc_start[t])
        prior = seen.get(ds, 0)
        if nxt[t] == ids.targets[0]:
            cat = 1 if prior else 0
            seen[ds] = prior + 1
        else:
            cat = 2
        visible = int(ds > 0 or tokens[0] == ids.bos)
        out.append((int(t), cat, int(t - ds), visible, prior))
    for p in (1, 2):
        for t in np.flatnonzero(ok & (tokens == ids.triggers[p]) & (nxt == ids.targets[p])):
            ds = int(doc_start[t])
            out.append((int(t), 2 + p, int(t - ds), int(ds > 0 or tokens[0] == ids.bos), 0))
    seen = {}
    for t in np.flatnonzero(ok & (tokens == ids.triggers[3])):
        ds = int(doc_start[t])
        prior = seen.get(ds, 0)
        cat = 5 if nxt[t] == ids.targets[3] else 6 if nxt[t] == ids.targets[4] else 7
        if cat != 7:
            seen[ds] = prior + 1
        out.append((int(t), cat, int(t - ds), int(ds > 0 or tokens[0] == ids.bos), prior))
    return out


def row_group(queries: list[tuple[int, ...]]) -> str | None:
    cats = {q[1] for q in queries}
    if cats & {0, 1, 2}:
        return "crit"
    if cats & {5, 6, 7}:
        return "wasn"
    if 3 in cats:
        return "produ"
    if 4 in cats:
        return "ex"
    return None


@dataclasses.dataclass
class Scan:
    """Kept rows and the continuation counts of every scanned position."""

    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list]] = dataclasses.field(default_factory=list)
    counts: dict[str, int] = dataclasses.field(default_factory=lambda: {g: 0 for g in ROW_CAPS})
    # [source (train, paloma), trigger, V] next-token counts
    next_hist: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((2, len(COUNT_TRIGGERS), 16384)))
    cat_counts: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((2, len(CATEGORIES))))  # [source, cat]
    tokens_scanned: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2))

    def full(self) -> bool:
        return all(self.counts[g] >= c for g, c in ROW_CAPS.items())

    def add(self, batch: common.TokenBatch, ids: Ids, src: int) -> None:
        for r in range(batch.tokens.shape[0]):
            w = batch.loss_weight[r]
            if w.sum() == 0:
                continue
            tok, seg = batch.tokens[r], batch.segment_ids[r]
            nxt = next_tokens(tok, seg)
            ok = (w > 0) & (nxt >= 0)
            self.tokens_scanned[src] += ok.sum()
            for j, trig in enumerate(ids.count_triggers):
                m = ok & (tok == trig)
                if m.any():
                    self.next_hist[src, j] += np.bincount(nxt[m], minlength=16384)
            qs = row_queries(tok, seg, w, ids)
            for q in qs:
                self.cat_counts[src, q[1]] += 1
            group = row_group(qs)
            if group is None or self.counts[group] >= ROW_CAPS[group]:
                continue
            self.counts[group] += 1
            self.rows.append((batch.source, tok.copy(), seg.copy(), w.copy(), qs))


def scan(ids: Ids, max_train: int, chunk: int) -> Scan:
    out = Scan()
    t0 = time.time()
    for b in common.paloma_batches(64):
        out.add(b, ids, 1)
    logger.info("paloma: %s rows, %d tokens (%.0fs)", out.counts, out.tokens_scanned[1], time.time() - t0)
    _scan_train(out, ids, common.TRAINED_SEQUENCES, max_train, chunk)
    logger.info("train: %s rows, %d tokens (%.0fs)", out.counts, out.tokens_scanned[0], time.time() - t0)
    return out


@common.off_mesh
def _train_chunks(start: int, num: int, chunk: int):
    ds = common.data_config().train_sets(Axis("position", S), key=jax.random.PRNGKey(0))["train"].as_sync_dataset()
    end = min(start + num, len(ds))
    for s in range(start, end, chunk):
        yield common.examples_to_batch("train", ds.get_batch(list(range(s, min(end, s + chunk)))))


def _scan_train(out: Scan, ids: Ids, start: int, num: int, chunk: int) -> None:
    # Fetch in slices so the scan can stop once every group is full.
    for s in range(start, start + num, TRAIN_SLICE * chunk):
        for b in _train_chunks(s, min(TRAIN_SLICE * chunk, start + num - s), chunk):
            out.add(b, ids, 0)
        logger.info("train scanned %d: %s", s + TRAIN_SLICE * chunk - start, out.counts)
        if out.full():
            return


def make_batches(rows) -> list[tuple[common.TokenBatch, np.ndarray, list]]:
    """Rows in batches of ``BATCH`` with their query table ``[Q, 6]``: row, pos, cat, doc offset, doc start visible,
    prior positives (unused slots: row -1); queries past ``Q`` in a batch are dropped."""
    out = []
    for group in ROW_CAPS:
        g = [r for r in rows if row_group(r[4]) == group]
        for s in range(0, len(g), BATCH):
            chunk = g[s : s + BATCH]
            qs = [(bi, q) for bi, r in enumerate(chunk) for q in r[4]][:Q]
            table = np.full((Q, 6), -1, np.int32)
            for j, (bi, q) in enumerate(qs):
                table[j] = (bi, *q)
            pad = BATCH - len(chunk)
            tokens = np.stack([r[1] for r in chunk] + [np.zeros(S, np.int32)] * pad)
            seg = np.stack([r[2] for r in chunk] + [np.zeros(S, np.int32)] * pad)
            weight = np.stack([r[3] for r in chunk] + [np.zeros(S, np.float32)] * pad)
            out.append((common.TokenBatch(group, tokens, seg, weight), table, chunk))
    return out


def query_arrays(table: np.ndarray, ids: Ids) -> dict[str, np.ndarray]:
    valid = table[:, 0] >= 0
    cat = np.where(valid, table[:, 2], 0)
    yi = np.asarray(CAT_TARGET)[cat]
    return dict(
        qflat=np.where(valid, table[:, 0] * S + table[:, 1], 0),
        qmask=valid,
        qy=np.asarray(ids.targets)[yi],
        qyi=yi,
        qcat=cat,
    )


def window(tokens: np.ndarray, doc_start: int, t: int, bos: int) -> tuple[np.ndarray, int]:
    """The query's last tokens (through its label) as one document of at most ``WINDOW`` tokens, after a BOS unless
    the window reaches the document start; returns the window and the query's position in it."""
    s0 = max(doc_start, t + 2 - (WINDOW - 1))
    w = tokens[s0 : t + 2]
    if s0 > doc_start or tokens[doc_start] != bos:
        w = np.concatenate([[bos], w])
    return w.astype(np.int32), len(w) - 2


@jax.jit
def intervene(model, tokens, seg, targets, keep, unit_mean, shared_mean, selected):
    """``log p`` of each row's label at ``targets`` with the candidates at that position kept where ``keep`` ``[R,
    NU]`` (else replaced by ``unit_mean`` / ``shared_mean``), the routing at the position frozen to ``selected`` ``[R,
    L, K]``."""
    m = forward.replicate(model)
    r, s = tokens.shape
    rows = jnp.arange(r)
    kf = keep.astype(jnp.bfloat16)
    rk = kf[:, :NR].reshape(r, L, K, I).transpose(1, 0, 2, 3)
    sk = kf[:, NR:].reshape(r, L, SW).transpose(1, 0, 2)
    routed_keep = jnp.ones((L, r, s, K, I), jnp.bfloat16).at[:, rows, targets].set(rk).reshape(L, r * s, K, I)
    shared_keep = jnp.ones((L, r, s, SW), jnp.bfloat16).at[:, rows, targets].set(sk).reshape(L, r * s, SW)
    q = rows * s + targets
    routing = forward.no_override(r * s)._replace(
        sel_mask=jnp.zeros((L, r * s), bool).at[:, q].set(True),
        sel_override=jnp.zeros((L, r * s, K), jnp.int32).at[:, q].set(selected.transpose(1, 0, 2)),
    )
    abl = forward.no_ablation()._replace(neuron_mean=unit_mean, shared_mean=shared_mean)
    edit = forward.UnitEdit(routed_keep=routed_keep, shared_keep=shared_keep)
    tr = forward.trace_forward(m, tokens, seg, abl, routing=routing, edit=edit)
    return -forward.replicate(tr["loss"])[rows, targets]


def _mask_top(order: np.ndarray, k: int) -> np.ndarray:
    """The top ``k`` of a descending candidate ``order``."""
    keep = np.zeros(NU, bool)
    keep[order[:k]] = True
    return keep


def attribution_pass(model, mesh, windows, means) -> dict[str, np.ndarray]:
    """``docs_analyse.attribute`` over the windows, ``G`` at a time."""
    res: dict[str, list] = {}
    s0_all = np.zeros((len(windows), NU), np.float32)
    sm_all = np.zeros((len(windows), NU), np.float32)
    for c in range(0, len(windows), G):
        chunk = windows[c : c + G]
        chunk = chunk + [chunk[-1]] * (G - len(chunk))
        tok = np.zeros((G, WINDOW), np.int32)
        seg = np.ones((G, WINDOW), np.int32)
        tg = np.zeros(G, np.int32)
        for j, (w, t) in enumerate(chunk):
            tok[j, : len(w)] = w
            seg[j, : len(w)] = 0
            tg[j] = t
        n = min(G, len(windows) - c)
        with set_mesh(mesh):
            out, s0, sm = docs_analyse.attribute(
                model, jnp.asarray(tok), jnp.asarray(seg), jnp.asarray(tg), means[0], means[1]
            )
            out = jax.device_get(out)
            s0_all[c : c + n] = np.asarray(s0)[:n]
            sm_all[c : c + n] = np.asarray(sm)[:n]
        for k, v in out.items():
            res.setdefault(k, []).append(np.asarray(v)[:n])
    out = {k: np.concatenate(v) for k, v in res.items()}
    out["s0"], out["sm"] = s0_all, sm_all
    return out


def intervention_pass(model, mesh, windows, attr, means, global_top: np.ndarray) -> dict[str, np.ndarray]:
    """Every intervention of every window (see the module docstring); ``global_top`` ``[n, GLOBAL_TOP]`` is each
    window's global unit ranking."""
    n = len(windows)
    selected = attr["selected"]  # [n, L, K]
    uids = np.stack([docs_analyse.candidate_unit_ids(selected[j]) for j in range(n)]).astype(np.int32)  # [n, NU]
    order_0 = np.argsort(-attr["s0"], axis=1, kind="stable").astype(np.int32)
    order_m = np.argsort(-attr["sm"], axis=1, kind="stable").astype(np.int32)
    layout = {
        "suff_zero": ("zero", len(SUFF_K)),
        "suff_mean": ("mean", len(SUFF_K)),
        "nec_zero": ("zero", len(NEC_K)),
        "nec_mean": ("mean", len(NEC_K)),
        "gsuff_zero": ("zero", len(GLOBAL_K)),
        "gnec_zero": ("zero", len(GLOBAL_K)),
        "single_zero": ("zero", SINGLE_TOP),
        "single_mean": ("mean", SINGLE_TOP),
    }

    def keep_mask(name: str, j: int, slot: int) -> np.ndarray:
        """The candidates kept at window ``j``'s position by intervention ``name`` at ``slot``."""
        order = order_m[j] if name.endswith("_mean") else order_0[j]
        if name.startswith("suff"):
            return _mask_top(order, SUFF_K[slot])
        if name.startswith("nec"):
            return ~_mask_top(order, NEC_K[slot])
        if name.startswith("gsuff"):
            return np.isin(uids[j], global_top[j, : GLOBAL_K[slot]])
        if name.startswith("gnec"):
            return ~np.isin(uids[j], global_top[j, : GLOBAL_K[slot]])
        return uids[j] != global_top[j, slot]

    specs: dict[str, list] = {"zero": [], "mean": []}
    for j in range(n):
        for name, (mode, width) in layout.items():
            specs[mode] += [(name, j, slot) for slot in range(width)]
    out = {name: np.full((n, width), np.nan, np.float32) for name, (_, width) in layout.items()}
    out["single_present"] = np.stack([np.isin(global_top[j, :SINGLE_TOP], uids[j]) for j in range(n)])
    zeros = (jnp.zeros_like(means[0]), jnp.zeros_like(means[1]))
    t0 = time.time()
    for mode, jobs in specs.items():
        um, sm_ = zeros if mode == "zero" else means
        for c in range(0, len(jobs), R):
            chunk = jobs[c : c + R]
            chunk = chunk + [chunk[-1]] * (R - len(chunk))
            tok = np.zeros((R, WINDOW), np.int32)
            seg = np.ones((R, WINDOW), np.int32)
            tg = np.zeros(R, np.int32)
            for r, (_, j, _) in enumerate(chunk):
                w, t = windows[j]
                tok[r, : len(w)] = w
                seg[r, : len(w)] = 0
                tg[r] = t
            keep = np.stack([keep_mask(*x) for x in chunk])
            sel = np.stack([selected[x[1]] for x in chunk]).astype(np.int32)
            with set_mesh(mesh):
                lp = intervene(
                    model,
                    jnp.asarray(tok),
                    jnp.asarray(seg),
                    jnp.asarray(tg),
                    jnp.asarray(keep),
                    um,
                    sm_,
                    jnp.asarray(sel),
                )
                lp = np.asarray(jax.device_get(lp))
            for (name, j, slot), v in zip(chunk, lp, strict=True):
                out[name][j, slot] = v
            if (c // R) % 100 == 0:
                logger.info("interventions %s %d/%d (%.0fs)", mode, c, len(jobs), time.time() - t0)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="tn-d512-ckpt1-rno")
    parser.add_argument("--scan-train", type=int, default=240000)
    parser.add_argument("--scan-chunk", type=int, default=512)
    parser.add_argument("--out", default="critics", help="output name (<out>.npz, <out>_rows.npz)")
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    vocab = common.vocab_strings()
    ids = resolve_ids(vocab)
    logger.info("ids %s", ids)

    sc = scan(ids, args.scan_train, args.scan_chunk)
    batches = make_batches(sc.rows)
    table_all = np.concatenate([t for _, t, _ in batches])
    keep_q = table_all[:, 0] >= 0
    rows_tokens = np.stack([r[1] for _, _, chunk in batches for r in chunk])
    rows_seg = np.stack([r[2] for _, _, chunk in batches for r in chunk])
    row_offset = np.cumsum([0] + [len(chunk) for _, _, chunk in batches])
    q_rows = np.concatenate(
        [np.where(t[:, 0] >= 0, t[:, 0] + row_offset[bi], -1) for bi, (_, t, _) in enumerate(batches)]
    )
    counts = dict(
        next_hist=sc.next_hist.astype(np.int64),
        cat_counts=sc.cat_counts,
        tokens_scanned=sc.tokens_scanned,
        count_triggers=np.asarray(COUNT_TRIGGERS),
    )
    # The query rows for the halting-model study: one table row per query (global row, pos, cat, offset, vis, prior).
    common.save_npz(
        args.tag,
        f"{args.out}_rows",
        tokens=rows_tokens,
        segment_ids=rows_seg,
        queries=np.concatenate([q_rows[keep_q, None], table_all[keep_q, 1:]], axis=1),
        source=np.asarray([r[0] for _, _, chunk in batches for r in chunk]),
        **counts,
    )
    logger.info("scan saved (%.0fs): %d rows, %d queries", time.time() - t0, len(rows_tokens), keep_q.sum())

    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/attrib.npz", "rb") as f:
        attrib = np.load(io.BytesIO(f.read()))
        tuned = jnp.asarray(attrib["tuned"], jnp.float32)
        head_mean, sub_mean = jnp.asarray(attrib["head_mean"]), jnp.asarray(attrib["sub_mean"])
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/stats.npz", "rb") as f:
        stats = np.load(io.BytesIO(f.read()))
        stats_mean = stats["act_sum"] / np.maximum(stats["expert_tokens"][..., None], 1.0)
    um, smean = docs_analyse.unit_means(args.tag)

    targets = jnp.asarray(ids.targets)
    with set_mesh(mesh):
        model = common.compute_model(params)
        rdir, rsdir = sri.neuron_directions(model, targets)
        means = (jnp.asarray(um), jnp.asarray(smean))

    def run_pass(focus_by_pair: np.ndarray, bset=None) -> tuple[dict, dict]:
        per_q: dict[str, list] = {}
        acc: dict = {}
        for bi, (b, table, _) in enumerate(batches):
            if bset is not None and bi not in bset:
                continue
            qa = query_arrays(table, ids)
            with set_mesh(mesh):
                tokens, seg, _ = common.device_batch(b, mesh)
                out, a = sri.analyse_jit(
                    model,
                    tokens,
                    seg,
                    jnp.asarray(qa["qflat"]),
                    jnp.asarray(qa["qmask"]),
                    jnp.asarray(qa["qy"]),
                    jnp.asarray(qa["qyi"]),
                    jnp.asarray(qa["qcat"]),
                    jnp.asarray(focus_by_pair[qa["qyi"]].astype(np.int32)),
                    tuned,
                    rdir,
                    rsdir,
                )
                out, a = jax.device_get((out, a))
            keep = qa["qmask"]
            for k, v in out.items():
                v = np.asarray(v)
                per_q.setdefault(k, []).append(v[keep] if v.ndim and v.shape[0] == Q else v[None])
            for k, v in a.items():
                acc[k] = np.asarray(v) if k not in acc else acc[k] + np.asarray(v)
        return {k: np.concatenate(v) for k, v in per_q.items()}, acc

    # Pass 1: unit DLA sums per category (focus placeholder); pass 2 with each pair's top-DLA units as the focus.
    placeholder = np.zeros((len(PAIRS), F, 3), np.int32)
    _, acc1 = run_pass(placeholder)
    n_cat = np.bincount(table_all[keep_q, 2], minlength=sri.C).astype(np.float64)
    focus = np.zeros((len(PAIRS), F, 3), np.int32)
    top_by_pair = []
    for p in range(len(PAIRS)):
        cats = [c for c in range(len(CATEGORIES)) if CAT_TARGET[c] == p and not CATEGORIES[c].endswith("_other")]
        score = acc1["neuron_dla"][cats].sum(0) / max(n_cat[cats].sum(), 1)
        order = np.argsort(-score.reshape(-1))[:50]
        top = np.stack(np.unravel_index(order, score.shape), axis=1)
        top_by_pair.append(top)
        focus[p] = top[:F]
    logger.info("focus %s (%.0fs)", focus.tolist(), time.time() - t0)
    per_q, acc2 = run_pass(focus)
    logger.info("analyse done (%.0fs)", time.time() - t0)

    # Windows for attribution and interventions.
    meta = table_all[keep_q]  # local row, pos, cat, offset, visible, prior
    gq_rows = q_rows[keep_q]
    rng = np.random.default_rng(0)
    pick = []
    for c in range(len(CATEGORIES)):
        idx = np.flatnonzero(meta[:, 2] == c)
        if len(idx) > INTERVENE_CAPS[c]:
            idx = np.sort(rng.choice(idx, INTERVENE_CAPS[c], replace=False))
        pick += idx.tolist()
    pick = np.asarray(sorted(pick))
    windows = [window(rows_tokens[gq_rows[j]], int(meta[j, 1] - meta[j, 3]), int(meta[j, 1]), ids.bos) for j in pick]
    attr = attribution_pass(model, mesh, windows, means)
    logger.info("attribution done for %d windows (%.0fs)", len(windows), time.time() - t0)
    # Per case, the units ranked by attribution summed over its positive windows; each window's global
    # interventions use its own case's ranking (the contrast pairs use crit's).
    cat_w = meta[pick, 2]
    global_top, global_sum, global_top_mean = {}, {}, {}
    window_case = np.zeros(len(pick), np.int32)
    for ci, (case, (positive, members)) in enumerate(CASES.items()):
        gsum = np.zeros(NUM_UNITS, np.float64)
        gsum_m = np.zeros(NUM_UNITS, np.float64)
        for j in np.flatnonzero(np.isin(cat_w, positive)):
            uid = docs_analyse.candidate_unit_ids(attr["selected"][j])
            np.add.at(gsum, uid, attr["s0"][j])
            np.add.at(gsum_m, uid, attr["sm"][j])
        global_top[case] = np.argsort(-gsum)[:GLOBAL_TOP]
        global_sum[case] = gsum[global_top[case]]
        global_top_mean[case] = np.argsort(-gsum_m)[:GLOBAL_TOP]
        window_case[np.isin(cat_w, members)] = ci
        logger.info("%s global top units %s", case, global_top[case][:12].tolist())
    tops = np.stack([global_top[c] for c in CASES])
    interventions = intervention_pass(model, mesh, windows, attr, means, tops[window_case])
    clean_err = np.nanmax(np.abs(interventions["suff_zero"][:, -1] - attr["logp"]))
    logger.info("keep-all intervention vs clean log p: max |diff| %.4g", clean_err)
    logger.info("interventions done (%.0fs)", time.time() - t0)

    # Ablations everywhere on the full rows.
    base = forward.no_ablation()
    nmean = jnp.asarray(stats_mean, jnp.float32)

    def units(spec, mode):
        mult = base.neuron_mult
        for li, e, i in spec:
            mult = mult.at[li, e, i].set(0.0)
        return base._replace(neuron_mult=mult, neuron_mean=nmean if mode == "mean" else base.neuron_mean)

    def expert(li, e):
        return base._replace(neuron_mult=base.neuron_mult.at[li, e].set(0.0))

    def sub(li, slot, mode):
        return base._replace(
            sub_mult=base.sub_mult.at[li, slot].set(0.0), sub_mean=sub_mean if mode == "mean" else base.sub_mean
        )

    expert_dla = per_q["expert_dla"]  # [Q, L, E]
    configs = [("none", base)]
    # Per case (top-DLA units of its main pair: crit -> "ics", wasn -> "'t").
    for case, pair in (("crit", 0), ("wasn", 3)):
        top_units = [tuple(int(x) for x in t) for t in top_by_pair[pair]]
        for t in top_units[:ABLATE_TOP]:
            configs.append((f"{case}:L{t[0]}.E{t[1]}.n{t[2]}:zero", units([t], "zero")))
        for n in (2, 5, 10, 25, 50):
            configs.append((f"{case}:top{n}:zero", units(top_units[:n], "zero")))
            configs.append((f"{case}:top{n}:mean", units(top_units[:n], "mean")))
        top_experts = np.argmax(expert_dla[np.isin(meta[:, 2], CASES[case][0])].mean(0), axis=-1)  # [L]
        for li in range(L):
            configs.append((f"{case}:L{li}.E{int(top_experts[li])}:zero", expert(li, int(top_experts[li]))))
    for li in range(L):
        configs.append((f"L{li}.routed:mean", sub(li, forward.SUB_ROUTED, "mean")))
        configs.append((f"L{li}.shared:mean", sub(li, forward.SUB_SHARED, "mean")))
        configs.append((f"L{li}.attn:mean", sub(li, forward.SUB_ATTN, "mean")))
        for h in range(H):
            configs.append(
                (f"L{li}.h{h}:mean", base._replace(head_mult=base.head_mult.at[li, h].set(0.0), head_mean=head_mean))
            )
    for li in forward.MLA_LAYERS:
        configs.append((f"L{li}.ve:zero", base._replace(sub_mult=base.sub_mult.at[li, forward.SUB_VE].set(0.0))))
    late = base.sub_mult.at[4].set(0.0).at[5].set(0.0)
    configs.append(("L4-L5:mean", base._replace(sub_mult=late, sub_mean=sub_mean)))
    abl_names, abl_logp = [], []
    for name, abl in configs:
        vals = []
        for b, table, _ in batches:
            qa = query_arrays(table, ids)
            with set_mesh(mesh):
                tokens, seg, _ = common.device_batch(b, mesh)
                lp = sri.ablated_logp(model, tokens, seg, jnp.asarray(qa["qflat"]), jnp.asarray(qa["qy"]), abl)
            vals.append(np.asarray(lp)[qa["qmask"]])
        abl_names.append(name)
        abl_logp.append(np.concatenate(vals))
    logger.info("ablations done (%.0fs)", time.time() - t0)

    # Context truncation of the positive queries and the contrasts.
    rows_q = []
    other = [c for c, name in enumerate(CATEGORIES) if name.endswith("_other")]
    for qi in np.flatnonzero(~np.isin(meta[:, 2], other)):
        rows_q.append((rows_tokens[gq_rows[qi]], int(meta[qi, 1] - meta[qi, 3]), int(meta[qi, 1]), int(qi)))
    trunc: dict[str, list] = {}
    for tb, ttable in sri.truncated_batches(rows_q, ids.bos, TRUNC):
        valid = ttable[:, 0] >= 0
        qid = np.where(valid, ttable[:, 1], 0)
        cat = meta[qid, 2]
        yi = np.asarray(CAT_TARGET)[cat]
        with set_mesh(mesh):
            tok_d, seg_d, _ = common.device_batch(tb, mesh)
            out, _ = sri.analyse_jit(
                model,
                tok_d,
                seg_d,
                jnp.asarray(np.where(valid, ttable[:, 0], 0)),
                jnp.asarray(valid),
                jnp.asarray(np.asarray(ids.targets)[yi]),
                jnp.asarray(yi),
                jnp.asarray(cat),
                jnp.asarray(focus[yi].astype(np.int32)),
                tuned,
                rdir,
                rsdir,
            )
        trunc.setdefault("trunc/query", []).append(qid[valid])
        trunc.setdefault("trunc/length_index", []).append(ttable[valid, 2])
        for k in ("logp_y", "rank_y", "f_act", "f_selected", "dla_terms"):
            trunc.setdefault(f"trunc/{k}", []).append(np.asarray(out[k])[valid])
    logger.info("truncation done (%.0fs)", time.time() - t0)

    ctx = np.stack(
        [
            np.pad(rows_tokens[gq_rows[j]][max(0, meta[j, 1] - CONTEXT_LEFT + 1) : meta[j, 1] + 1], (CONTEXT_LEFT, 0))[
                -CONTEXT_LEFT:
            ]
            for j in range(len(meta))
        ]
    )
    results: dict[str, np.ndarray] = {f"q/{k}": v for k, v in per_q.items()}
    results.update({f"acc/{k}": v for k, v in acc2.items() if k.startswith("neuron") or k == "shared_dla"})
    results.update(
        {
            "q/meta": meta,
            "q/row": gq_rows,
            "q/context": ctx,
            "n_cat": n_cat,
            "focus": focus,
            "top_by_pair": np.stack(top_by_pair),
            "abl/names": np.asarray(abl_names),
            "abl/logp": np.stack(abl_logp),
            "w/pick": pick,
            **{f"global/{c}/top": v for c, v in global_top.items()},
            **{f"global/{c}/sum": v for c, v in global_sum.items()},
            **{f"global/{c}/top_mean": v for c, v in global_top_mean.items()},
            "w/case": window_case,
            "suff_k": np.asarray(SUFF_K),
            "nec_k": np.asarray(NEC_K),
            "global_k": np.asarray(GLOBAL_K),
            "trunc_lengths": np.asarray(TRUNC),
            "categories": np.asarray(CATEGORIES),
            "terms": np.asarray(sri.TERMS),
            "ids_json": np.asarray(json.dumps(dataclasses.asdict(ids))),
            **{f"count/{k}": v for k, v in counts.items()},
        }
    )
    for k, v in attr.items():
        if k not in ("s0", "sm"):
            results[f"w/{k}"] = v
    results["w/top_uid"] = np.stack(
        [docs_analyse.candidate_unit_ids(attr["selected"][j])[attr["top_idx"][j]] for j in range(len(pick))]
    ).astype(np.int32)
    results.update({f"w/{k}": v for k, v in interventions.items()})
    results.update({k: np.concatenate(v) for k, v in trunc.items()})
    common.save_npz(args.tag, args.out, **results)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
