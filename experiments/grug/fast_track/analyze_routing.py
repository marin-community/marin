# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze expert-routing dumps (``routing_step<N>.npz``, written by ``GrugTrainerConfig.routing_dump_steps``).

Each dump holds per-layer ``[V, E]`` counts of routed (token, expert) assignments: by current token
(``counts``), by previous token (``counts_prev``) and combine-weight weighted (``counts_weighted``). The report:

- per layer: mutual information I(token; expert) (bits, plus normalized by H(expert) and the plug-in
  estimate's finite-sample value under independence), the same for the previous token, expert load and
  the mean per-token routing entropy H(expert | token);
- per expert (last dump): the top tokens by lift P(e | v) / P(e), and the token-class mix P(class | e)
  against P(class);
- over time (several dumps of one run): normalized MI per layer at each step, and each expert's centered
  token profile's cosine with its own profile at the last step;
- across seeds (``--compare``): experts Hungarian-matched by the cosine of their P(v | e) profiles, raw and
  centered on P(v).

Usage::

    python -m experiments.grug.fast_track.analyze_routing run/routing/routing_step{250,1000,2817}.npz \\
        --compare seed1/routing/routing_step2817.npz --tokenizer hero-bpe-v16384 --out report.md --json report.json
"""

import dataclasses
import io
import json
import re
from collections.abc import Callable, Sequence

import click
import fsspec
import numpy as np
from levanter.tokenizers import load_tokenizer
from scipy.optimize import linear_sum_assignment

from experiments.grug.fast_track.byte_targets import token_bytes

# The fast_track ladder's tokenizer (``launch.V16384_TOKENIZER``).
DEFAULT_TOKENIZER = "hero-bpe-v16384"
_LOG2 = np.log(2.0)


@dataclasses.dataclass(frozen=True)
class RoutingDump:
    step: int
    counts: np.ndarray  # [L, V, E]
    counts_prev: np.ndarray
    counts_weighted: np.ndarray
    num_experts: int
    top_k: int
    layer_kinds: tuple[str, ...]


def load_dump(path: str) -> RoutingDump:
    with fsspec.open(path, "rb") as f:
        data = np.load(io.BytesIO(f.read()))
        return RoutingDump(
            step=int(data["step"]),
            counts=data["counts"].astype(np.float64),
            counts_prev=data["counts_prev"].astype(np.float64),
            counts_weighted=data["counts_weighted"].astype(np.float64),
            num_experts=int(data["num_experts"]),
            top_k=int(data["top_k"]),
            layer_kinds=tuple(str(k) for k in data["layer_kinds"]),
        )


@dataclasses.dataclass(frozen=True)
class MutualInfo:
    bits: float
    normalized: float  # bits / H(expert)
    bias_bits: float  # the plug-in estimate's expected value under independence, (V-1)(E-1) / 2N


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / _LOG2)


def mutual_info(counts: np.ndarray) -> MutualInfo:
    """Plug-in I(V; E) of a ``[V, E]`` count table, each assignment one sample."""
    n = counts.sum()
    joint = counts / n
    pv, pe = joint.sum(axis=1), joint.sum(axis=0)
    h_e = _entropy_bits(pe)
    bits = _entropy_bits(pv) + h_e - _entropy_bits(joint.reshape(-1))
    dof = (np.count_nonzero(pv) - 1) * (np.count_nonzero(pe) - 1)
    return MutualInfo(bits=bits, normalized=bits / h_e if h_e > 0 else 0.0, bias_bits=dof / (2 * n * _LOG2))


@dataclasses.dataclass(frozen=True)
class LayerSummary:
    layer: int
    kind: str
    mi: MutualInfo
    mi_prev: MutualInfo
    expert_entropy_bits: float  # H(E) of the load
    token_routing_entropy_bits: float  # mean over tokens of H(E | V = v), = H(E) - I
    load_max_over_mean: float
    load_min_over_mean: float


def layer_summaries(dump: RoutingDump) -> list[LayerSummary]:
    out = []
    for layer in range(dump.counts.shape[0]):
        c = dump.counts[layer][:, : dump.num_experts]
        load = c.sum(axis=0)
        mi = mutual_info(c)
        h_e = _entropy_bits(load / load.sum())
        out.append(
            LayerSummary(
                layer=layer,
                kind=dump.layer_kinds[layer],
                mi=mi,
                mi_prev=mutual_info(dump.counts_prev[layer][:, : dump.num_experts]),
                expert_entropy_bits=h_e,
                token_routing_entropy_bits=h_e - mi.bits,
                load_max_over_mean=float(load.max() / load.mean()),
                load_min_over_mean=float(load.min() / load.mean()),
            )
        )
    return out


# Token classes of the per-expert breakdown; a token can be in several.
TOKEN_CLASSES: dict[str, Callable[[str], bool]] = {
    "word_start": lambda t: len(t) > 1 and t[0] == " " and not t[1].isspace(),
    "continuation": lambda t: bool(t) and t[0].isalnum(),
    "digits": lambda t: bool(t.strip()) and t.strip().isdigit(),
    "punct": lambda t: bool(t.strip()) and all(not ch.isalnum() for ch in t.strip()),
    "upper_start": lambda t: bool(t.strip()) and t.strip()[0].isupper(),
    "whitespace_newline": lambda t: bool(t) and ("\n" in t or t.isspace()),
}


def class_masks(token_strings: Sequence[str]) -> dict[str, np.ndarray]:
    return {name: np.asarray([fn(t) for t in token_strings]) for name, fn in TOKEN_CLASSES.items()}


@dataclasses.dataclass(frozen=True)
class TokenLift:
    token: int
    text: str
    lift: float  # P(e | v) / P(e)
    p_expert_given_token: float
    count: int  # assignments of v to e


@dataclasses.dataclass(frozen=True)
class ExpertSummary:
    layer: int
    expert: int
    load: float  # P(e)
    top_tokens: list[TokenLift]
    class_ratio: dict[str, float]  # P(class | e) / P(class)
    class_share: dict[str, float]  # P(class | e)


def expert_summaries(
    counts: np.ndarray, layer: int, token_strings: Sequence[str], *, top: int, min_count: int
) -> list[ExpertSummary]:
    """Per expert of one layer's ``[V, E]`` counts: top tokens by lift (tokens with at least ``min_count``
    assignments over all experts) and the token-class mix."""
    n = counts.sum()
    n_v, n_e = counts.sum(axis=1), counts.sum(axis=0)
    p_e = n_e / n
    p_e_given_v = counts / np.maximum(n_v, 1)[:, None]
    eligible = n_v >= min_count
    masks = class_masks(token_strings)
    p_class = {name: float(n_v[m].sum() / n) for name, m in masks.items()}
    out = []
    for e in range(counts.shape[1]):
        lift = np.where(eligible, p_e_given_v[:, e] / max(p_e[e], 1e-12), -np.inf)
        order = np.argsort(-lift)[:top]
        tokens = [
            TokenLift(int(v), token_strings[v], float(lift[v]), float(p_e_given_v[v, e]), int(counts[v, e]))
            for v in order
            if np.isfinite(lift[v])
        ]
        share = {name: float(counts[m, e].sum() / max(n_e[e], 1)) for name, m in masks.items()}
        ratio = {name: share[name] / p_class[name] if p_class[name] > 0 else float("nan") for name in masks}
        out.append(ExpertSummary(layer, e, float(p_e[e]), tokens, ratio, share))
    return out


def _profiles(counts: np.ndarray, centered: bool) -> np.ndarray:
    """``[E, V]`` unit-norm P(v | e) profiles, optionally minus P(v)."""
    p = (counts / np.maximum(counts.sum(axis=0), 1)).T
    if centered:
        p = p - counts.sum(axis=1) / counts.sum()
    return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)


@dataclasses.dataclass(frozen=True)
class ExpertMatch:
    layer: int
    pairs: list[tuple[int, int]]  # (expert in A, expert in B)
    similarity: list[float]  # centered cosine of each matched pair
    raw_similarity: list[float]  # raw P(v | e) cosine of each matched pair
    unmatched_mean: float  # mean centered cosine over all (A, B) pairs, the chance level


def match_experts(a: np.ndarray, b: np.ndarray, layer: int) -> ExpertMatch:
    """Hungarian-match the experts of two ``[V, E]`` count tables by centered-profile cosine."""
    sim = _profiles(a, True) @ _profiles(b, True).T
    raw = _profiles(a, False) @ _profiles(b, False).T
    rows, cols = linear_sum_assignment(-sim)
    return ExpertMatch(
        layer=layer,
        pairs=[(int(r), int(c)) for r, c in zip(rows, cols, strict=True)],
        similarity=[float(sim[r, c]) for r, c in zip(rows, cols, strict=True)],
        raw_similarity=[float(raw[r, c]) for r, c in zip(rows, cols, strict=True)],
        unmatched_mean=float(sim.mean()),
    )


def self_similarity(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Centered-profile cosine of each expert with the same-index expert of another dump of the same run."""
    return [float(x) for x in np.sum(_profiles(a, True) * _profiles(b, True), axis=1)]


def _fmt_token(text: str) -> str:
    return "`" + text.replace("\n", "\\n").replace("\t", "\\t").replace("`", "'").replace("|", "\\|") + "`"


def build_report(
    dumps: Sequence[RoutingDump],
    token_strings: Sequence[str],
    *,
    compare: RoutingDump | None = None,
    top: int = 30,
    min_count: int = 20,
    layers: Sequence[int] | None = None,
) -> tuple[str, dict]:
    """Markdown report and a JSON-able dict for ``dumps`` (one run, step order); per-expert detail is for the
    last dump."""
    dumps = sorted(dumps, key=lambda d: d.step)
    last = dumps[-1]
    num_layers = last.counts.shape[0]
    layers = list(range(num_layers)) if layers is None else list(layers)
    e = last.num_experts
    lines = [f"# Routing report: step {last.step}, {e} experts, top-{last.top_k}", ""]
    result: dict = {"step": last.step, "num_experts": e, "top_k": last.top_k}

    summaries = layer_summaries(last)
    lines += [
        "## Per layer",
        "",
        f"H(expert) max = log2({e}) = {np.log2(e):.3f} bits. MI bias = the plug-in MI expected with no "
        "specialization, (V-1)(E-1)/2N over the tokens V and experts E seen; compare I against it.",
        "",
        "| layer | kind | I(tok;e) bits | I/H(e) | MI bias | I(prev;e) bits | I(prev)/H(e) | H(e|tok) bits | "
        "load max/mean | load min/mean |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.layer} | {s.kind} | {s.mi.bits:.3f} | {s.mi.normalized:.3f} | {s.mi.bias_bits:.3f} | "
            f"{s.mi_prev.bits:.3f} | {s.mi_prev.normalized:.3f} | {s.token_routing_entropy_bits:.3f} | "
            f"{s.load_max_over_mean:.2f} | {s.load_min_over_mean:.2f} |"
        )
    result["layers"] = [dataclasses.asdict(s) for s in summaries]

    if len(dumps) > 1:
        lines += [
            "",
            "## Over time: I(tok;e)/H(e)",
            "",
            "| layer | " + " | ".join(f"step {d.step}" for d in dumps) + " |",
        ]
        lines.append("|---|" + "---|" * len(dumps))
        history = [layer_summaries(d) for d in dumps]
        for layer in range(num_layers):
            lines.append(f"| {layer} | " + " | ".join(f"{h[layer].mi.normalized:.3f}" for h in history) + " |")
        lines += [
            "",
            f"Mean over experts of each expert's centered-profile cosine with its own step-{last.step} profile:",
            "",
        ]
        lines.append("| layer | " + " | ".join(f"step {d.step}" for d in dumps[:-1]) + " |")
        lines.append("|---|" + "---|" * (len(dumps) - 1))
        stability = {}
        for layer in range(num_layers):
            sims = [self_similarity(d.counts[layer][:, :e], last.counts[layer][:, :e]) for d in dumps[:-1]]
            stability[layer] = sims
            lines.append(f"| {layer} | " + " | ".join(f"{np.mean(s):.3f}" for s in sims) + " |")
        result["over_time"] = {
            "steps": [d.step for d in dumps],
            "normalized_mi": [[h[layer].mi.normalized for h in history] for layer in range(num_layers)],
            "self_similarity_to_last": stability,
        }

    if compare is not None:
        lines += [
            "",
            f"## Across seeds (step {last.step} vs step {compare.step})",
            "",
            "Hungarian match on the centered P(v|e) cosine; chance = mean cosine over all pairs.",
            "",
            "| layer | matched mean | matched min | matched raw mean | chance |",
            "|---|---|---|---|---|",
        ]
        matches = [
            match_experts(last.counts[layer][:, :e], compare.counts[layer][:, :e], layer) for layer in range(num_layers)
        ]
        for m in matches:
            lines.append(
                f"| {m.layer} | {np.mean(m.similarity):.3f} | {np.min(m.similarity):.3f} | "
                f"{np.mean(m.raw_similarity):.3f} | {m.unmatched_mean:.3f} |"
            )
        result["cross_seed"] = [dataclasses.asdict(m) for m in matches]

    lines += ["", f"## Per expert (step {last.step}; top {top} by lift, tokens with >= {min_count} assignments)"]
    result["experts"] = []
    for layer in layers:
        experts = expert_summaries(last.counts[layer][:, :e], layer, token_strings, top=top, min_count=min_count)
        result["experts"] += [dataclasses.asdict(x) for x in experts]
        lines += ["", f"### Layer {layer} ({last.layer_kinds[layer]})", ""]
        lines.append("| expert | load | " + " | ".join(TOKEN_CLASSES) + " | top tokens (lift) |")
        lines.append("|---|---|" + "---|" * len(TOKEN_CLASSES) + "---|")
        for x in experts:
            classes = " | ".join(f"{x.class_ratio[name]:.2f}" for name in TOKEN_CLASSES)
            tokens = " ".join(f"{_fmt_token(t.text)}({t.lift:.1f})" for t in x.top_tokens)
            lines.append(f"| {x.expert} | {x.load:.3f} | {classes} | {tokens} |")
    lines += ["", "Class columns: P(class | expert) / P(class)."]
    return "\n".join(lines) + "\n", result


def token_strings_from_tokenizer(name_or_path: str, vocab_size: int) -> list[str]:
    """Each token id's text with its leading space kept (special tokens decode to ``<special id>``)."""
    tokenizer = load_tokenizer(name_or_path)
    dot = tokenizer.encode(".", add_special_tokens=False)[0]
    special = frozenset(tokenizer.all_special_ids)
    out = []
    for idx in range(vocab_size):
        if idx in special or idx >= len(tokenizer):
            out.append(f"<special {idx}>")
            continue
        out.append(token_bytes(tokenizer, idx, dot, special).decode("utf-8", errors="replace"))
    return out


def _parse_layers(text: str | None) -> list[int] | None:
    return None if not text else [int(x) for x in re.split(r"[,\s]+", text) if x]


@click.command()
@click.argument("dumps", nargs=-1, required=True)
@click.option("--compare", default=None, help="A dump of another seed, matched against the last DUMP.")
@click.option("--tokenizer", default=DEFAULT_TOKENIZER, show_default=True, help="Tokenizer name/path, or 'none'.")
@click.option("--top", default=30, show_default=True, help="Top tokens by lift per expert.")
@click.option("--min-count", default=20, show_default=True, help="Minimum token count for the lift ranking.")
@click.option("--layers", default=None, help="Comma-separated layers for the per-expert tables (default: all).")
@click.option("--out", default=None, help="Write the markdown report here (default: stdout).")
@click.option("--json", "json_path", default=None, help="Also write the report data as JSON.")
def main(
    dumps: tuple[str, ...],
    compare: str | None,
    tokenizer: str,
    top: int,
    min_count: int,
    layers: str | None,
    out: str | None,
    json_path: str | None,
) -> None:
    loaded = [load_dump(p) for p in dumps]
    vocab = loaded[0].counts.shape[1]
    strings = [f"<{i}>" for i in range(vocab)] if tokenizer == "none" else token_strings_from_tokenizer(tokenizer, vocab)
    report, data = build_report(
        loaded,
        strings,
        compare=None if compare is None else load_dump(compare),
        top=top,
        min_count=min_count,
        layers=_parse_layers(layers),
    )
    if out is None:
        click.echo(report)
    else:
        with fsspec.open(out, "w") as f:
            f.write(report)
    if json_path is not None:
        with fsspec.open(json_path, "w") as f:
            json.dump(data, f, indent=1)


if __name__ == "__main__":
    main()
