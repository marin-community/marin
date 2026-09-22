# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive analysis of a trained omni-neurons checkpoint: where do the MLP input neurons point?

For each layer the omni MLP reads the concatenation of every prior sublayer snapshot. We load the
trained ``w_gate`` / ``w_up`` (shape ``[C*d, I]`` for ``C`` components) and measure, per component,
how much input weight the MLP puts on it -- so we can tell whether neurons attend to individual layer
branch outputs (``attn_out`` / ``mlp_out``) or to the accumulated residual-stream sums
(``resid_post_attn`` / ``resid_post_mlp``), and how that depends on source-layer recency.

Run in-region (checkpoint bucket) via iris; prints a JSON report to stdout and writes PNG plots to
``--out``. Read the checkpoint locally to the cluster -- never pulls weights across regions.
"""

import dataclasses
import io
import json

import click
import fsspec
import jax
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint

from experiments.grug.fast_track.launch import _h100_ladder_model, _h100_ladder_rung
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import GrugTrainState

_TYPES = ("embed", "attn_out", "resid_post_attn", "mlp_out", "resid_post_mlp")
_BRANCH = ("attn_out", "mlp_out")  # individual sublayer outputs
_RESID = ("resid_post_attn", "resid_post_mlp")  # accumulated residual-stream sums


def _component_labels(layer_idx: int) -> list[tuple[str, int]]:
    """Ordered (type, source_layer) for every component feeding layer ``layer_idx``'s MLP."""
    labs: list[tuple[str, int]] = [("embed", -1)]
    for j in range(layer_idx):
        labs += [("attn_out", j), ("resid_post_attn", j), ("mlp_out", j), ("resid_post_mlp", j)]
    labs += [("attn_out", layer_idx), ("resid_post_attn", layer_idx)]
    return labs


def _load_omni_model(size: str, checkpoint_path: str, component_norm: bool) -> Transformer:
    cfg = dataclasses.replace(
        _h100_ladder_model(_h100_ladder_rung(size), dense=True),
        vocab_size=16384,
        omni_mlp=True,
        omni_component_norm=component_norm,
    )
    mesh = jax.make_mesh((1, 1, 1, 1), ("replica_dcn", "data", "expert", "model"))
    jax.set_mesh(mesh)
    model_shape = jax.eval_shape(lambda k: Transformer.init(cfg, key=k), jax.random.PRNGKey(0))
    # The checkpoint stores the whole GrugTrainState; read only the fp32 ``params`` subtree (others None
    # -> not deserialized), so opt_state / master_params bytes are never fetched.
    exemplar = GrugTrainState(
        step=None,
        params=model_shape,
        master_params=None,
        ema_params=None,
        opt_state=None,
        pending_qb_betas=None,
    )
    concrete = latest_checkpoint_path(checkpoint_path) or checkpoint_path
    state = load_checkpoint(exemplar, concrete, mesh=mesh, allow_partial=True)
    return state.params


def analyze(model: Transformer) -> dict:
    d = model.config.hidden_dim
    blocks = model.omni_blocks
    assert blocks is not None, "checkpoint is not an omni model"
    per_layer = []
    for i, blk in enumerate(blocks):
        wg = np.asarray(blk.mlp.w_gate, dtype=np.float32)  # [C*d, I]
        wu = np.asarray(blk.mlp.w_up, dtype=np.float32)
        C = wg.shape[0] // d
        wg = wg.reshape(C, d, -1)
        wu = wu.reshape(C, d, -1)
        # per-component Frobenius norm of the input block (gate & up combined)
        n_gate = np.sqrt((wg**2).sum(axis=(1, 2)))
        n_up = np.sqrt((wu**2).sum(axis=(1, 2)))
        comb = np.sqrt(n_gate**2 + n_up**2)  # [C]
        share = comb / comb.sum()
        labels = _component_labels(i)
        assert len(labels) == C, (len(labels), C)
        # per-neuron argmax: which component dominates each hidden neuron's input
        per_comp_per_neuron = np.sqrt((wg**2).sum(axis=1) + (wu**2).sum(axis=1))  # [C, I]
        neuron_argmax = per_comp_per_neuron.argmax(axis=0)  # [I]
        # optional per-component learnable gain (component-norm variant)
        comp_gain = None
        if blk.omni_component_gain is not None:
            g = np.asarray(blk.omni_component_gain, dtype=np.float32)  # [C, d]
            comp_gain = np.sqrt((g**2).mean(axis=1)).tolist()  # RMS gain per component
        per_layer.append(
            {
                "layer": i,
                "num_components": int(C),
                "labels": labels,
                "component_share": share.tolist(),
                "component_norm": comb.tolist(),
                "neuron_argmax_counts": np.bincount(neuron_argmax, minlength=C).tolist(),
                "component_gain_rms": comp_gain,
            }
        )
    return {"hidden_dim": d, "num_layers": len(blocks), "per_layer": per_layer}


def _aggregate(report: dict) -> dict:
    """Roll up per-component shares into type / branch-vs-resid / recency views."""
    by_type_per_layer = []
    branch_vs_resid = []
    recency = {}  # source-layer distance -> summed share (over layers where it applies)
    neuron_type_per_layer = []
    for L in report["per_layer"]:
        share = np.array(L["component_share"])
        counts = np.array(L["neuron_argmax_counts"], dtype=float)
        type_share = {t: 0.0 for t in _TYPES}
        type_neurons = {t: 0.0 for t in _TYPES}
        for (typ, _src), s, c in zip(L["labels"], share, counts, strict=True):
            type_share[typ] += float(s)
            type_neurons[typ] += float(c)
        by_type_per_layer.append({"layer": L["layer"], **type_share})
        n = counts.sum()
        neuron_type_per_layer.append({"layer": L["layer"], **{t: type_neurons[t] / n for t in _TYPES}})
        b = sum(type_share[t] for t in _BRANCH)
        r = sum(type_share[t] for t in _RESID)
        branch_vs_resid.append({"layer": L["layer"], "branch_outputs": b, "resid_sums": r, "embed": type_share["embed"]})
        for (_typ, src), s in zip(L["labels"], share, strict=True):
            if src >= 0:
                dist = L["layer"] - src
                recency[dist] = recency.get(dist, 0.0) + float(s)
    # overall type share (weight-norm-weighted mean over layers)
    overall = {t: float(np.mean([row[t] for row in by_type_per_layer])) for t in _TYPES}
    return {
        "overall_type_share_mean_over_layers": overall,
        "by_type_per_layer": by_type_per_layer,
        "branch_vs_resid_per_layer": branch_vs_resid,
        "neuron_argmax_type_share_per_layer": neuron_type_per_layer,
        "recency_share_by_distance": dict(sorted(recency.items())),
    }


def _plots(report: dict, agg: dict, out: str) -> list[str]:
    written = []
    fs = fsspec.filesystem(out.split("://", 1)[0]) if "://" in out else fsspec.filesystem("file")

    def _save(fig, name):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        path = f"{out.rstrip('/')}/{name}"
        with fs.open(path, "wb") as f:
            f.write(buf.getvalue())
        written.append(path)

    # 1. stacked type share per layer
    layers = [r["layer"] for r in agg["by_type_per_layer"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(layers))
    for t in _TYPES:
        vals = np.array([r[t] for r in agg["by_type_per_layer"]])
        ax.bar(layers, vals, bottom=bottom, label=t)
        bottom += vals
    ax.set_xlabel("layer")
    ax.set_ylabel("input-weight-norm share")
    ax.set_title("Omni MLP input attention by component type")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _save(fig, "type_share_by_layer.png")

    # 2. branch vs resid
    fig, ax = plt.subplots(figsize=(8, 4))
    b = [r["branch_outputs"] for r in agg["branch_vs_resid_per_layer"]]
    r = [r["resid_sums"] for r in agg["branch_vs_resid_per_layer"]]
    e = [r_["embed"] for r_ in agg["branch_vs_resid_per_layer"]]
    ax.plot(layers, b, "o-", label="branch outputs (attn_out+mlp_out)")
    ax.plot(layers, r, "s-", label="residual sums (resid_post_*)")
    ax.plot(layers, e, "^-", label="embed")
    ax.set_xlabel("layer")
    ax.set_ylabel("share")
    ax.set_title("Branch outputs vs residual-stream sums")
    ax.legend(fontsize=8)
    _save(fig, "branch_vs_resid.png")

    # 3. recency curve
    rec = agg["recency_share_by_distance"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([int(k) for k in rec], list(rec.values()))
    ax.set_xlabel("source-layer distance (layer - source)")
    ax.set_ylabel("summed share")
    ax.set_title("Recency of attended components")
    _save(fig, "recency.png")
    return written


@click.command()
@click.option("--run-id", required=True)
@click.option("--size", required=True)
@click.option("--checkpoint-path", required=True)
@click.option("--component-norm", is_flag=True, help="Checkpoint was trained with omni_component_norm.")
@click.option("--out", default=None, help="S3/gs/local prefix for PNG plots (optional).")
def main(run_id, size, checkpoint_path, component_norm, out):
    model = _load_omni_model(size, checkpoint_path, component_norm)
    report = analyze(model)
    agg = _aggregate(report)
    plot_paths = _plots(report, agg, out) if out else []
    out_blob = {"run_id": run_id, "size": size, "aggregate": agg, "per_layer": report["per_layer"], "plots": plot_paths}
    print("OMNI_ANALYSIS_JSON_BEGIN")
    print(json.dumps(out_blob, indent=2))
    print("OMNI_ANALYSIS_JSON_END")


if __name__ == "__main__":
    main()
