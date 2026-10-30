# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit errors in saved random-subset predictions without fitting or generation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from scipy.stats import rankdata, spearmanr

from experiments.evaluation.mini_policy.experiment import read_bytes, tournament


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    base = args.study
    s = json.loads(read_bytes(base + "/settings.json"))
    models = s["models"]
    bench = s["target_benchmarks"]
    assert len(bench) == 21
    assert not set(bench) & set(s["excluded_benchmarks"])
    rows = []
    for budget in [210, 840]:
        ds = [json.loads(read_bytes(base + f"/oof-random-{seed}-{budget}.json")) for seed in range(3)]
        assert all(d["benchmarks"] == bench and d["models"] == models for d in ds)
        a = np.asarray(ds[0]["actual"])
        sem = np.asarray(ds[0]["actual_sem"])
        for d in ds:
            np.testing.assert_array_equal(a, np.asarray(d["actual"]))
            np.testing.assert_array_equal(sem, np.asarray(d["actual_sem"]))
        ps = np.asarray([d["predicted"] for d in ds])
        ss = np.asarray([d["predicted_sem"] for d in ds])
        assert a.shape == (len(models), len(bench)), (a.shape, bench)
        ref = tournament(a, sem)
        pts = np.asarray([tournament(p, t) for p, t in zip(ps, ss, strict=True)])
        err = ps - a
        order = np.argsort(ref, kind="stable")
        ranks = rankdata(-ref)
        group = []
        for name, inds in zip(["lower", "middle", "upper"], np.array_split(order, 3), strict=True):
            group.append(
                dict(
                    group=name,
                    mae_pp=float(100 * np.abs(err[:, inds]).mean()),
                    bias_pp=float(100 * err[:, inds].mean()),
                    tournament_bias_pp=float(100 * (pts[:, inds] - ref[inds]).mean()),
                )
            )
        ms = [
            dict(
                model=m,
                rank=float(ranks[i]),
                predicted_rank=float(np.mean([rankdata(-x)[i] for x in pts])),
                predicted_ranks=[float(rankdata(-x)[i]) for x in pts],
                mae_pp=float(100 * np.abs(err[:, i]).mean()),
                bias_pp=float(100 * err[:, i].mean()),
            )
            for i, m in enumerate(models)
        ]
        bs = []
        for k, b in enumerate(bench):
            y = a[:, k]
            p = ps[:, :, k].mean(axis=0)
            bs.append(
                dict(
                    benchmark=b,
                    mae_pp=float(100 * np.abs(err[:, :, k]).mean()),
                    bias_pp=float(100 * err[:, :, k].mean()),
                    slope=(
                        float(np.dot(y - y.mean(), p - p.mean()) / np.dot(y - y.mean(), y - y.mean()))
                        if y.std() > 0
                        else None
                    ),
                )
            )
        empty = dict(correct_decisive=0, became_tie=0, reversed=0, correct_tie=0, false_decisive=0)
        confusion = dict(empty)
        bybench = {b: dict(empty) for b in bench}
        gaps = {b: dict(empty) for b in ["close", "far"]}
        for p, t in zip(ps, ss, strict=True):
            for i in range(len(models)):
                for j in range(i):
                    threshold = 1.96 * np.hypot(sem[i], sem[j]) / np.sqrt(0.9)
                    d = a[i] - a[j]
                    pt = 1.96 * np.hypot(t[i], t[j]) / np.sqrt(0.9)
                    pd = p[i] - p[j]
                    labels = np.where(d > threshold, 1, np.where(d < -threshold, -1, 0))
                    pl = np.where(pd > pt, 1, np.where(pd < -pt, -1, 0))
                    for k, b in enumerate(bench):
                        key = (
                            ("correct_tie" if pl[k] == 0 else "false_decisive")
                            if labels[k] == 0
                            else (
                                "correct_decisive"
                                if pl[k] == labels[k]
                                else ("became_tie" if pl[k] == 0 else "reversed")
                            )
                        )
                        confusion[key] += 1
                        bybench[b][key] += 1
                        gaps["close" if abs(ranks[i] - ranks[j]) <= 5 else "far"][key] += 1
        rows.append(
            dict(
                budget=budget,
                groups=group,
                models=sorted(ms, key=lambda x: x["mae_pp"], reverse=True),
                benchmarks=sorted(bs, key=lambda x: x["mae_pp"], reverse=True),
                confusion=confusion,
                by_benchmark=bybench,
                by_rank_gap=gaps,
                rank_spearman=[float(spearmanr(ref, p).statistic) for p in pts],
                strength_error_spearman=float(spearmanr(ref, np.abs(err).mean(axis=(0, 2))).statistic),
            )
        )
    out = dict(
        source=base,
        target_benchmarks=bench,
        definition="Descriptive audit of saved family-held-out random subsets, three seeds; "
        "strength tertiles use reference tournament ranks; no refitting or generations.",
        results=rows,
    )
    payload = json.dumps(out, indent=2).encode()
    fs, key = filesystem_for(base + "-error-audit/summary.json")
    fs.pipe_file(key, payload)
    Path(os.environ["IRIS_OUTPUT_DIR"], "summary.json").write_bytes(payload)
    print("Audit saved", flush=True)


if __name__ == "__main__":
    main()
