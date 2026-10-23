# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Field inventory of the fast_track configs: default, the Ckpt 1 / launcher values, recent-run usage.
import dataclasses
import json
import sys
import types
from pathlib import Path

_quack = types.ModuleType("levanter.grug._moe.quack_symmetric_cute")
_quack.quack_symmetric_gemm = lambda x: x
sys.modules["levanter.grug._moe.quack_symmetric_cute"] = _quack

sys.path.insert(0, "scratchpad/clean")
import hlo_hash  # noqa: E402

from experiments.grug.fast_track import belay_ckpt1, launch  # noqa: E402
from experiments.grug.fast_track import train as ft_train  # noqa: E402
from experiments.grug.fast_track.model import GrugModelConfig  # noqa: E402
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig  # noqa: E402

H100_CASES = ["belay-d512", "belay-d768", "belay-d1024", "belay-d1280", "baseline-d512", "dense-d512"]


def _settings(args, flag):
    return dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == flag)


def gb200_model():
    args = belay_ckpt1.recipe_args("gb200-d6144")
    rung = launch._h100_ladder_rung("gb200-d6144")
    model = launch._kma_model(launch._h100_ladder_model(rung))
    return launch._apply_settings(model, _settings(args, "--model-set"))


def gb200_opt():
    args = belay_ckpt1.recipe_args("gb200-d6144")
    return launch._apply_settings(GrugMoeMuonHConfig(), _settings(args, "--opt-set"))


def fmt(v):
    s = repr(v.value if hasattr(v, "value") and not isinstance(v, (int, float, str)) else v)
    return s if len(s) < 40 else s[:37] + "..."


def main():
    usage = json.loads(Path("scratchpad/clean/recent_usage_h100.json").read_text())
    usage_all = json.loads(Path("scratchpad/clean/recent_usage.json").read_text())
    configs = {case: hlo_hash.run_config(case) for case in H100_CASES}
    out = {}
    sections = {
        "GrugModelConfig": (GrugModelConfig, lambda c: c.model, gb200_model()),
        "GrugMoeMuonHConfig": (GrugMoeMuonHConfig, lambda c: c.optimizer, gb200_opt()),
        "GrugTrainerConfig": (ft_train.GrugTrainerConfig, lambda c: c.trainer, None),
        "GrugEvalConfig": (ft_train.GrugEvalConfig, lambda c: c.eval, None),
        "GrugRunConfig": (ft_train.GrugRunConfig, lambda c: c, None),
        "H100LadderRung": (launch.H100LadderRung, None, None),
    }
    for name, (cls, get, gb) in sections.items():
        rows = []
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:
                default = f.default_factory()
            else:
                default = "(required)"
            vals = {}
            if get is not None:
                for case, cfg in configs.items():
                    sub = get(cfg)
                    if sub is not None and hasattr(sub, f.name):
                        vals[case] = getattr(sub, f.name)
            gbv = getattr(gb, f.name) if gb is not None else None
            key = f.name
            ucount = usage["model"].get(key, 0) if name == "GrugModelConfig" else usage["opt"].get(key, 0)
            uall = usage_all["model"].get(key, 0) if name == "GrugModelConfig" else usage_all["opt"].get(key, 0)
            rows.append(
                dict(
                    field=f.name,
                    default=fmt(default),
                    belay={c: fmt(v) for c, v in vals.items() if c.startswith("belay") and v != default},
                    baseline={c: fmt(v) for c, v in vals.items() if not c.startswith("belay") and v != default},
                    gb200=fmt(gbv) if gb is not None and gbv != default else "",
                    recent_h100=ucount,
                    recent_all=uall,
                )
            )
        out[name] = rows
    Path("scratchpad/clean/inventory.json").write_text(json.dumps(out, indent=1))
    for name, rows in out.items():
        print(name, len(rows))


if __name__ == "__main__":
    main()
