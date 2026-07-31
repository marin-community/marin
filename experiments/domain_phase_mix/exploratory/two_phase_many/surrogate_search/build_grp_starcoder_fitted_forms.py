# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Write LaTeX for the fitted 2-phase StarCoder GRP forms."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (
    fit_starcoder_grp,
    load_completed_two_phase_starcoder_packet,
    subset_packet,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_TEX = SCRIPT_DIR / "grp_starcoder_fitted_forms.tex"
OUTPUT_JSON = SCRIPT_DIR / "grp_starcoder_fitted_forms.json"
OUTPUT_SLIDE_TEX = SCRIPT_DIR / "grp_starcoder_fitted_forms_slide.tex"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _sci(value: float, digits: int = 3) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _model_block(
    *,
    title: str,
    alpha: float,
    eta: float,
    lam: float,
    tau: float,
    reg: float,
    intercept: float,
    coef_n: float,
    coef_s: float,
    coef_penalty: float,
    c0_n: float,
    c0_s: float,
    c1_n: float,
    c1_s: float,
) -> str:
    xn_phase1 = eta * c1_n
    xs_phase1 = eta * c1_s
    lam_str = _fmt(lam, 6) if lam >= 1e-4 else _sci(lam, 3)
    penalty_str = _fmt(coef_penalty, 4) if abs(coef_penalty) >= 1e-4 else _sci(coef_penalty, 3)
    with_line = (
        rf"\text{{with }}\alpha={_fmt(alpha)},\;\eta={_fmt(eta)},\;"
        rf"\lambda={lam_str},\;\tau={_fmt(tau)},\;\mathrm{{reg}}={_sci(reg, 3)}."
    )
    slice_line = (
        r"\text{On the U-shaped slice }"
        r"(p_{0,\mathrm{N}}=1,\;p_{0,\mathrm{S}}=0,\;p_{1,\mathrm{S}}=z,\;"
        r"p_{1,\mathrm{N}}=1-z):"
    )
    block = rf"""
\paragraph{{{title}}}
\[
x_{{\mathrm{{N}}}}
= e^{{-{lam_str}(1-p_{{1,\mathrm{{N}}}})}} \cdot {_fmt(c0_n)}\,p_{{0,\mathrm{{N}}}}
\;+ {_fmt(xn_phase1)}\,p_{{1,\mathrm{{N}}}}
\]
\[
x_{{\mathrm{{S}}}}
= e^{{-{lam_str}(1-p_{{1,\mathrm{{S}}}})}} \cdot {_fmt(c0_s)}\,p_{{0,\mathrm{{S}}}}
\;+ {_fmt(xs_phase1)}\,p_{{1,\mathrm{{S}}}}
\]
\[
\hat y
= {_fmt(intercept)}
- {_fmt(coef_n)}\,\log\!\bigl(1 + {_fmt(alpha)}\,x_{{\mathrm{{N}}}}\bigr)
- {_fmt(coef_s)}\,\log\!\bigl(1 + {_fmt(alpha)}\,x_{{\mathrm{{S}}}}\bigr)
+ {penalty_str}\,P(x_{{\mathrm{{N}}}}, x_{{\mathrm{{S}}}})
\]
\[
P(x_{{\mathrm{{N}}}}, x_{{\mathrm{{S}}}})
= \operatorname{{softplus}}\!\bigl(\log(1+x_{{\mathrm{{N}}}})-{_fmt(tau)}\bigr)^2
+ \operatorname{{softplus}}\!\bigl(\log(1+x_{{\mathrm{{S}}}})-{_fmt(tau)}\bigr)^2
\]
\[
{with_line}
\]
\[
{slice_line}
\]
\[
x_{{\mathrm{{N}}}}(z)
= {_fmt(c0_n)}\,e^{{-{lam_str} z}} + {_fmt(xn_phase1)}(1-z),
\qquad
x_{{\mathrm{{S}}}}(z) = {_fmt(xs_phase1)}\,z
\]
\[
\hat y(z)
= {_fmt(intercept)}
- {_fmt(coef_n)}\,\log\!\bigl(1 + {_fmt(alpha)}\,x_{{\mathrm{{N}}}}(z)\bigr)
- {_fmt(coef_s)}\,\log\!\bigl(1 + {_fmt(alpha)}\,x_{{\mathrm{{S}}}}(z)\bigr)
+ {penalty_str}\,P(x_{{\mathrm{{N}}}}(z), x_{{\mathrm{{S}}}}(z))
\]
""".strip()
    return block


def _slide_block(
    *,
    title: str,
    x_n: str,
    x_s: str,
    y_hat: str,
    penalty: str | None,
    interpretation: str,
) -> str:
    lines = [
        rf"\paragraph{{{title}}}",
        r"\[",
        x_n,
        r"\qquad",
        x_s,
        r"\]",
        r"\[",
        y_hat,
        r"\]",
    ]
    if penalty is not None:
        lines.extend(
            [
                r"\[",
                penalty,
                r"\]",
            ]
        )
    lines.append(rf"\emph{{{interpretation}}}")
    return "\n".join(lines)


def main() -> None:
    packet = load_completed_two_phase_starcoder_packet()
    slice_mask = packet.frame["phase_0_nemotron_full"].round(4).eq(1.0).to_numpy(dtype=bool)
    subset = subset_packet(packet, slice_mask)

    subset_params, subset_model = fit_starcoder_grp(subset, seed=0)
    all_params, all_model = fit_starcoder_grp(packet, seed=0)

    c0_n, c0_s = [float(x) for x in packet.c0]
    c1_n, c1_s = [float(x) for x in packet.c1]
    subset_coef_n, subset_coef_s, subset_coef_penalty = [float(x) for x in subset_model.coef_]
    all_coef_n, all_coef_s, all_coef_penalty = [float(x) for x in all_model.coef_]

    latex = "\n\n".join(
        [
            "% Fitted GRP forms for the 2-phase StarCoder packet.",
            "% Domains: N = Nemotron (broad-text), S = StarCoder (tech-code).",
            (
                rf"% Epoch multipliers: c_{{0,\mathrm{{N}}}}={_fmt(c0_n)}, "
                rf"c_{{0,\mathrm{{S}}}}={_fmt(c0_s)}, "
                rf"c_{{1,\mathrm{{N}}}}={_fmt(c1_n)}, "
                rf"c_{{1,\mathrm{{S}}}}={_fmt(c1_s)}."
            ),
            _model_block(
                title="Fit on U-shaped subset only",
                alpha=float(subset_params["alpha"]),
                eta=float(subset_params["eta"]),
                lam=float(subset_params["lam"]),
                tau=float(subset_params["tau"]),
                reg=float(subset_params["reg"]),
                intercept=float(subset_model.intercept_),
                coef_n=subset_coef_n,
                coef_s=subset_coef_s,
                coef_penalty=subset_coef_penalty,
                c0_n=c0_n,
                c0_s=c0_s,
                c1_n=c1_n,
                c1_s=c1_s,
            ),
            _model_block(
                title="Fit on all 2-phase StarCoder data",
                alpha=float(all_params["alpha"]),
                eta=float(all_params["eta"]),
                lam=float(all_params["lam"]),
                tau=float(all_params["tau"]),
                reg=float(all_params["reg"]),
                intercept=float(all_model.intercept_),
                coef_n=all_coef_n,
                coef_s=all_coef_s,
                coef_penalty=all_coef_penalty,
                c0_n=c0_n,
                c0_s=c0_s,
                c1_n=c1_n,
                c1_s=c1_s,
            ),
        ]
    )

    summary = {
        "c0": {"nemotron": c0_n, "starcoder": c0_s},
        "c1": {"nemotron": c1_n, "starcoder": c1_s},
        "subset_fit": {
            "params": subset_params,
            "intercept": float(subset_model.intercept_),
            "coef": {
                "nemotron_signal": subset_coef_n,
                "starcoder_signal": subset_coef_s,
                "penalty": subset_coef_penalty,
            },
        },
        "all_data_fit": {
            "params": all_params,
            "intercept": float(all_model.intercept_),
            "coef": {
                "nemotron_signal": all_coef_n,
                "starcoder_signal": all_coef_s,
                "penalty": all_coef_penalty,
            },
        },
    }

    slide_tex = "\n\n".join(
        [
            r"% Slide-ready GRP forms for the 2-phase StarCoder packet.",
            (
                r"% On the U-shaped slice: "
                r"z = p_{1,\mathrm{S}},\; p_{1,\mathrm{N}} = 1-z,\; "
                r"p_{0,\mathrm{N}} = 1,\; p_{0,\mathrm{S}} = 0."
            ),
            _slide_block(
                title="Fit on U-shaped subset only",
                x_n=r"x_{\mathrm{N}}(z) \approx 0.50 + 5.73(1-z)",
                x_s=r"x_{\mathrm{S}}(z) = 151.69\,z",
                y_hat=(
                    r"\hat y(z) \approx 3.17"
                    r" - 0.54\,\log\!\bigl(1 + 3.22\,x_{\mathrm{N}}(z)\bigr)"
                    r" - 0.16\,\log\!\bigl(1 + 3.22\,x_{\mathrm{S}}(z)\bigr)"
                ),
                penalty=None,
                interpretation=(
                    r"On the U-shaped slice, this is essentially a two-channel "
                    r"saturated total-exposure law with negligible retention and "
                    r"negligible overexposure penalty."
                ),
            ),
            _slide_block(
                title="Fit on all 2-phase StarCoder data",
                x_n=(r"x_{\mathrm{N}}(z)" r" \approx 0.50\,e^{-2.51 z} + 0.49(1-z)"),
                x_s=r"x_{\mathrm{S}}(z) = 12.89\,z",
                y_hat=(
                    r"\hat y(z) \approx 2.42"
                    r" - 0.42\,\log\!\bigl(1 + 8.31\,x_{\mathrm{N}}(z)\bigr)"
                    r" - 0.26\,\log\!\bigl(1 + 8.31\,x_{\mathrm{S}}(z)\bigr)"
                    r" + 2.80\,P\bigl(x_{\mathrm{N}}(z), x_{\mathrm{S}}(z)\bigr)"
                ),
                penalty=(
                    r"P(x_{\mathrm{N}}, x_{\mathrm{S}})"
                    r" = \operatorname{softplus}\!\bigl(\log(1+x_{\mathrm{N}})-3.22\bigr)^2"
                    r" + \operatorname{softplus}\!\bigl(\log(1+x_{\mathrm{S}})-3.22\bigr)^2"
                ),
                interpretation=(
                    r"On the full packet, the fit is still simple, but now it "
                    r"learns meaningful retention on Nemotron and a real "
                    r"overexposure penalty."
                ),
            ),
        ]
    )

    OUTPUT_TEX.write_text(latex + "\n")
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))
    OUTPUT_SLIDE_TEX.write_text(slide_tex + "\n")

    print(latex)
    print("\n--- Slide-ready version ---\n")
    print(slide_tex)
    print(f"\nWrote {OUTPUT_TEX}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_SLIDE_TEX}")


if __name__ == "__main__":
    main()
