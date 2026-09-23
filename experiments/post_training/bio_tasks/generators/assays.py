# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mass spectrometry, isotope tracing, plate assays, and qPCR."""

import math
import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def generate_assays(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation.startswith("mgf"):
        blocks = []
        ions = []
        for i in range(3):
            charge = i + 1
            mass = 600 + i * 150 + rng.randint(1, 20)
            mz = mass / charge + 1.007276466621
            a, b = rng.randint(10, 30), rng.randint(10, 30)
            peaks = [(100.0, a), (200.0, b), (300.0, a + b + 1), (400.0, 0)]
            if operation == "mgf-fragment-matching":
                first_ppm, second_ppm = rng.randint(2, 4), -rng.randint(3, 8)
                first_mz, second_mz = 500 * (1 + first_ppm / 1e6), 1000 * (1 + second_ppm / 1e6)
                peaks = [(first_mz, a), (500.0045, b), (second_mz, a + b + 1), (1500.033, 10)]
                for ion, theoretical, observed, error in [
                    ("b1", 500, first_mz, float(first_ppm)),
                    ("y2", 1000, second_mz, float(second_ppm)),
                    ("y3", 1500, None, None),
                ]:
                    ions.append({"spectrum": f"s{i}", "ion": ion, "mz": theoretical})
                    expected[f"s{i}:{ion}"] = {
                        "matched": int(observed is not None),
                        "observed_mz": observed,
                        "error_ppm": error,
                    }
            elif operation == "mgf-precursor-neutral-mass":
                expected[f"s{i}"] = {"neutral_mass": float(mass), "charge": charge}
            else:
                expected[f"s{i}"] = {"tic": 2 * (a + b) + 1, "base_mz": 300.0, "base_intensity": a + b + 1, "peaks": 4}
            rng.shuffle(peaks)
            blocks.append(
                f"BEGIN IONS\nTITLE=s{i}\nPEPMASS={mz:.9f} 99999\nCHARGE={charge}+\nRTINSECONDS={60+i}\n"
                + "".join(f"{mass_value:.6f} {intensity}\n" for mass_value, intensity in peaks)
                + "END IONS\n"
            )
        inputs = {"spectra.mgf": "\n".join(blocks)}
        if operation == "mgf-precursor-neutral-mass":
            columns = {
                "neutral_mass": Column(
                    kind="number", unit="Da", description="z*(precursor m/z - proton mass)", atol=1e-6, rtol=1e-8
                ),
                "charge": Column(
                    kind="integer", unit="positive charges", description="single declared precursor charge"
                ),
            }
            prompt = (
                "For each TITLE in spectra.mgf, compute neutral precursor monoisotopic mass from its "
                "positive CHARGE and first PEPMASS value (m/z): M=z*(m/z-1.007276466621 Da). The second "
                "PEPMASS value is intensity, not mass. Each block has one unambiguous positive charge. "
                "Report charge and neutral mass."
            )
            wrong = [
                {"id": k, **v, "neutral_mass": v["neutral_mass"] + 1.007276466621 * v["charge"]}
                for k, v in expected.items()
            ]
            reason = "forgot_proton_mass"
        elif operation == "mgf-fragment-matching":
            inputs["ions.csv"] = csv_text(ions)
            columns = {
                "matched": Column(
                    kind="integer", unit="decision", description="1 if an observed fragment lies within 10 ppm"
                ),
                "observed_mz": Column(
                    kind="number",
                    unit="m/z",
                    description="closest observed fragment; null if unmatched",
                    nullable=True,
                    atol=1e-7,
                    rtol=1e-9,
                ),
                "error_ppm": Column(
                    kind="number",
                    unit="ppm",
                    description="1e6*(observed-theoretical)/theoretical; null if unmatched",
                    nullable=True,
                    atol=1e-6,
                    rtol=1e-8,
                ),
            }
            prompt = (
                "Match each ions.csv theoretical fragment to peaks in its spectra.mgf TITLE block within "
                "inclusive 10 ppm tolerance, using theoretical m/z as the ppm denominator. Choose the "
                "smallest absolute ppm error (then lower observed m/z on ties). Match ions independently, "
                "retain unmatched ions with matched=0 and null numeric fields. Preserve signed error. Use "
                "spectrum:ion id."
            )
            wrong = [
                {"id": k, **v, "error_ppm": abs(v["error_ppm"]) if v["error_ppm"] is not None else None}
                for k, v in expected.items()
            ]
            reason = "discarded_mass_error_sign"
        else:
            columns = {
                "tic": Column(kind="integer", unit="intensity units", description="sum of listed fragment intensities"),
                "base_mz": Column(
                    kind="number", unit="m/z", description="m/z of maximum-intensity fragment", atol=1e-8, rtol=1e-8
                ),
                "base_intensity": Column(
                    kind="integer", unit="intensity units", description="maximum fragment intensity"
                ),
                "peaks": Column(
                    kind="integer", unit="peaks", description="listed fragment rows including zero intensity"
                ),
            }
            prompt = (
                "Summarize each spectra.mgf TITLE block: total ion current from listed fragment "
                "intensities, base-peak m/z and intensity, and number of listed fragment peaks (including "
                "zero-intensity rows). Exclude PEPMASS precursor intensity. Break base-peak ties by lower "
                "m/z."
            )
            wrong = [{"id": k, **v, "tic": v["tic"] + 99999} for k, v in expected.items()]
            reason = "included_precursor_intensity_in_fragment_tic"
    elif operation == "metabolite-isotope-correction":
        p = 0.1
        matrix = [[(1 - p) ** 2, 0, 0], [2 * p * (1 - p), 1 - p, 0], [p * p, p, 1]]
        weights = [rng.randint(2, 8), rng.randint(2, 5), rng.randint(1, 4)]
        fractions = [value / sum(weights) for value in weights]
        observations = [sum(value * fraction for value, fraction in zip(row, fractions, strict=True)) for row in matrix]
        inputs = {
            "correction.csv": csv_text(
                [{"observed": i, "true": j, "coefficient": matrix[i][j]} for i in range(3) for j in range(3)]
            ),
            "observed.csv": csv_text(
                [{"isotopologue": f"M+{i}", "fraction": value} for i, value in enumerate(observations)]
            ),
        }
        expected = {f"M+{i}": {"corrected_fraction": value} for i, value in enumerate(fractions)}
        columns = {
            "corrected_fraction": Column(
                kind="number", unit="fraction", description="solution x of observed=A*x", atol=1e-9, rtol=1e-8
            )
        }
        prompt = (
            "Correct the three observed isotopologue fractions in observed.csv using correction.csv. "
            "Matrix rows are observed mass increments and columns true labeled-carbon counts; "
            "observed=A*true. Solve this full linear system. Do not transpose the matrix, subtract a "
            "constant, or renormalize the input before correction. Report corrected fractions with M+0,"
            " M+1, M+2 IDs. This supplied calibration matrix defines the model."
        )
        wrong = [{"id": f"M+{i}", "corrected_fraction": value} for i, value in enumerate(observations)]
        reason = "reported_uncorrected_isotope_fractions"
    elif operation == "dose-response-ic50":
        scale = rng.choice([0.1, 1, 10])
        rows = []
        for name, responses in [("cross", [90, 70, 30, 10]), ("outside", [90, 80, 70, 60]), ("exact", [90, 70, 50, 20])]:
            for concentration, response in zip([0.1, 1, 10, 100], responses, strict=True):
                rows.append({"compound": name, "concentration_uM": concentration * scale, "viability_percent": response})
        rng.shuffle(rows)
        inputs = {"dose_response.csv": csv_text(rows)}
        expected = {
            "cross": {"ic50": math.sqrt(10) * scale, "status": "bracketed"},
            "outside": {"ic50": None, "status": "not_bracketed"},
            "exact": {"ic50": 10.0 * scale, "status": "observed"},
        }
        columns = {
            "ic50": Column(
                kind="number",
                unit="micromolar",
                description="50% viability crossing on log10 concentration axis",
                nullable=True,
                atol=1e-8,
                rtol=1e-8,
            ),
            "status": Column(kind="text", unit="classification", description="observed, bracketed, or not_bracketed"),
        }
        prompt = (
            "Estimate each compound IC50 as the 50% viability crossing in dose_response.csv. Use an "
            "exact observed 50% point if present; otherwise interpolate viability linearly against "
            "log10 concentration between adjacent bracketing points. Do not fit a Hill model or "
            "extrapolate: absent crossing gives null and not_bracketed. Other statuses are observed or "
            "bracketed. Use compound id."
        )
        wrong = [{"id": k, **v, "ic50": 5.5 * scale if k == "cross" else v["ic50"]} for k, v in expected.items()]
        reason = "interpolated_on_linear_concentration_axis"
    elif operation == "plate-control-normalization":
        rows = []
        for plate in range(2):
            background = rng.randint(5, 20)
            span = rng.randint(20, 50)
            low_percent, high_percent = rng.randint(10, 60), rng.randint(110, 150)
            for well, kind, value in [
                ("A1", "negative", background - 2),
                ("A2", "negative", background + 2),
                ("B1", "positive", background + span - 2),
                ("B2", "positive", background + span + 2),
                ("C1", "sample", background + span * low_percent / 100),
                ("C2", "sample", background + span * high_percent / 100),
            ]:
                rows.append({"plate": f"p{plate}", "well": well, "role": kind, "signal": value})
            expected[f"p{plate}:C1"] = {"percent_response": float(low_percent)}
            expected[f"p{plate}:C2"] = {"percent_response": float(high_percent)}
        rng.shuffle(rows)
        inputs = {"plate.csv": csv_text(rows)}
        columns = {
            "percent_response": Column(
                kind="number",
                unit="percent",
                description="100*(signal-mean negative)/(mean positive-mean negative)",
                atol=1e-8,
                rtol=1e-8,
            )
        }
        prompt = (
            "Normalize sample wells in plate.csv to their own plate controls: 100*(signal-mean "
            "negative)/(mean positive-mean negative). Average control replicates separately per plate. "
            "Report sample wells only, using plate:well id. Preserve values outside 0..100 rather than "
            "clipping."
        )
        wrong = [{"id": k, "percent_response": min(100, v["percent_response"])} for k, v in expected.items()]
        reason = "clipped_super_control_response"
    else:
        assert operation == "qpcr-delta-delta-ct"
        rows = []
        control_delta = rng.randint(3, 6)
        for sample, delta in [
            ("control", control_delta),
            ("treated1", control_delta - 2),
            ("treated2", control_delta + 1),
        ]:
            reference = rng.randint(18, 22)
            for gene, mean in [("reference", reference), ("target", reference + delta)]:
                for i, noise in enumerate([-0.2, 0.2]):
                    rows.append({"sample": sample, "gene": gene, "replicate": i, "ct": mean + noise})
            ddct = delta - control_delta
            expected[sample] = {"delta_ct": float(delta), "delta_delta_ct": float(ddct), "fold_change": 2.0 ** (-ddct)}
        rng.shuffle(rows)
        inputs = {"ct.csv": csv_text(rows)}
        columns = {
            name: Column(
                kind="number",
                unit="fold" if name == "fold_change" else "cycles",
                description=description,
                atol=1e-8,
                rtol=1e-8,
            )
            for name, description in [
                ("delta_ct", "mean target Ct minus mean reference Ct"),
                ("delta_delta_ct", "sample deltaCt minus control deltaCt"),
                ("fold_change", "2 raised to negative deltaDeltaCt"),
            ]
        }
        prompt = (
            "Analyze ct.csv by the delta-delta-Ct method, assuming equal 100% amplification "
            "efficiencies. Average technical Ct replicates within sample/gene before subtraction. "
            "deltaCt=target-reference; calibrator sample is control; deltaDeltaCt=sample "
            "deltaCt-control deltaCt; fold=2^(-deltaDeltaCt). Include the control row and use sample "
            "id."
        )
        wrong = [{"id": k, **v, "fold_change": 1 / v["fold_change"]} for k, v in expected.items()]
        reason = "reversed_delta_delta_ct_sign"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "mgf-precursor-neutral-mass": ("precursor-charge", "proton-mass", "mgf-metadata"),
    "mgf-fragment-matching": ("ppm-tolerance", "signed-mass-error", "peak-matching"),
    "mgf-total-ion-current": ("fragment-intensities", "base-peak", "precursor-exclusion"),
    "metabolite-isotope-correction": ("isotopologues", "matrix-orientation", "calibration"),
    "dose-response-ic50": ("log-concentration", "interpolation", "no-extrapolation"),
    "plate-control-normalization": ("plate-specific-controls", "replicates", "unclipped-response"),
    "qpcr-delta-delta-ct": ("technical-replicates", "reference-gene", "fold-direction"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        ("mascot-generic-format", "csv-header") if name.startswith("mgf") else ("csv-header",),
        (
            ("https://www.matrixscience.com/help/data_file_help.html",)
            if name.startswith("mgf")
            else ("https://www.ncbi.nlm.nih.gov/books/NBK53196/",)
        ),
        partial(generate_assays, operation=name),
    )
    for name, skills in SKILLS.items()
)
