# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Spectral parsing, calibration inversion, and assay calculations."""

import math
from collections import defaultdict
from functools import partial
from itertools import pairwise
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import table
from experiments.post_training.bio_tasks.solvers.statistics import reduced_rows


def mgf(path: Path) -> list[tuple[dict, list[tuple[float, float]]]]:
    spectra = []
    for line in path.read_text().splitlines():
        if line == "BEGIN IONS":
            metadata, peaks = {}, []
        elif line == "END IONS":
            spectra.append((metadata, peaks))
        elif "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
        elif line.strip():
            x, y = line.split()
            peaks.append((float(x), float(y)))
    return spectra


def solve_assays(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation.startswith("mgf"):
        spectra = mgf(inputs / "spectra.mgf")
        for metadata, peaks in spectra:
            name = metadata["TITLE"]
            if operation == "mgf-precursor-neutral-mass":
                charge = int(metadata["CHARGE"].rstrip("+"))
                mass = charge * float(metadata["PEPMASS"].split()[0]) - charge * 1.007276466621
                answer.append({"id": name, "neutral_mass": mass, "charge": charge})
            elif operation == "mgf-total-ion-current":
                mass, intensity = max(peaks, key=lambda pair: (pair[1], -pair[0]))
                answer.append(
                    {
                        "id": name,
                        "tic": int(sum(value for _, value in peaks)),
                        "base_mz": mass,
                        "base_intensity": int(intensity),
                        "peaks": len(peaks),
                    }
                )
            else:
                for row in table(inputs / "ions.csv"):
                    if row["spectrum"] != name:
                        continue
                    theoretical = float(row["mz"])
                    closest = min((mass for mass, _ in peaks), key=lambda mass: (abs(mass - theoretical), mass))
                    error = (closest - theoretical) / theoretical * 1e6
                    matched = abs(error) <= 10
                    answer.append(
                        {
                            "id": name + ":" + row["ion"],
                            "matched": int(matched),
                            "observed_mz": closest if matched else None,
                            "error_ppm": error if matched else None,
                        }
                    )
    elif operation == "metabolite-isotope-correction":
        entries = table(inputs / "correction.csv")
        observed = table(inputs / "observed.csv")
        matrix = [[0.0] * 4 for _ in range(3)]
        for row in entries:
            matrix[int(row["observed"])][int(row["true"])] = row["coefficient"]
        for row in observed:
            matrix[int(row["isotopologue"].split("+")[1])][-1] = row["fraction"]
        answer = [{"id": f"M+{i}", "corrected_fraction": float(row[-1])} for i, row in enumerate(reduced_rows(matrix))]
    elif operation == "dose-response-ic50":
        compounds = defaultdict(list)
        for row in table(inputs / "dose_response.csv"):
            compounds[row["compound"]].append((float(row["concentration_uM"]), float(row["viability_percent"])))
        for name, points in compounds.items():
            points.sort()
            exact = [x for x, y in points if y == 50]
            result = {"id": name, "ic50": None, "status": "not_bracketed"}
            if exact:
                result.update(ic50=exact[0], status="observed")
            else:
                for (x1, y1), (x2, y2) in pairwise(points):
                    if (y1 - 50) * (y2 - 50) < 0:
                        logx = math.log10(x1) + (50 - y1) / (y2 - y1) * (math.log10(x2) - math.log10(x1))
                        result.update(ic50=10**logx, status="bracketed")
                        break
            answer.append(result)
    elif operation == "plate-control-normalization":
        rows = table(inputs / "plate.csv")
        controls = defaultdict(list)
        for row in rows:
            if row["role"] != "sample":
                controls[row["plate"], row["role"]].append(float(row["signal"]))
        means = {key: sum(values) / len(values) for key, values in controls.items()}
        for row in rows:
            if row["role"] == "sample":
                lo, hi = means[row["plate"], "negative"], means[row["plate"], "positive"]
                answer.append(
                    {
                        "id": row["plate"] + ":" + row["well"],
                        "percent_response": 100 * (float(row["signal"]) - lo) / (hi - lo),
                    }
                )
    else:
        groups = defaultdict(list)
        for row in table(inputs / "ct.csv"):
            groups[row["sample"], row["gene"]].append(float(row["ct"]))
        means = {key: sum(values) / len(values) for key, values in groups.items()}
        delta = {sample: means[sample, "target"] - means[sample, "reference"] for sample, _ in groups}
        for name, value in delta.items():
            difference = value - delta["control"]
            answer.append(
                {"id": name, "delta_ct": value, "delta_delta_ct": difference, "fold_change": 2 ** (-difference)}
            )
    return answer


NAMES = (
    "mgf-precursor-neutral-mass",
    "mgf-fragment-matching",
    "mgf-total-ion-current",
    "metabolite-isotope-correction",
    "dose-response-ic50",
    "plate-control-normalization",
    "qpcr-delta-delta-ct",
)
SOLVERS = {name: partial(solve_assays, operation=name) for name in NAMES}
