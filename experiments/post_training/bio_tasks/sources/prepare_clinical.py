# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyreadr==0.5.6", "statsmodels==0.14.6"]
# ///

"""Preserve published PBC observations and independently fit fixed Cox references."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pyreadr
from statsmodels.duration.hazard_regression import PHReg

SOURCE_ID = "survival:PBC"


def prepare(source: Path, revision: str, output: Path) -> dict:
    """Convert the source R data without imputing or changing clinical observations."""
    tables = pyreadr.read_r(str(source))
    assert {name: frame.shape for name, frame in tables.items()} == {"pbc": (418, 20), "pbcseq": (1945, 19)}
    assert not tables["pbc"].duplicated("id").any()
    assert not tables["pbcseq"].duplicated(["id", "day"]).any()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assets = {}
    raw_url = f"https://raw.githubusercontent.com/therneau/survival/{revision}/data/pbc.rda"
    for name, frame in tables.items():
        content = frame.to_csv(sep="\t", index=False, na_rep="", float_format="%.17g").encode()
        filename = f"pbc-{name}.tsv.gz"
        compressed = gzip.compress(content, mtime=0)
        (output / filename).write_bytes(compressed)
        assets[filename] = {
            "url": raw_url,
            "download_sha256": source_hash,
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
            "content_bytes": len(content),
            "vendored_bytes": len(compressed),
            "transformation": (
                "R dataframe converted to TSV; column values and row order preserved, missing values empty."
            ),
        }
    models = {}
    frame = tables["pbc"]
    for treatment in ("all", "1", "2"):
        selected = frame[frame.trt.notna()]
        if treatment != "all":
            selected = selected[selected.trt.astype(float) == int(treatment)]
        for biomarker in ("bili", "albumin", "protime"):
            complete = selected.dropna(subset=["time", "status", "age", "sex", biomarker])
            marker = complete[biomarker].to_numpy(dtype=float)
            if biomarker != "albumin":
                marker = np.log(marker)
            design = np.column_stack(
                [marker, complete.age.to_numpy(dtype=float) / 10, (complete.sex == "m").astype(float)]
            )
            events = complete.status.to_numpy(dtype=int) != 0
            fit = PHReg(complete.time.to_numpy(dtype=float), design, status=events, ties="breslow").fit(
                maxiter=100, tol=1e-11
            )
            models[f"{treatment}/{biomarker}"] = {
                "patients": len(complete),
                "events": int(events.sum()),
                "coefficients": fit.params.tolist(),
                "standard_errors": fit.bse.tolist(),
                "pvalues": fit.pvalues.tolist(),
                "log_likelihood": float(fit.llf),
            }
    reference = {
        "method": "statsmodels PHReg, Breslow ties, composite death/transplant endpoint, age in decades, male indicator",
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("statsmodels", "pyreadr", "numpy", "pandas")
        },
        "models": models,
    }
    content = (json.dumps(reference, indent=2, allow_nan=False) + "\n").encode()
    filename = "pbc-cox-reference.json.gz"
    compressed = gzip.compress(content, mtime=0)
    (output / filename).write_bytes(compressed)
    assets[filename] = {
        "url": raw_url,
        "download_sha256": source_hash,
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
        "content_bytes": len(content),
        "vendored_bytes": len(compressed),
        "transformation": reference["method"],
        "preparation_versions": reference["package_versions"],
    }
    return {
        "landing_page": f"https://github.com/therneau/survival/tree/{revision}",
        "lineage": "Mayo-PBC-1974-1984",
        "revision": revision,
        "license": "LGPL-2.0-or-later",
        "license_source": f"https://github.com/therneau/survival/blob/{revision}/DESCRIPTION",
        "citation": (
            "Therneau and Grambsch (2000), Modeling Survival Data, ISBN 0-387-98784-3; "
            "Murtaugh et al. (1994), Hepatology 20:126-134."
        ),
        "retrieved_at": "2026-09-24",
        "data_origin": "real",
        "benchmark_screening": (
            "No matching study identifiers in inspected permitted ID metadata; full artifact-lineage screening pending. "
            "BioMysteryBench held out as OOD and not used for authoring."
        ),
        "limitations": (
            "pbcseq contains corrected baseline values and extended follow-up; "
            "do not interchange its baselines with pbc. Missing visits/measurements may be informative. "
            "Observational associations do not establish treatment benefit."
        ),
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    catalog["sources"][SOURCE_ID] = prepare(args.source, args.revision, args.output)
    args.catalog.write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
