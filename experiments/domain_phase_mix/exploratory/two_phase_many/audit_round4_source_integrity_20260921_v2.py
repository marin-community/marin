# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Verify final round-four provenance without evaluating or fitting outcomes."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import nextgen_two_phase_round4_20260921_v2 as candidate

CODE = Path(__file__).resolve().parent
ROOT = candidate.ROOT


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    """Preserve an existing identical snapshot and reject any overwrite."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Append-only snapshot differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def without_amendment(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            indices = [
                i for i, key in enumerate(node.keys) if isinstance(key, ast.Constant) and key.value == "audit_amendment"
            ]
            for index in reversed(indices):
                del node.keys[index]
                del node.values[index]
    return ast.dump(tree)


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def check_hashes(name: str, values: dict[str, str], base: Path | None = None) -> None:
        mismatches = []
        for filename, expected in values.items():
            path = Path(filename) if base is None else base / filename
            actual = digest(path) if path.exists() else None
            if actual != expected:
                mismatches.append({"path": str(path), "expected": expected, "actual": actual})
        check(name, not mismatches, {"files": len(values), "mismatches": mismatches})

    freezes = [ROOT / "source_freeze.json", ROOT / "numerical_v2/source_freeze.json"]
    initial_freeze_hashes = {str(path): digest(path) for path in freezes}
    for path in freezes:
        record = json.loads(path.read_text())
        check_hashes(f"{path.parent.name} frozen sources", record["sources"])
        check_hashes(f"{path.parent.name} frozen drivers", record["driver_sources"])
    freeze = json.loads(freezes[1].read_text())
    sources = candidate.source_hashes()
    check("live v2 fitting sources equal frozen sources", sources == freeze["sources"], {"source_count": len(sources)})
    environment = candidate.benchmark.environment_identity()
    check("pinned live environment equals v2 freeze", environment == freeze["environment"], environment)
    sc_paths = sorted((ROOT / "starcoder_screen_v2/shards").glob("*/complete.json"))
    synthetic_paths = sorted((ROOT / "synthetic_v2").glob("*/*/*/*/*/complete.json"))
    sc_cells, synthetic_cells = [], []
    source_mismatch, environment_mismatch = [], []
    for path in sc_paths:
        complete = json.loads(path.read_text())
        identity = complete["identity"]
        sc_cells.append((identity["model"], identity["target"], identity["outer_fold"]))
        if identity["sources"] != sources:
            source_mismatch.append(str(path))
        if identity["environment"] != environment:
            environment_mismatch.append(str(path))
        check_hashes(f"SC outputs {path.parent.name}", complete["outputs"], path.parent)
    prepared = candidate.benchmark.DEFAULT_OUTPUT / "v3/prepared"
    metadata_path = prepared / "starcoder_wsd80_dense/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    target_map = {
        selector: sorted(
            target for target in metadata["targets"] if candidate.benchmark.target_matches(target, selector)
        )
        for selector in candidate.benchmark.QUICK_TASKS
    }
    check(
        "registered task aliases resolve uniquely", all(len(values) == 1 for values in target_map.values()), target_map
    )
    expected_targets = [target for values in target_map.values() for target in values]
    expected_sc = set(itertools.product(candidate.MODELS, expected_targets, range(5)))
    check(
        "75 SC fits cover exactly all registered cells",
        len(sc_cells) == 75 and set(sc_cells) == expected_sc,
        {"count": len(sc_cells), "unique": len(set(sc_cells)), "expected": len(expected_sc)},
    )
    for path in synthetic_paths:
        complete = json.loads(path.read_text())
        config = complete["config"]
        synthetic_cells.append((config["model"], config["world"], config["sampler"], config["noise_sd"], config["seed"]))
        if any(config["sources"].get(name) != value for name, value in sources.items()):
            source_mismatch.append(str(path))
        if config["environment"] != environment:
            environment_mismatch.append(str(path))
        check_hashes(f"synthetic sources {path.parent}", config["sources"])
        check_hashes(f"synthetic outputs {path.parent}", complete["outputs"], path.parent)
        check(
            f"synthetic frozen revision {path.parent}",
            config["source_freeze"]["sha256"] == initial_freeze_hashes[str(freezes[1])],
            config["source_freeze"],
        )
    expected_synthetic = {
        (model, world, sampler, noise, seed)
        for model, world, sampler in itertools.product(
            candidate.MODELS, ("null", "boundary", "misspecified_null"), ("balanced", "positive")
        )
        for noise, seed in [(0.0, 20260921), *((0.002, seed) for seed in (20260921, 20260922, 20260923))]
    }
    check(
        "72 synthetic fits cover exactly all registered cells",
        len(synthetic_cells) == 72 and set(synthetic_cells) == expected_synthetic,
        {"count": len(synthetic_cells), "unique": len(set(synthetic_cells)), "expected": len(expected_synthetic)},
    )
    check("all147 per-fit scientific sources match v2", not source_mismatch, source_mismatch)
    check("all147 per-fit environments match pinned v2", not environment_mismatch, environment_mismatch)
    mapping_path = ROOT / "numerical_v2/structural_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    check(
        "mapping scientific sources equal frozen v2",
        mapping["sources"] == sources,
        {"source_count": len(mapping["sources"])},
    )
    check_hashes("mapping audit sources", mapping["audit_sources"])
    old_driver = CODE / "audit_round4_mapping_geometry_20260921_v2.py"
    final_driver = CODE / "audit_round4_mapping_geometry_20260921_v3.py"
    imports = [
        alias.name
        for node in ast.walk(ast.parse(final_driver.read_text()))
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    check(
        "mapping v3 imports v2 fitting runner",
        "nextgen_two_phase_round4_20260921_v2" in imports and "nextgen_two_phase_round4_20260921" not in imports,
        imports,
    )
    corrected_source = old_driver.read_text().replace(
        '(OUTPUT / "starcoder_screen_v1/shards")', '(candidate.ROOT / "starcoder_screen_v2/shards")'
    )
    check(
        "only mapping path and amendment metadata changed",
        without_amendment(corrected_source) == without_amendment(final_driver.read_text()),
        mapping["audit_amendment"],
    )
    amendment = mapping["audit_amendment"]
    check(
        "mapping amendment names verified predecessor",
        amendment["previous_driver"]["sha256"] == digest(old_driver),
        amendment,
    )
    check(
        "mapping records75heads90generics20ties",
        len(mapping["learned_heads"]) == 75 and len(mapping["generic"]) == 90 and len(mapping["query_row_ids"]) == 20,
        {"models": mapping["model_status"], "status": mapping["status"]},
    )
    mapping_paths = {row["fit"]["path"] for row in mapping["learned_heads"]}
    expected_paths = {str((path.parent / "fit.json").resolve()) for path in sc_paths}
    check("mapping covers exact final SC fit files", mapping_paths == expected_paths, {"count": len(mapping_paths)})
    check_hashes("mapping fit receipts", {row["fit"]["path"]: row["fit"]["sha256"] for row in mapping["learned_heads"]})
    snapshot = ROOT / "final_audit_sources"
    snapshot.mkdir(exist_ok=True)
    paths = [
        final_driver,
        Path(__file__),
        ROOT / "reviews/math_checks.py",
        ROOT / "reviews/math_disposition.md",
        ROOT / "reviews/geometry_math_checks.json",
        ROOT / "reviews/review.md",
        ROOT / "source_integrity_audit_v1_failure.json",
        CODE / "audit_round4_estimator_oracle_20260921.py",
        ROOT / "numerical_v2/readout_diagnostic.py",
        ROOT / "oracle_attribution/gamma_transport_diagnostic.json",
    ]
    files = []
    for path in paths:
        target = snapshot / path.name
        write_once(target, path.read_bytes())
        files.append({"source": str(path.resolve()), "snapshot": str(target.resolve()), "sha256": digest(target)})
    manifest = {
        "scope": "Append-only final audit sources; no original source freeze is changed",
        "previous_manifest": candidate.benchmark.receipt(snapshot / "manifest.json"),
        "files": files,
    }
    write_once(snapshot / "manifest_v2.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    check(
        "original freezes unchanged during audit",
        all(digest(Path(path)) == value for path, value in initial_freeze_hashes.items()),
        initial_freeze_hashes,
    )
    failures = [item for item in checks if not item["passed"]]
    report = {
        "status": "pass" if not failures else "fail",
        "scope": "Read-only provenance checks, no new fits or predictions",
        "command": f"uv run --no-sync python {Path(__file__).relative_to(Path.cwd())}",
        "source": candidate.benchmark.receipt(Path(__file__)),
        "snapshot_manifest": candidate.benchmark.receipt(snapshot / "manifest_v2.json"),
        "audit_correction": {
            "previous_receipt": candidate.benchmark.receipt(ROOT / "source_integrity_audit_v1_failure.json"),
            "reason": (
                "The first audit compared canonical fit targets with short benchmark aliases; "
                "this audit resolves the aliases against frozen metadata using benchmark.target_matches."
            ),
            "metadata": candidate.benchmark.receipt(metadata_path),
            "scientific_sources_changed": False,
        },
        "checks": checks,
        "discrepancies": failures,
        "fit_counts": {"SC": len(sc_paths), "synthetic": len(synthetic_paths)},
    }
    candidate.benchmark.write_json(ROOT / "source_integrity.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": len(checks),
                "discrepancies": failures,
                "fit_counts": report["fit_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
