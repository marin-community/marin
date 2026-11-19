# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score frozen complete-document BME windows with the deployed junkgate model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.cluster.quality.fast_transformer.calibrate import load_calibration
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_windowed

MODEL_DIR = "s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2_marin"
MODEL_FILES = ("pooled_junkgate2.eqx", "pooled_junkgate2_meta.json", "pooled_junkgate2_remap.json", "calib_bme.json")


def model_digests(model_dir: str | None = None) -> dict[str, str]:
    if model_dir is None:
        model_dir = MODEL_DIR
    digests = {}
    for name in MODEL_FILES:
        with (StoragePath(model_dir) / name).open("rb") as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    hashes = model_digests()
    scorer = load_pooled_scorer(MODEL_DIR)
    calibration = load_calibration(MODEL_DIR)
    samples = [json.loads(path.read_text()) for path in sorted(args.sample.glob("sample-*.json"))]
    rows = []
    for start in range(0, len(samples), 64):
        batch = samples[start : start + 64]
        windows = [[np.asarray(window, dtype=np.int32) for window in row["bme_token_ids"]] for row in batch]
        raw = score_windowed(scorer, windows)
        calibrated = calibration.apply(raw, types=None)
        rows.extend(
            {"sample_id": sample["sample_id"], "raw_score": float(raw_score), "score": float(score)}
            for sample, raw_score, score in zip(batch, raw, calibrated, strict=True)
        )
    receipt = {
        "model": MODEL_DIR,
        "model_sha256": hashes,
        "model_tokenizer": scorer.tokenizer_name,
        "method": "mean of full-document begin/middle/end 512-token windows, then saved calibration",
        "scores": rows,
    }
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"scored": len(rows), "tokenizer": scorer.tokenizer_name}))


if __name__ == "__main__":
    main()
