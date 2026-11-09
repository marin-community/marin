# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare serving observations with independently scored, identical histories."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from marin.testing.inference.backend_parity import NextTokenObservation, TokenScore


def compare(reference_paths: list[Path], observations: Path, maximum_probability_error: float) -> list[dict]:
    expected = {}
    for directory in reference_paths:
        metadata = json.loads((directory / "scores.json").read_text())
        for row, values in zip(metadata["rows"], np.load(directory / "logprobs.npy"), strict=True):
            key = (row["case_id"], row["position"])
            assert key not in expected, key
            expected[key] = (row["history_sha256"], values)
    results = []
    for row in json.loads(observations.read_text()):
        history, values = expected[(row["case_id"], row["position"])]
        assert hashlib.sha256(np.asarray(row["token_ids"], dtype=np.int32).tobytes()).hexdigest() == history
        raw = row["logprobs"]
        actual = {
            int(token): float(score) for token, score in (raw.items() if isinstance(raw, dict) else enumerate(raw))
        }
        observation = NextTokenObservation.from_logprob_map(
            row["case_id"], row["output_token_id"], actual, backend_rank=row.get("rank", 0)
        )
        top = np.argsort(values)[-5:][::-1]
        parity = observation.parity_against(tuple(TokenScore(float(values[token]), int(token)) for token in top))
        results.append(
            asdict(parity)
            | {
                "position": row["position"],
                "variant": row["variant"],
                "max_returned_logprob_error": max(abs(actual[token] - float(values[token])) for token in actual),
                "max_returned_probability_error": max(
                    abs(float(np.exp(actual[token])) - float(np.exp(values[token]))) for token in actual
                ),
            }
        )
        parity.assert_matches(max_probability_error=maximum_probability_error)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--max-probability-error", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = compare(args.reference, args.observations, args.max_probability_error)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"rows": len(results), "max_probability_error": max(x["max_probability_error"] for x in results)}))


if __name__ == "__main__":
    main()
