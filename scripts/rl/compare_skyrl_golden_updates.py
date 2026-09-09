# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare captured and offline-replayed Torch updates in the artifact's region."""

import argparse
import json
import logging
import posixpath

import fsspec
import numpy as np

LOGGER = logging.getLogger(__name__)


def read_json(uri: str) -> dict:
    with fsspec.open(uri, "rt") as source:
        return json.load(source)


def deviation(actual, expected, *, atol: float, rtol: float) -> dict:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": float(difference.max(initial=0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def canonical_group_ids(group_ids: np.ndarray) -> np.ndarray:
    """Number groups in first-occurrence order, independently of opaque labels."""
    _, first, inverse = np.unique(group_ids, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inverse]


def compare_updates(first_uri: str, replay_uri: str, ranks: int) -> dict:
    """Check token-level replay and bounded model/master-parameter update probes."""
    identity_fields = ("uri", "size", "etag", "version_id", "checksum_sha256")
    source_objects = []
    for uri in (first_uri, replay_uri):
        identity_uri = posixpath.join(posixpath.dirname(uri), "source-identity.json")
        record = read_json(identity_uri)
        source_objects.append(
            sorted(
                ({name: item[name] for name in identity_fields} for item in record["source_objects"]),
                key=lambda item: item["uri"],
            )
        )
    checks = {
        "provenance/source_objects": {
            "passed": bool(source_objects[0]) and source_objects[0] == source_objects[1],
            "first_object_count": len(source_objects[0]),
            "replay_object_count": len(source_objects[1]),
        }
    }
    with (
        fsspec.open(first_uri, "rb") as first_source,
        fsspec.open(replay_uri, "rb") as replay_source,
        np.load(first_source, allow_pickle=False) as first,
        np.load(replay_source, allow_pickle=False) as replay,
    ):
        for name in (
            "sequences",
            "attention_mask",
            "response_mask",
            "loss_mask",
            "objective_partition_ids",
            "rewards",
            "behavior_logprobs",
        ):
            checks[name] = deviation(replay[name], first[name], atol=0, rtol=0)
        checks["group_ids"] = deviation(
            canonical_group_ids(replay["group_ids"]), canonical_group_ids(first["group_ids"]), atol=0, rtol=0
        )
        for name in ("old_logprobs", "advantages", "logprob_gradients"):
            checks[name] = deviation(replay[name], first[name], atol=1e-7, rtol=1e-5)
        first_manifest = json.loads(str(first["manifest"]))
        replay_manifest = json.loads(str(replay["manifest"]))
        checks["oracle_loss"] = deviation(
            replay_manifest["oracle"]["loss"], first_manifest["oracle"]["loss"], atol=1e-7, rtol=1e-5
        )
        for name in ("algorithm", "policy"):
            checks[f"config/{name}"] = {
                "passed": first_manifest["config"]["trainer"][name] == replay_manifest["config"]["trainer"][name]
            }
        for name in ("source_sha256", "initial_policy", "initial_optimizer", "rng_seed", "torch_version"):
            checks[f"provenance/{name}"] = {
                "passed": first_manifest["provenance"][name] == replay_manifest["provenance"][name]
            }
    for rank in range(ranks):
        first = read_json(f"{first_uri}.rank-{rank}.json")
        replay = read_json(f"{replay_uri}.rank-{rank}.json")
        prefix = f"rank/{rank}"
        checks[f"{prefix}/layout"] = {"passed": first["before"]["names"] == replay["before"]["names"]}
        checks[f"{prefix}/grad_norm"] = deviation(replay["grad_norm"], first["grad_norm"], atol=1e-7, rtol=1e-5)
        for name in ("values", "optimizer_parameter_values"):
            checks[f"{prefix}/initial/{name}"] = deviation(replay["before"][name], first["before"][name], atol=0, rtol=0)
            first_delta = np.subtract(first["after"][name], first["before"][name])
            replay_delta = np.subtract(replay["after"][name], replay["before"][name])
            checks[f"{prefix}/update/{name}"] = deviation(replay_delta, first_delta, atol=1e-8, rtol=1e-4)
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "first_uri": first_uri,
        "replay_uri": replay_uri,
        "ranks": ranks,
        "checks": checks,
        "scope": "fixed-rollout Torch replay with bounded parameter probes; not full optimizer-state equivalence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-uri", required=True)
    parser.add_argument("--replay-uri", required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args()
    if args.ranks < 1:
        raise ValueError("ranks must be positive")
    result = compare_updates(args.first_uri, args.replay_uri, args.ranks)
    with fsspec.open(args.output_uri, "wt") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    LOGGER.info("Replay comparison passed=%s: %s", result["passed"], args.output_uri)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
