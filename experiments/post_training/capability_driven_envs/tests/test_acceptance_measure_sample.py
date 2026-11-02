# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The acceptance measurement's sampler and manifest, offline."""

import json

import click
import pytest

pytest.importorskip("taskforge")

from experiments.post_training.capability_driven_envs.acceptance_measure.driver import (
    Store,
    ensure_manifest,
    manifest,
    manifest_digest,
    shard_allocations,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.sample import (
    Eligible,
    draw,
    sample_document,
    scan,
    stratified_order,
    wave_capabilities,
)

SEED = "test-seed"
GIT_SHA = "0" * 40


def proposal(verification: str = "code", null_reason: str | None = None, composite: list | None = None) -> bytes:
    header = {"verification": verification, "null_reason": null_reason, "composite": composite or []}
    return json.dumps({"header": header, "body": "x"}).encode()


def run_tree(
    capabilities: dict[str, dict[int, tuple[bytes, str]]], unfinished: tuple[str, ...] = ()
) -> dict[str, bytes]:
    files = {}
    for capability, slots in capabilities.items():
        for slot, (raw, decision) in slots.items():
            files[f"run/{capability}/slot-{slot:02d}.proposal.json"] = raw
            files[f"run/{capability}/slot-{slot:02d}.verdict.json"] = json.dumps({"decision": decision}).encode()
        if capability not in unfinished:
            files[f"run/{capability}/result.json"] = b"{}"
    return files


def scanned(files: dict[str, bytes]) -> dict:
    def listing(path: str) -> list[str]:
        depth = path.count("/") + 1
        return sorted({"/".join(p.split("/")[: depth + 1]) for p in files if p.startswith(path + "/")})

    return scan("run", lambda paths: {p: files[p] for p in paths}, listing)


def test_scan_keeps_finished_code_slots_that_preflight_accepted():
    files = run_tree(
        {
            "d01.a.x": {1: (proposal(), "accept"), 2: (proposal("decision_model"), "accept"), 3: (proposal(), "refuse")},
            "d01.a.y": {
                1: (proposal(null_reason="no task"), "accept"),
                2: (proposal(composite=[{"kind": "code"}]), "accept"),
            },
            "d02.b.z": {4: (proposal(), "accept")},
            "d02.b.w": {1: (proposal(), "accept")},
        },
        unfinished=("d02.b.w",),
    )
    result = scanned(files)

    assert (result["capabilities_listed"], result["capabilities_with_result"]) == (4, 3)
    assert sorted((e.capability, e.slot) for e in result["eligible"]) == [("d01.a.x", 1), ("d02.b.z", 4)]


def test_an_empty_eligible_set_is_refused_rather_than_written():
    files = run_tree({"d01.a.x": {1: (proposal("llm_as_judge"), "accept")}})

    with pytest.raises(click.ClickException, match="refusing an empty sample"):
        sample_document(scanned(files), root="run", seed=SEED, size=4, wave=2, catalog_capabilities=None)


def eligible(n_by_domain: dict[str, int]) -> list[Eligible]:
    return [
        Eligible(f"{d}.c{i}", slot, f"run/{d}.c{i}/slot-{slot:02d}.proposal.json", f"{d}{i}{slot}")
        for d, n in n_by_domain.items()
        for i in range(n)
        for slot in (1, 2, 3)
    ]


def test_every_prefix_of_the_order_keeps_each_domain_near_its_share():
    capabilities = sorted({e.capability for e in eligible({"d01": 60, "d02": 30, "d03": 10})})
    order = stratified_order(capabilities, SEED)

    assert sorted(order) == capabilities
    first = order[:20]
    assert [sum(c.startswith(d) for c in first) for d in ("d01", "d02", "d03")] == [12, 6, 2]


def test_the_draw_is_deterministic_and_takes_one_slot_per_capability():
    items = eligible({"d01": 5, "d02": 5})

    first, again = draw(items, SEED, 6), draw(list(reversed(items)), SEED, 6)

    assert first == again
    assert len({s.capability for s in first}) == 6 and [s.rank for s in first] == list(range(6))
    assert draw(items, "another-seed", 6) != first


def test_a_sample_is_marked_incomplete_while_the_run_is_in_flight_and_waves_split_it():
    files = run_tree({f"d01.c{i}": {1: (proposal(), "accept")} for i in range(5)}, unfinished=("d01.c4",))
    document = sample_document(scanned(files), root="run", seed=SEED, size=10, wave=2, catalog_capabilities=None)

    assert document["population"]["complete"] is False
    assert (document["size"], document["population"]["eligible_capabilities"]) == (4, 4)
    assert [e["rank"] for e in wave_capabilities(document, 1)] == [0, 1]
    assert [e["rank"] for e in wave_capabilities(document, 2)] == [2, 3]


def test_shards_partition_the_wave_and_hold_every_arm_of_a_capability():
    entries = [
        {"rank": r, "capability": f"d01.c{r}", "domain": "d01", "slot": 1, "proposal": f"p{r}", "sha256": "s"}
        for r in range(32)
    ]
    sample = {"sample": entries, "wave_size": 32}

    shards = [shard_allocations(sample, "1", s, 11, ("A", "B", "C")) for s in range(11)]

    keys = [a.key for shard in shards for a in shard]
    assert len(keys) == len(set(keys)) == 96
    assert all({a.arm for a in shard if a.capability == shard[0].capability} == {"A", "B", "C"} for shard in shards)


def test_the_manifest_is_written_once_and_a_differing_configuration_is_refused(tmp_path):
    store = Store(str(tmp_path / "out"))
    sample = b'{"sample": []}'
    first = manifest(sample, GIT_SHA)

    digest = ensure_manifest(store, first)
    written = json.loads(store.read("manifest.json"))

    assert written["digest"] == digest == manifest_digest(first)
    assert set(written["arms"]) == {"A", "B", "C"}
    assert len({arm["policy_digest"] for arm in written["arms"].values()}) == 3
    assert written["arms"]["C"]["arm_files"][0]["path"] == "A.json"
    assert ensure_manifest(store, manifest(sample, GIT_SHA)) == digest
    with pytest.raises(click.ClickException, match="sample_sha256"):
        ensure_manifest(store, manifest(b'{"sample": [1]}', GIT_SHA))
