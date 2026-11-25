# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IFM-style SFT filter audit over the pinned Mochi SFT mix (report-only; #12264 item 1/2).

Runs the three IFM loop rules over every pack of every store in the SFT manifest and
reports per-source fire rates, a precision sample for hand-checking, and the wait-behavior
count after unchanged terminal output. Nothing is filtered or rebuilt:

    uv run --package marin-core python -m experiments.post_training.mochi_loop_guards.ifm_filter_audit \\
        --manifest <216-store manifest.json> --output <json path> --sample 50
"""

import argparse
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from rolloutengine.repetition import action_identity, observation_identity, wait_only
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

RUN_LENGTH = 8
FREQUENT_COUNT = 10
FREQUENT_SHARE = 0.3
MESSAGE_RUN_LENGTH = 8
MESSAGE_OBSERVATION_SHARE = 0.8

RUN8 = "run8"
FREQUENT = "frequent"
TERMINUS8 = "terminus8"
RULES = (RUN8, FREQUENT, TERMINUS8)


def pack_actions(messages: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, bool]]:
    """One (action identity, observation identity, wait-only) triple per assistant turn.

    The observation is the conversation after the assistant message, exactly as the audit
    and the recorder read it.
    """
    triples = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        end = next(
            (stop for stop in range(index + 1, len(messages)) if messages[stop].get("role") == "assistant"),
            len(messages),
        )
        observations = observation_identity(messages[index + 1 : end])
        triples.append((action_identity(message), observations, wait_only(message)))
    return triples


def fire_rules(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """The IFM rules a pack's conversation trips, in report-only form."""
    actions = pack_actions(messages)
    fired: list[str] = []
    seen: set[str] = set()
    previous: str | None = None
    run = 0
    for action, _, wait in actions:
        run = run + 1 if action == previous else 1
        previous = action
        if wait:
            continue
        if run >= RUN_LENGTH and RUN8 not in seen:
            seen.add(RUN8)
            fired.append(RUN8)
    counts: dict[str, int] = {}
    for action, _, is_wait in actions:
        if not is_wait:
            counts[action] = counts.get(action, 0) + 1
    frequent = {
        action for action, count in counts.items() if count >= FREQUENT_COUNT and count >= FREQUENT_SHARE * len(actions)
    }
    previous = None
    run = 0
    for action, _, _ in actions:
        run = run + 1 if action == previous else 1
        previous = action
        if run >= 3 and action in frequent and FREQUENT not in seen:
            seen.add(FREQUENT)
            fired.append(FREQUENT)
    texts = [message.get("content") or "" for message in messages if message.get("role") == "assistant"]
    previous = None
    run = 0
    for index, text in enumerate(texts):
        run = run + 1 if text == previous else 1
        previous = text
        if run < MESSAGE_RUN_LENGTH:
            continue
        if _observation_covers(messages, index - run + 1, run) and TERMINUS8 not in seen:
            seen.add(TERMINUS8)
            fired.append(TERMINUS8)
    return fired


def _observation_covers(messages: Sequence[Mapping[str, Any]], start: int, length: int) -> bool:
    """Whether one observation batch precedes at least the required share of a run's turns."""
    assistant_indexes = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if start + length > len(assistant_indexes):
        return False
    counts: dict[str, int] = {}
    for position in range(start + 1, start + length):
        index = assistant_indexes[position]
        end = next(
            (stop for stop in range(index + 1, len(messages)) if messages[stop].get("role") == "assistant"),
            len(messages),
        )
        identity = observation_identity(messages[index + 1 : end])
        counts[identity] = counts.get(identity, 0) + 1
    return bool(counts) and max(counts.values()) >= MESSAGE_OBSERVATION_SHARE * length


def wait_after_unchanged_output(messages: Sequence[Mapping[str, Any]]) -> int:
    """Assistant turns that wait when the terminal output is unchanged between turns.

    A wait counts when the observation preceding it is identical to the observation that
    preceded the previous assistant turn: the agent polled and nothing changed.
    """
    turns = pack_actions(messages)
    count = 0
    for index in range(2, len(turns)):
        _, previous_observation, _ = turns[index - 2]
        _, observation, _ = turns[index - 1]
        if observation == previous_observation and turns[index][2]:
            count += 1
    return count


def audit_store(name: str, packs: Iterator[Mapping[str, Any]], *, sample_size: int) -> dict[str, Any]:
    """Fire rates, precision sample, and wait counts for one store's packs."""
    fired = {rule: 0 for rule in RULES}
    wait_counts = 0
    pack_count = 0
    sample: list[dict[str, Any]] = []
    for pack in packs:
        messages = pack.get("messages") or pack.get("conversations") or ()
        if not messages:
            continue
        pack_count += 1
        rules = fire_rules(messages)
        for rule in rules:
            fired[rule] += 1
        wait_counts += wait_after_unchanged_output(messages)
        if rules and len(sample) < sample_size:
            sample.append({"source": name, "rules": rules, "actions": pack_actions(messages)})
    rates = {rule: fired[rule] / pack_count if pack_count else 0.0 for rule in RULES}
    return {
        "source": name,
        "packs": pack_count,
        "fires": fired,
        "fire_rates": rates,
        "repeat_turns_total": None,
        "wait_after_unchanged_output": wait_counts,
        "precision_sample": sample,
    }


def read_manifest_stores(manifest: Mapping[str, Any]) -> list[str]:
    """The store URIs of the pinned SFT manifest, in manifest order."""
    stores = manifest.get("stores") or manifest.get("candidate_store_uri")
    if isinstance(stores, str):
        return [stores]
    if isinstance(stores, list):
        return [store.get("uri", store) if isinstance(store, Mapping) else store for store in stores]
    raise ValueError("the SFT manifest must list its stores under 'stores' (or one 'candidate_store_uri')")


def audit_mix(manifest_path: str, output: str, *, sample_size: int) -> dict[str, Any]:
    """Audit every store of the manifest and write the JSON and Markdown reports."""
    manifest = json.loads(StoragePath(manifest_path).read_text())
    stores = read_manifest_stores(manifest)
    reports = []
    for index, uri in enumerate(stores):
        name = f"store_{index:03d}"
        logger.info("auditing %s (%s)", name, uri)

        def packs(uri: str = uri) -> Iterator[Mapping[str, Any]]:
            yield from load_file(uri)

        reports.append(audit_store(name, packs(), sample_size=sample_size))
    fired_total = {rule: sum(report["fires"][rule] for report in reports) for rule in RULES}
    document = {
        "manifest": manifest_path,
        "stores": len(stores),
        "total_fires": fired_total,
        "wait_after_unchanged_output": sum(report["wait_after_unchanged_output"] for report in reports),
        "reports": reports,
    }
    Path(output).write_text(json.dumps(document, indent=2) + "\n")
    markdown = [
        "| Store | Packs | run8 | frequent | terminus8 | Waits after unchanged output |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        markdown.append(
            f"| {report['source']} | {report['packs']} | {report['fire_rates'][RUN8]:.4f} "
            f"| {report['fire_rates'][FREQUENT]:.4f} | {report['fire_rates'][TERMINUS8]:.4f} "
            f"| {report['wait_after_unchanged_output']} |"
        )
    (output.removesuffix(".json") + ".md").write_text("\n".join(markdown) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="The pinned SFT store manifest JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=50, help="Hand-check sample size per store")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    document = audit_mix(args.manifest, args.output, sample_size=args.sample)
    print(json.dumps(document["total_fires"], indent=2))


if __name__ == "__main__":
    main()
