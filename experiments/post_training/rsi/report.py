# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the policies of one ``evaluate`` run on the same tasks, with paired bootstrap intervals over tasks.

Run on the output directory of ``run.py evaluate``::

    python -m experiments.post_training.rsi.report s3://<bucket>/<prefix>/evaluate --pair rsi:control
"""

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

import click
from rigging.filesystem.storage_path import StoragePath

BOOTSTRAP_SAMPLES = 10000
SEED = 0


def policy_name(path: StoragePath) -> str:
    return path.name.removeprefix("groups-").removesuffix(".jsonl")


def response_tokens(path: StoragePath) -> dict[str, list[int]]:
    """The sampled tokens of each attempt of one policy's records, by report group."""
    tokens: defaultdict[str, list[int]] = defaultdict(list)
    for line in path.read_text().splitlines():
        record = json.loads(line)
        for group in groups_of(record["label"]):
            tokens[group] += record["response_tokens"]
    return tokens


def task_rates(path: StoragePath) -> dict[str, tuple[str, float]]:
    """The label and the pass rate of each graded task of one policy's records."""
    rates = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["pass_rate"] is not None:
            rates[record["task_id"]] = (record["label"], record["pass_rate"])
    return rates


def groups_of(label: str) -> list[str]:
    """The report groups of a label: ``probe:<capability>`` counts in ``probe:*``, ``pool:<stratum>`` in ``pool:*``."""
    kind = label.split(":", 1)[0]
    return [label, f"{kind}:*"]


def paired_delta(first: Sequence[float], second: Sequence[float], rng: random.Random) -> dict[str, float]:
    """The mean of ``first - second`` over tasks, with a 95% bootstrap interval over tasks."""
    differences = [a - b for a, b in zip(first, second, strict=True)]
    means = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "delta": sum(differences) / len(differences),
        "low": means[int(0.025 * BOOTSTRAP_SAMPLES)],
        "high": means[int(0.975 * BOOTSTRAP_SAMPLES)],
    }


def compare(
    rates: Mapping[str, Mapping[str, tuple[str, float]]], pairs: Sequence[tuple[str, str]]
) -> dict[str, dict[str, object]]:
    """For each report group: the tasks, each policy's mean pass rate, and the paired delta of each pair."""
    names = list(rates)
    shared = set.intersection(*(set(policy_rates) for policy_rates in rates.values()))
    members: defaultdict[str, list[str]] = defaultdict(list)
    for task_id in sorted(shared):
        for group in groups_of(rates[names[0]][task_id][0]):
            members[group].append(task_id)
    rng = random.Random(SEED)
    table = {}
    for group, task_ids in sorted(members.items()):
        values = {name: [rates[name][task_id][1] for task_id in task_ids] for name in names}
        row: dict[str, object] = {
            "tasks": len(task_ids),
            **{name: sum(series) / len(series) for name, series in values.items()},
        }
        for first, second in pairs:
            row[f"{first}-{second}"] = paired_delta(values[first], values[second], rng)
        table[group] = row
    return table


def unlocked(rates: Mapping[str, Mapping[str, tuple[str, float]]], base: str) -> dict[str, dict[str, float]]:
    """For each report group, the share of the tasks that ``base`` never passes and that each policy passes at all.

    These are the capabilities that the base policy does not have: no sample of it passes the task.
    """
    shared = set.intersection(*(set(policy_rates) for policy_rates in rates.values()))
    unsolved = [task_id for task_id in sorted(shared) if rates[base][task_id][1] == 0.0]
    members: defaultdict[str, list[str]] = defaultdict(list)
    for task_id in unsolved:
        for group in groups_of(rates[base][task_id][0]):
            members[group].append(task_id)
    return {
        group: {
            "tasks": len(task_ids),
            **{name: sum(rates[name][task_id][1] > 0 for task_id in task_ids) / len(task_ids) for name in rates},
        }
        for group, task_ids in sorted(members.items())
    }


def markdown(table: Mapping[str, Mapping[str, object]], names: Sequence[str], pairs: Sequence[tuple[str, str]]) -> str:
    header = ["group", "tasks", *names, *(f"{first} - {second} (95% CI)" for first, second in pairs)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for group, row in table.items():
        cells = [group, str(row["tasks"]), *(f"{row[name]:.3f}" for name in names)]
        for first, second in pairs:
            delta = row[f"{first}-{second}"]
            assert isinstance(delta, dict)
            cells.append(f"{delta['delta']:+.3f} [{delta['low']:+.3f}, {delta['high']:+.3f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@click.command()
@click.argument("evaluation_dir")
@click.option("--pair", "pairs", multiple=True, help="FIRST:SECOND policies to compare; repeatable.")
@click.option("--base", default="base", show_default=True, help="The policy before training.")
def main(evaluation_dir: str, pairs: tuple[str, ...], base: str) -> None:
    """Print the comparison table of an evaluation and write it as report.md and report.json."""
    root = StoragePath(evaluation_dir)
    paths = sorted((root / "groups-*.jsonl").glob(), key=str)
    rates = {policy_name(path): task_rates(path) for path in paths}
    split = [tuple(pair.split(":", 1)) for pair in pairs]
    table = compare(rates, split)
    text = markdown(table, list(rates), split)
    lengths = {policy_name(path): response_tokens(path) for path in paths}
    newly = unlocked(rates, base)
    text += "\nTasks that the base policy never passes, and the share that each policy passes at least once:\n\n"
    text += "| group | tasks | " + " | ".join(rates) + " |\n|" + "---|" * (len(rates) + 2) + "\n"
    for group, row in newly.items():
        text += f"| {group} | {row['tasks']} | " + " | ".join(f"{row[name]:.3f}" for name in rates) + " |\n"
    text += (
        "\nMean sampled tokens of an attempt:\n\n| group | "
        + " | ".join(lengths)
        + " |\n|"
        + "---|" * (len(lengths) + 1)
    )
    text += "\n"
    for group in sorted(next(iter(lengths.values()))):
        cells = [f"{sum(lengths[name][group]) / max(len(lengths[name][group]), 1):.0f}" for name in lengths]
        text += f"| {group} | " + " | ".join(cells) + " |\n"
    (root / "report.md").write_text(text)
    (root / "report.json").write_text(json.dumps({"pass_rates": table, "unlocked": newly}, indent=1))
    click.echo(text)


if __name__ == "__main__":
    main()
