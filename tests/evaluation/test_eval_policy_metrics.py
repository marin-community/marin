# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep the published policy's metric map aligned with its checked-in evaluators."""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from marin.evaluation.eval_policy_metrics import POLICY_METRICS_2026_10_16

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled"
EVALCHEMY = ROOT / "vendor/evalchemy/eval"
HARBOR = ROOT / "vendor/harbor/src/harbor"
MRCR_CONTEXT_VARIANTS = frozenset({"mrcr-32k", "mrcr-65k"})
LM_EVAL_OVERRIDES = {
    "gsm8k": EVALCHEMY / "lm_eval_tasks/gsm8k/gsm8k.yaml",
    "triviaqa": EVALCHEMY / "lm_eval_tasks/triviaqa/default.yaml",
    "truthfulqa_mc2": EVALCHEMY / "lm_eval_tasks/truthfulqa/truthfulqa_mc2.yaml",
}


def _class_metric(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Benchmark"):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "PRIMARY_METRIC" for target in statement.targets
            ):
                continue
            return ast.literal_eval(statement.value)
    raise AssertionError(f"{path}: no benchmark PRIMARY_METRIC declaration")


def test_policy_metric_map_covers_campaign_configs_and_vendored_evalchemy():
    evalchemy_configs = CAMPAIGN / "evalchemy-configs"
    harbor_configs = CAMPAIGN / "harbor-configs"
    config_names = {
        *(path.stem for path in evalchemy_configs.glob("*.yaml") if path.stem not in MRCR_CONTEXT_VARIANTS),
        *(path.stem for path in harbor_configs.glob("*.yaml")),
    }
    assert set(POLICY_METRICS_2026_10_16) == config_names

    for name, metric in POLICY_METRICS_2026_10_16.items():
        if metric.mechanism == "harbor":
            config = yaml.safe_load((harbor_configs / f"{name}.yaml").read_text())
            assert config["datasets"], name
            assert metric.task is None
            continue

        config = yaml.safe_load((evalchemy_configs / f"{name}.yaml").read_text())
        assert config["tasks"] == [metric.task], name
        assert metric.task is not None
        source = EVALCHEMY / "chat_benchmarks" / metric.task / "eval_instruct.py"
        if metric.task == "NUPA5K-Loose":
            source = EVALCHEMY / "chat_benchmarks/NUPA-Loose/eval_instruct.py"
        if source.exists():
            declared = _class_metric(source)
            if metric.task == "OlympiadBench":
                assert declared == "accuracy"
                assert 'return "accuracy_avg" if self.n_repeat > 1' in source.read_text()
            else:
                assert metric.evaluator_metric == declared, name
        else:
            override = LM_EVAL_OVERRIDES.get(metric.task)
            if override is not None:
                source_text = override.read_text()
                assert re.search(rf"^  - metric: {re.escape(metric.evaluator_metric)}$", source_text, re.MULTILINE)


def test_policy_metric_map_names_match_harbor_and_publication_surfaces():
    harbor_metadata = (HARBOR / "models/benchmark.py").read_text()
    assert 'name="reward"' in harbor_metadata
    harbor_metrics = {
        metric.evaluator_metric for metric in POLICY_METRICS_2026_10_16.values() if metric.mechanism == "harbor"
    }
    assert harbor_metrics == {"reward"}

    crux = POLICY_METRICS_2026_10_16["cruxeval"]
    crux_source = (EVALCHEMY / "chat_benchmarks/CruxEval/eval_instruct.py").read_text()
    assert 'groups={"direction": "{group}_{metric}"}' in crux_source
    assert crux.published_metric == f"output_{crux.evaluator_metric}"
    assert crux.record_metric == "pass_at_1"

    sotopia = POLICY_METRICS_2026_10_16["sotopia-hard"]
    rank_source = (CAMPAIGN / "analysis/sotopia_length_control/rank.py").read_text()
    assert f"score.{sotopia.published_metric}" in rank_source
    assert sotopia.published_metric != sotopia.record_metric


def test_evalchemy_metric_names_project_to_evaldash_record_keys():
    metrics = {
        metric.evaluator_metric: metric.record_metric
        for metric in POLICY_METRICS_2026_10_16.values()
        if metric.mechanism == "evalchemy"
    }
    package_paths = [EVALCHEMY.parent, EVALCHEMY.parent / "packages/evalchemy-config/src"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from eval.contracts.benchmark_metadata import canonical_metric_name; "
            "print(json.dumps({name: canonical_metric_name(name) for name in json.load(sys.stdin)}))",
        ],
        input=json.dumps(list(metrics)),
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(str(path) for path in package_paths)},
    )
    assert json.loads(result.stdout) == metrics
