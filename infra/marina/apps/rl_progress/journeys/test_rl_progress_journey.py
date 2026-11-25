# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from urllib.parse import urlencode

from finestore.rl.rl_progress import Comparison, EvaluationScore
from marina.journeys import Journey
from playwright.sync_api import expect

from infra.marina.tests.rl_progress_fixtures import APPLET_ID, grader_report, progress_record


def test_stored_attempts_keep_links_breakdowns_and_refresh(journey: Journey) -> None:
    api = f"{journey.origin}/a/{APPLET_ID}/api/runs"
    record = progress_record() | {
        "run_id": "journey/run",
        "native_run_id": "native/journey/run",
        "label": "Learning fixture",
        "model": "snowball",
        "grader_report": grader_report(),
        "grafana_url": "https://grafana.oa.dev/d/marin-rl-runs/rl-post-training?from=1&to=now&var-run=journey%2Frun",
        "evaluations": [
            Comparison(metric=metric, role="slice", run=value, tasks=2, n_run=1, n_control=0).model_dump(mode="json")
            for metric, value in (
                ("eval_lite_100@25/family/small/mean", 0.25),
                ("eval_lite_100@25/aai_category/coding/mean", 0.375),
                ("eval_lite_100@25/difficulty/hard/mean", 0.5),
            )
        ],
    }

    def publish(value):
        response = journey.page.request.post(api, data=value)
        assert response.ok, response.text()
        return response.json()

    assert publish(record) == record
    second = record | {
        "attempt_id": "native-second",
        "status": "failed",
        "gate_result": None,
        "started_at": "2026-11-03T00:00:00Z",
        "completed_at": "2026-11-03T01:00:00Z",
        "failed_facts": ["eval_lite_100@25/task_coverage"],
    }
    second.pop("issue")
    publish(second)
    publish(
        progress_record()
        | {"run_id": record["native_run_id"], "attempt_id": "legacy-native", "label": "Earlier native attempt"}
    )
    route = urlencode({"run": record["run_id"], "attempt": record["attempt_id"]})
    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/#{route}", wait_until="domcontentloaded")
    dialog = journey.page.get_by_role("dialog")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("link", name="Grafana")).to_have_attribute("href", record["grafana_url"])
    expect(dialog.get_by_role("link", name=f"Experiment #{record['issue']}")).to_have_attribute(
        "href", f"https://github.com/marin-community/marin/issues/{record['issue']}"
    )
    native_route = urlencode({"run": record["native_run_id"], "attempt": record["attempt_id"]})
    journey.page.goto("about:blank")
    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/#{native_route}", wait_until="domcontentloaded")
    expect(dialog.get_by_text("Learning fixture", exact=True)).to_be_visible()
    expect(journey.page.get_by_role("cell", name="snowball", exact=True)).to_be_visible()
    for metric, value in (("family/small", "0.2500"), ("aai_category/coding", "0.3750"), ("difficulty/hard", "0.5000")):
        expect(dialog.get_by_text(f"eval_lite_100@25/{metric}/mean: {value}", exact=False)).to_be_visible()
    expect(journey.page.locator("#attempts").get_by_role("link", name="Grafana")).to_have_attribute(
        "href", record["grafana_url"]
    )
    dialog.get_by_text("eval_lite_100@25/difficulty/hard/mean: 0.5000", exact=False).scroll_into_view_if_needed()
    journey.shoot("stored-attempt-breakdowns")
    dialog.get_by_role("link", name="Grader stats").click()
    expect(journey.page.get_by_role("cell", name="test/source", exact=True).first).to_be_visible()
    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/#{route}", wait_until="domcontentloaded")
    expect(dialog).to_be_visible()
    journey.page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    journey.click("Learning fixture")
    expect(dialog).to_be_visible()
    picker = journey.page.get_by_role("combobox", name="Attempt for Learning fixture")
    dismissed_hash = journey.page.evaluate(
        """() => {
        document.querySelector('dialog').requestClose();
        const dismissedHash = location.hash;
        const picker = document.querySelector('#attempts select');
        picker.value = 'native-second';
        picker.dispatchEvent(new Event('change'));
        return dismissedHash;
    }"""
    )
    assert dismissed_hash == ""
    expect(dialog).to_contain_text("eval_lite_100@25/task_coverage")
    expect(dialog.get_by_role("link", name="Experiment #")).to_have_count(0)
    journey.page.mouse.click(5, 5)
    expect(dialog).not_to_be_visible()
    picker.select_option("native-first")
    expect(dialog).to_be_visible()
    journey.page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    changed = deepcopy(record)
    changed["label"] = "Updated learning fixture"
    changed["evaluations"][0]["run"] = 0.625
    assert publish(changed) == changed
    journey.click("Refresh")
    journey.click("Updated learning fixture")
    expect(dialog).to_contain_text("family/small/mean: 0.6250")
    journey.page.reload(wait_until="domcontentloaded")
    expect(dialog).to_be_visible()
    expect(dialog).to_contain_text("family/small/mean: 0.6250")
    expect(journey.page.locator("#attempts > tr")).to_have_count(2)
    response = journey.page.request.get(api)
    assert response.ok and len(response.json()["runs"]) == 3
    journey.page.keyboard.press("Escape")
    publish(record | {"run_id": "journey/mochi", "label": "Mochi fixture", "model": "mochi"})
    journey.click("Refresh")
    model_filter = journey.page.get_by_role("combobox", name="Model", exact=True)
    model_filter.select_option("mochi")
    expect(journey.page.get_by_role("button", name="Mochi fixture")).to_be_visible()
    expect(journey.page.get_by_role("button", name="Updated learning fixture")).to_have_count(0)
    model_filter.select_option("snowball")
    expect(journey.page.get_by_role("button", name="Updated learning fixture")).to_be_visible()
    expect(journey.page.get_by_role("button", name="Mochi fixture")).to_have_count(0)


def test_progress_shows_main_segments_and_optional_branch_attempts(journey: Journey) -> None:
    api = f"{journey.origin}/a/{APPLET_ID}/api/runs"
    metric = "eval_lite/family_mean"

    def attempt(name, day, raw, *, comparable="tasks-a/grader-a", final=True):
        return progress_record() | {
            "run_id": name,
            "attempt_id": name,
            "label": name,
            "model": "snowball",
            "branch": "main",
            "dirty": False,
            "is_control": True,
            "commit": f"{day:040x}",
            "base_commit": f"{day:040x}",
            "preset_version": f"bbbbbbb{day}",
            "artifact_uri": f"s3://example/runs/{name}",
            "started_at": f"2026-11-0{day}T00:00:00Z",
            "completed_at": f"2026-11-0{day}T01:00:00Z",
            "gate_result": None,
            "evaluations": [
                Comparison(
                    metric="eval_lite@25/family_mean",
                    role="slice",
                    run=raw,
                    n_run=1,
                    n_control=0,
                    evaluation_score=EvaluationScore(
                        profile="eval_lite",
                        step=25 if final else 0,
                        statistic="family_mean",
                        final=final,
                        comparability=comparable,
                    ),
                ).model_dump(mode="json")
            ],
        }

    controls = [
        attempt(f"control-{i}", i + 1, raw, comparable=identity) | {"evaluation_data_identity": f"panel@selection-v{i}"}
        for i, (raw, identity) in enumerate(
            ((0.5, "tasks-a/grader-a"), (0.4, "tasks-a/grader-a"), (0.7, "tasks-b/grader-a"), (0.75, "tasks-b/grader-b"))
        )
    ]
    controls[0]["run_id"] = controls[1]["run_id"] = "shared-index-run"
    controls[0]["commit"] = controls[0]["base_commit"][:10]
    branch = attempt("branch-at-first-preset", 5, 0.55) | {
        "branch": "experiment",
        "is_control": False,
        "commit": "d" * 40,
        "base_preset_version": controls[0]["preset_version"],
        "settings": ["trainer.max_steps=25"],
        "launch_config": {"skyrl": {"trainer": {"max_steps": 25}}},
        "wandb_url": "https://wandb.ai/marin-community/marin-rl/runs/fixture",
        "grafana_url": "https://grafana.oa.dev/d/marin-rl-runs/rl-post-training?var-run=fixture",
        "gate_result": progress_record()["gate_result"],
    }
    superseded = attempt("superseded", 1, 0.15) | {
        "artifact_uri": controls[0]["artifact_uri"],
        "completed_at": "2026-11-01T00:30:00Z",
    }
    early_only = attempt("early-only", 6, 0.2, final=False)
    unpushed_main = attempt("unpushed-main", 7, 0.65) | {"base_commit": controls[0]["base_commit"]}
    main_without_control = attempt("main-without-control", 8, 0.68) | {"is_control": False}
    for record in (*controls, branch, superseded, early_only, unpushed_main, main_without_control):
        response = journey.page.request.post(api, data=record)
        assert response.ok, response.text()

    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/", wait_until="domcontentloaded")
    journey.page.get_by_role("combobox", name="Preset", exact=True).select_option(label="snowball · learning-medium")
    chart = journey.page.get_by_role("group", name=metric, exact=True)
    expect(chart.locator("line.history-global")).to_have_count(3)
    chain = chart.locator("line.history-chain")
    expect(chain).to_have_count(1)
    expect(chain).to_have_attribute("data-from-attempt", "control-0")
    expect(chain).to_have_attribute("data-to-attempt", "control-1")
    expect(chart.locator("line.history-marker")).to_have_count(2)
    for attempt_id in ("control-2", "control-3"):
        expect(chart.locator(f'line.history-marker[data-attempt="{attempt_id}"]')).to_have_count(1)
    expect(chart.get_by_role("link")).to_have_count(4)
    for name in ("superseded", "early-only", "unpushed-main", "main-without-control"):
        expect(chart.get_by_role("link", name=re.compile(f"^{name}"))).to_have_count(0)
    journey.shoot("preset-main-controls")
    chart.get_by_role("link", name="control-0 · 50.0%").click()
    expect(journey.page.get_by_role("dialog")).to_contain_text("control-0")
    journey.page.keyboard.press("Escape")

    toggle = journey.page.get_by_role("checkbox", name="Show branch runs")
    toggle.check()
    expect(chart.get_by_role("link")).to_have_count(5)
    expect(chart.locator("line.history-global")).to_have_count(3)
    journey.shoot("preset-branch-runs")
    chart.get_by_role("link", name="branch-at-first-preset · 55.0%").click()
    dialog = journey.page.get_by_role("dialog")
    expect(dialog).to_contain_text(branch["base_preset_version"])
    expect(dialog).to_contain_text(branch["commit"])
    expect(dialog).to_contain_text("trainer.max_steps=25")
    expect(dialog.get_by_role("link", name="W&B run")).to_have_attribute("href", branch["wandb_url"])
    expect(dialog.get_by_role("link", name="Grafana")).to_have_attribute("href", branch["grafana_url"])
    dialog.get_by_text("Composed launch configuration", exact=True).click()
    expect(dialog.locator("pre")).to_contain_text('"max_steps": 25')
    journey.shoot("branch-run-detail")
