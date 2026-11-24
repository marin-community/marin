# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from urllib.parse import urlencode

from finestore.rl.rl_progress import Comparison
from marina.journeys import Journey
from playwright.sync_api import expect

from infra.marina.tests.rl_progress_fixtures import APPLET_ID, grader_report, progress_record


def test_stored_attempts_keep_links_breakdowns_and_refresh(journey: Journey) -> None:
    api = f"{journey.origin}/a/{APPLET_ID}/api/runs"
    record = progress_record() | {
        "run_id": "journey/run",
        "label": "Learning fixture",
        "model": "snowball",
        "evaldash_run_ids": ["first-evaluation", "second-evaluation"],
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
    route = urlencode({"run": record["run_id"], "attempt": record["attempt_id"]})
    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/#{route}", wait_until="domcontentloaded")
    dialog = journey.page.get_by_role("dialog")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("link", name="Grafana")).to_have_attribute("href", record["grafana_url"])
    expect(dialog.get_by_role("link", name=f"Experiment #{record['issue']}")).to_have_attribute(
        "href", f"https://github.com/marin-community/marin/issues/{record['issue']}"
    )
    for evaluation_id in record["evaldash_run_ids"]:
        expect(dialog.get_by_role("link", name=f"EvalDash {evaluation_id}")).to_have_attribute(
            "href", f"https://marina.oa.dev/evaldash/runs/{evaluation_id}"
        )
    filters = urlencode({"rl_run": record["run_id"], "rl_attempt": record["attempt_id"], "view": "runs"})
    expect(dialog.get_by_role("link", name="All EvalDash evaluations of this attempt")).to_have_attribute(
        "href", f"https://marina.oa.dev/evaldash/runs?{filters}"
    )
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
    expect(journey.page.locator("#attempts > tr")).to_have_count(1)
    response = journey.page.request.get(api)
    assert response.ok and len(response.json()["runs"]) == 2
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


def test_progress_anchors_experiments_and_keeps_the_global_protocol_point(journey: Journey) -> None:
    api = f"{journey.origin}/a/{APPLET_ID}/api/runs"
    overall, family = "eval_lite/family_mean", "eval_lite/family/small/mean"
    revisions = [f"2026.11.{day}" for day in (1, 2, 3)]
    commits = [letter * 10 for letter in "abc"]

    def attempt(name, revision, day, raw):
        return progress_record() | {
            "run_id": name,
            "attempt_id": name,
            "label": name,
            "model": "snowball",
            "preset": "learning-medium",
            "preset_revision": revision,
            "started_at": f"2026-11-0{day}T00:00:00Z",
            "completed_at": f"2026-11-0{day}T01:00:00Z",
            "gate_result": None,
            "evaluations": [
                Comparison(
                    metric=metric.replace("eval_lite/", "eval_lite@25/"), role="slice", run=raw, n_run=1, n_control=0
                ).model_dump(mode="json")
                for metric in (overall, family)
            ],
        }

    def link(run, base, delta, *, invalid=False):
        return progress_record()["gate_result"] | {
            "attempt_ids": [run],
            "control_commit": commits[base],
            "control_revision": revisions[base],
            "verdict": "invalid" if invalid else "pass",
            "reason": "evaluation protocol differs" if invalid else None,
            "comparisons": [
                Comparison(metric=metric, role=role, delta=delta, tolerance=0.03, n_run=1, n_control=1).model_dump(
                    mode="json"
                )
                for metric, role in ((overall, "guard"), (family, "slice"))
            ],
        }

    for i, (revision, raw) in enumerate(zip(revisions, (0.6, 0.4, 0.7), strict=True)):
        name = f"control-{i}"
        record = attempt(name, revision, i + 1, raw) | {
            "artifact_uri": f"s3://example/control/learning-medium-{commits[i]}-r0/{revision}",
            "evaluation_data_identity": f"panel@{revision}",
            "evaluation_protocol_identity": "protocol-a" if i < 2 else "protocol-b",
            "gate_result": link(name, i - 1, 0.1, invalid=i == 2) if i else None,
        }
        response = journey.page.request.post(api, data=record)
        assert response.ok, response.text()

    branch = attempt("branch-at-first-preset", "dev", 4, 0.55) | {
        "branch": "experiment",
        "base_preset_revision": revisions[0],
        "gate_result": link("branch-at-first-preset", 0, -0.05),
    }
    response = journey.page.request.post(api, data=branch)
    assert response.ok, response.text()

    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/", wait_until="domcontentloaded")
    chart = journey.page.get_by_role("group", name=overall, exact=True)
    expect(chart.locator("line.history-chain")).to_have_count(1)
    expect(chart.locator("line[aria-label='Evaluation set changed']")).to_have_count(1)
    expect(chart.locator("line[aria-label='Protocol changed']")).to_have_count(1)
    expect(chart.locator("circle.history-global > title").last).to_contain_text("Experiment global 55.0%")
    expect(chart.locator("circle.history-global > title").nth(2)).to_contain_text("Global 70.0%")
    expect(journey.page.get_by_role("group", name=family, exact=True).locator("line.history-chain")).to_have_count(1)
    chart.get_by_role("link", name="branch-at-first-preset · 25.0%").locator("circle").first.click()
    expect(journey.page.get_by_role("dialog")).to_contain_text(revisions[0])
