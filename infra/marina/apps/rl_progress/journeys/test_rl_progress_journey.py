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
