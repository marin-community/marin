# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from urllib.parse import urlencode

from marina.journeys import Journey
from playwright.sync_api import expect

from infra.marina.tests.rl_progress_fixtures import APPLET_ID, progress_record


def test_stored_attempts_keep_links_breakdowns_and_refresh(journey: Journey) -> None:
    api = f"{journey.origin}/a/{APPLET_ID}/api/runs"
    record = progress_record() | {
        "run_id": "journey/run",
        "label": "Learning fixture",
        "grafana_url": "https://grafana.oa.dev/d/marin-rl-runs/rl-post-training?from=1&to=now&var-run=journey%2Frun",
        "evaluations": [
            {"metric": metric, "role": "slice", "run": value, "tasks": 2, "n_run": 1, "n_control": 0}
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
        "failed_facts": ["eval_lite_100@25/scored_fraction"],
    }
    publish(second)
    route = urlencode({"run": record["run_id"], "attempt": record["attempt_id"]})
    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/#{route}", wait_until="domcontentloaded")
    dialog = journey.page.get_by_role("dialog")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("link", name="Grafana")).to_have_attribute("href", record["grafana_url"])
    for metric, value in (("family/small", "0.2500"), ("aai_category/coding", "0.3750"), ("difficulty/hard", "0.5000")):
        expect(dialog.get_by_text(f"eval_lite_100@25/{metric}/mean: {value}", exact=False)).to_be_visible()
    expect(journey.page.locator("#attempts").get_by_role("link", name="Grafana")).to_have_attribute(
        "href", record["grafana_url"]
    )
    dialog.get_by_text("eval_lite_100@25/difficulty/hard/mean: 0.5000", exact=False).scroll_into_view_if_needed()
    journey.shoot("stored-attempt-breakdowns")
    journey.page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    picker = journey.page.get_by_role("combobox")
    picker.select_option("native-second")
    expect(dialog).to_contain_text("eval_lite_100@25/scored_fraction")
    journey.page.mouse.click(5, 5)
    expect(dialog).not_to_be_visible()
    picker.select_option("native-first")
    journey.page.keyboard.press("Escape")
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
