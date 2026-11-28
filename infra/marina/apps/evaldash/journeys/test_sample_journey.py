# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from urllib.parse import parse_qs, urlsplit

from marina.journeys import Journey
from playwright.sync_api import expect


def test_incorrect_sample_can_be_opened_and_shared_with_its_grading_evidence(journey: Journey) -> None:
    journey.visit("/inspect")
    page = journey.page
    expect(page).to_have_url(re.compile(r"/runs\?.*\bstatus=succeeded"))
    search = page.get_by_role("searchbox", name="Search runs", exact=True)
    search.fill("native-fixture-mmlu")
    search.press("Enter")
    page.get_by_role("link", name="Samples for mmlu", exact=True).click()
    expect(page).to_have_url(re.compile(r"/runs/native-fixture-mmlu/samples(?:\?|$)"))
    expect(page.get_by_role("button", name=re.compile(r"^all\s*\(3\)$", re.IGNORECASE))).to_be_visible()
    page.get_by_role("button", name=re.compile(r"^incorrect\s*\(1\)$", re.IGNORECASE)).click()
    journey.sees("1 of 1").sees("doc 1")
    journey.sees("Which option is correct for question 1?").sees("score 0").sees("picked A, gold B")
    expect(page.get_by_text("model pick", exact=True)).to_be_visible()
    expect(page.get_by_text("target", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_disabled()
    journey.shoot("incorrect-sample")

    query = parse_qs(urlsplit(page.url).query)
    assert query["task"] == ["mmlu"]
    assert query["filter"] == ["incorrect"]
    page.reload(wait_until="domcontentloaded")
    journey.sees("Which option is correct for question 1?").sees("score 0")
