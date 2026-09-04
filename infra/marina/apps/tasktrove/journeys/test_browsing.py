# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader arrives at the sources, opens the audited sample, and looks for a task.

Everything here reads from the corpus files under the data root; the datasets-server
fetches that open one task are left to the person, since a journey should not depend on
Hugging Face answering.
"""

import re
from typing import Any, cast

from marina.journeys import Journey


def test_sources_are_listed_and_filterable(journey: Journey) -> None:
    journey.visit("/").shoot("sources")
    assert "input:Filter sources by name or description" in journey.offers()
    journey.fill("Filter sources by name or description", "nl2bash")
    journey.sees(re.compile(r"\d+ sources, [\d,]+ tasks"))
    assert "DCAgent2__nl2bash-tasks-cleaned-oracle-v2" in journey.reads()
    journey.shoot("sources-filtered")


def test_audited_sample_opens_from_the_shell_nav(journey: Journey) -> None:
    journey.visit("/").click("Audited sample").shoot("sampled")
    journey.widths("sampled")
    assert journey.page.url.endswith("/tasktrove/sampled")


def test_shell_knows_the_app_and_the_caller(journey: Journey) -> None:
    journey.visit("/")
    apps = cast(dict[str, Any], journey.api("/api/marina/apps"))
    assert any(app["name"] == "tasktrove" for app in apps["apps"])
    assert journey.api("/api/marina/me") == {"user": "anonymous", "role": "admin"}
