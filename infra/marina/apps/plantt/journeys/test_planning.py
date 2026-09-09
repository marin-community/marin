# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import pytest


@pytest.mark.timeout(120)
def test_create_example_plan(journey) -> None:
    journey.visit("/").sees("Database-backed plans").click("Use example")
    journey.sees("Example Project Plan").sees("Compute capacity").sees("Launch")
    journey.shoot("example-plan")
