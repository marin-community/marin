# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget spend follows publicly visible Task lifecycle state."""


def test_budget_spend_tracks_running_tasks_and_ignores_pending_or_resource_less_work(journey):
    user = "budget-user"
    journey.set_budget(user)
    job = journey.submit("spend", user=user, tasks=2)

    assert journey.budget_spent(user) == 0

    journey.settle()
    assert journey.budget_spent(user) == 12

    journey.succeed_all(job)
    journey.settle()
    assert journey.budget_spent(user) == 0

    resource_less = journey.submit("resource-less", user=user, include_resources=False)
    journey.settle()

    assert journey.task(resource_less[0]).current_attempt_id == 0
    assert journey.budget_spent(user) == 0
