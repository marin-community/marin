# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the Fray actor API contract."""

from fray.job import SimpleActor, create_job_ctx, fray_default_job_ctx


def test_thread_actor_supports_remote_and_options() -> None:
    """ThreadContext actor methods should look Ray-ish (remote/options)."""
    ctx = create_job_ctx(context_type="threadpool", max_workers=1)
    try:
        with fray_default_job_ctx(ctx):
            actor = ctx.create_actor(SimpleActor, 0, name="test_actor_api_contract", get_if_exists=True)

            assert ctx.get(actor.increment.remote()) == 1
            assert ctx.get(actor.increment.options(enable_task_events=False).remote(2)) == 3
            assert ctx.get(actor.get_value.remote()) == 3
    finally:
        ctx.executor.shutdown(wait=True)
