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

"""Job execution context module."""

from fray.job.context import (
    ActorHandle,
    ActorMethod,
    ContextConfig,
    JobContext,
    SimpleActor,
    create_job_ctx,
    fray_default_job_ctx,
    get_default_job_ctx,
)
from fray.job.ray_ctx import RayContext
from fray.job.sync_ctx import SyncActorHandle, SyncActorMethod, SyncContext, SyncFuture
from fray.job.thread_ctx import GeneratorFuture, ThreadActorHandle, ThreadActorMethod, ThreadContext, ThreadFuture
from fray.fray_rpc import RustyContext

# Alias for backwards compatibility
RpcContext = RustyContext

__all__ = [
    "ActorHandle",
    "ActorMethod",
    "ContextConfig",
    "GeneratorFuture",
    "JobContext",
    "RayContext",
    "RpcContext",
    "RustyContext",
    "SimpleActor",
    "SyncActorHandle",
    "SyncActorMethod",
    "SyncContext",
    "SyncFuture",
    "ThreadActorHandle",
    "ThreadActorMethod",
    "ThreadContext",
    "ThreadFuture",
    "create_job_ctx",
    "fray_default_job_ctx",
    "get_default_job_ctx",
]
