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

"""Tests for controller core data structures."""

import threading

import pytest

from fluster.rpc import cluster_pb2
from fluster.cluster.controller.state import ControllerEndpoint, ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import JobId, WorkerId


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def make_resource_spec():
    """Create a minimal ResourceSpec for testing."""

    def _make() -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=1, memory="1g", disk="10g")

    return _make


def test_controller_state_fifo_order(make_job_request):
    """Verify jobs are returned in FIFO order."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"), submitted_at_ms=100)
    job2 = ControllerJob(job_id=JobId("j2"), request=make_job_request("job2"), submitted_at_ms=200)
    state.add_job(job1)
    state.add_job(job2)

    # Jobs should be popped in the order they were added
    popped1 = state.pop_next_pending()
    assert popped1 is not None
    assert popped1.job_id == "j1"

    popped2 = state.pop_next_pending()
    assert popped2 is not None
    assert popped2.job_id == "j2"

    # Queue should be empty now
    assert state.pop_next_pending() is None


def test_controller_state_skip_non_pending(make_job_request):
    """Verify pop_next_pending skips jobs that are not in PENDING state."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))
    job1.state = cluster_pb2.JOB_STATE_RUNNING  # Already started
    job2 = ControllerJob(job_id=JobId("j2"), request=make_job_request("job2"))
    state.add_job(job1)
    state.add_job(job2)

    # Should skip j1 since it's not PENDING
    popped = state.pop_next_pending()
    assert popped is not None
    assert popped.job_id == "j2"

    # Queue should be empty now
    assert state.pop_next_pending() is None


def test_controller_state_worker_operations(make_resource_spec):
    """Test add/get/list workers."""
    state = ControllerState()
    worker1 = ControllerWorker(worker_id=WorkerId("w1"), address="host1:8080", resources=make_resource_spec())
    worker2 = ControllerWorker(worker_id=WorkerId("w2"), address="host2:8080", resources=make_resource_spec())

    # Add workers
    state.add_worker(worker1)
    state.add_worker(worker2)

    # Get individual worker
    retrieved = state.get_worker(WorkerId("w1"))
    assert retrieved is not None
    assert retrieved.address == "host1:8080"
    assert retrieved.healthy is True

    # Get all available workers
    available = state.get_available_workers()
    assert len(available) == 2
    assert {w.worker_id for w in available} == {"w1", "w2"}

    # Mark one worker unhealthy
    worker1.healthy = False
    available = state.get_available_workers()
    assert len(available) == 1
    assert available[0].worker_id == "w2"


def test_controller_state_gang_tracking(make_job_request):
    """Verify gang jobs are tracked correctly."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"), gang_id="gang1")
    job2 = ControllerJob(job_id=JobId("j2"), request=make_job_request("job2"), gang_id="gang1")
    job3 = ControllerJob(job_id=JobId("j3"), request=make_job_request("job3"), gang_id="gang2")

    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    # Get jobs in gang1
    gang1_jobs = state.get_gang_jobs("gang1")
    assert len(gang1_jobs) == 2
    assert {j.job_id for j in gang1_jobs} == {"j1", "j2"}

    # Get jobs in gang2
    gang2_jobs = state.get_gang_jobs("gang2")
    assert len(gang2_jobs) == 1
    assert gang2_jobs[0].job_id == "j3"

    # Non-existent gang returns empty list
    assert state.get_gang_jobs("nonexistent") == []


def test_controller_state_thread_safety(make_job_request):
    """Verify concurrent access doesn't corrupt state."""
    state = ControllerState()
    num_threads = 10
    jobs_per_thread = 50
    barrier = threading.Barrier(num_threads)
    errors = []

    def add_jobs(thread_id: int):
        try:
            # Wait for all threads to be ready
            barrier.wait()

            # Add jobs
            for i in range(jobs_per_thread):
                job_id = f"t{thread_id}_j{i}"
                job = ControllerJob(job_id=JobId(job_id), request=make_job_request(f"job-{job_id}"))
                state.add_job(job)
        except Exception as e:
            errors.append(e)

    # Start threads
    threads = [threading.Thread(target=add_jobs, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Check no errors occurred
    assert not errors, f"Errors during concurrent execution: {errors}"

    # Verify all jobs were added
    expected_count = num_threads * jobs_per_thread
    popped_count = 0
    while state.pop_next_pending() is not None:
        popped_count += 1

    assert popped_count == expected_count, f"Expected {expected_count} jobs, got {popped_count}"


def test_controller_state_multiple_gangs(make_job_request):
    """Test tracking multiple gangs simultaneously."""
    state = ControllerState()

    # Create multiple gangs with different sizes
    for gang_num in range(5):
        gang_id = f"gang{gang_num}"
        for job_num in range(gang_num + 1):  # gang0 has 1 job, gang1 has 2, etc.
            job_id = JobId(f"g{gang_num}_j{job_num}")
            job = ControllerJob(job_id=job_id, request=make_job_request(f"job-{job_id}"), gang_id=gang_id)
            state.add_job(job)

    # Verify each gang has correct number of jobs
    for gang_num in range(5):
        gang_id = f"gang{gang_num}"
        gang_jobs = state.get_gang_jobs(gang_id)
        expected_count = gang_num + 1
        assert (
            len(gang_jobs) == expected_count
        ), f"Gang {gang_id} should have {expected_count} jobs, got {len(gang_jobs)}"


def test_controller_state_requeue_job(make_job_request):
    """Test that jobs can be re-queued by calling add_job again."""
    state = ControllerState()
    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    # Add job
    state.add_job(job)

    # Pop it
    popped = state.pop_next_pending()
    assert popped is not None
    assert popped.job_id == "j1"

    # Queue should be empty
    assert state.pop_next_pending() is None

    # Re-queue the same job
    state.add_job(job)

    # Should be available again
    popped = state.pop_next_pending()
    assert popped is not None
    assert popped.job_id == "j1"


def test_controller_state_action_log():
    """Test action log functionality."""
    state = ControllerState()

    # Initially empty
    assert state.get_recent_actions() == []

    # Log some actions
    state.log_action("job_submitted", job_id=JobId("j1"), details="Test job")
    state.log_action("worker_registered", worker_id=WorkerId("w1"))
    state.log_action("job_started", job_id=JobId("j1"), worker_id=WorkerId("w1"))

    # Should have 3 actions
    actions = state.get_recent_actions()
    assert len(actions) == 3

    # Check order (oldest first)
    assert actions[0].action == "job_submitted"
    assert actions[0].job_id == "j1"
    assert actions[0].details == "Test job"
    assert actions[1].action == "worker_registered"
    assert actions[1].worker_id == "w1"
    assert actions[2].action == "job_started"

    # Check timestamps are set
    for action in actions:
        assert action.timestamp_ms > 0


def test_controller_state_action_log_limit():
    """Test action log respects limit parameter."""
    state = ControllerState()

    # Log many actions
    for i in range(10):
        state.log_action(f"action_{i}")

    # Get with limit
    actions = state.get_recent_actions(limit=3)
    assert len(actions) == 3

    # Should be most recent 3
    assert actions[0].action == "action_7"
    assert actions[1].action == "action_8"
    assert actions[2].action == "action_9"


def test_controller_state_action_log_bounded():
    """Test action log deque is bounded to 100 entries."""
    state = ControllerState()

    # Log more than 100 actions
    for i in range(150):
        state.log_action(f"action_{i}")

    # Should only have 100
    actions = state.get_recent_actions(limit=200)
    assert len(actions) == 100

    # Oldest should be action_50 (first 50 were evicted)
    assert actions[0].action == "action_50"
    assert actions[-1].action == "action_149"


# =============================================================================
# Hierarchical Job Tests
# =============================================================================


def test_controller_state_get_children_returns_direct_children(make_job_request):
    """Verify get_children returns only direct children of a parent job."""
    state = ControllerState()

    # Create parent job
    parent = ControllerJob(job_id=JobId("parent"), request=make_job_request("parent"))
    state.add_job(parent)

    # Create child jobs
    child1 = ControllerJob(
        job_id=JobId("child1"),
        request=make_job_request("child1"),
        parent_job_id=JobId("parent"),
    )
    child2 = ControllerJob(
        job_id=JobId("child2"),
        request=make_job_request("child2"),
        parent_job_id=JobId("parent"),
    )
    state.add_job(child1)
    state.add_job(child2)

    # Create an unrelated job with no parent
    unrelated = ControllerJob(job_id=JobId("unrelated"), request=make_job_request("unrelated"))
    state.add_job(unrelated)

    # Get children of parent
    children = state.get_children(JobId("parent"))
    assert len(children) == 2
    assert {c.job_id for c in children} == {"child1", "child2"}


def test_controller_state_get_children_returns_empty_for_no_children(make_job_request):
    """Verify get_children returns empty list when job has no children."""
    state = ControllerState()

    # Create a job with no children
    job = ControllerJob(job_id=JobId("lonely"), request=make_job_request("lonely"))
    state.add_job(job)

    children = state.get_children(JobId("lonely"))
    assert children == []


def test_controller_state_get_children_only_returns_direct_not_grandchildren(make_job_request):
    """Verify get_children only returns direct children, not grandchildren."""
    state = ControllerState()

    # Create a 3-level hierarchy: grandparent -> parent -> child
    grandparent = ControllerJob(job_id=JobId("grandparent"), request=make_job_request("grandparent"))
    parent = ControllerJob(
        job_id=JobId("parent"),
        request=make_job_request("parent"),
        parent_job_id=JobId("grandparent"),
    )
    child = ControllerJob(
        job_id=JobId("child"),
        request=make_job_request("child"),
        parent_job_id=JobId("parent"),
    )
    state.add_job(grandparent)
    state.add_job(parent)
    state.add_job(child)

    # get_children of grandparent should only return parent, not grandchild
    children = state.get_children(JobId("grandparent"))
    assert len(children) == 1
    assert children[0].job_id == "parent"


# =============================================================================
# Endpoint Registry Tests
# =============================================================================


def test_add_and_lookup_endpoint():
    """Test basic endpoint registration and lookup."""
    state = ControllerState()

    # Create a running job first
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    # Register endpoint with prefixed name
    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Lookup by full prefixed name
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 1
    assert results[0].address == "10.0.0.1:8080"
    assert results[0].endpoint_id == "ep-1"


def test_endpoint_not_returned_for_non_running_job():
    """Test that endpoints for non-RUNNING jobs are filtered out."""
    state = ControllerState()

    # Create a completed job
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Should not return endpoint because job is not running
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0


def test_transition_job_to_terminal_removes_endpoints():
    """Test that transition_job removes endpoints when job reaches terminal state."""
    state = ControllerState()

    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Verify endpoint is visible
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 1

    # Transition to terminal state via transition_job
    from fluster.cluster.controller.job import TransitionResult

    result, removed = state.transition_job(JobId("ns-1"), cluster_pb2.JOB_STATE_SUCCEEDED, now_ms=1000)

    assert result == TransitionResult.COMPLETE
    assert len(removed) == 1
    assert removed[0].endpoint_id == "ep-1"

    # Endpoint should be gone
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0


def test_namespace_isolation_via_prefix():
    """Test that namespace isolation works via name prefixing."""
    state = ControllerState()

    job1 = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    job2 = ControllerJob(
        job_id=JobId("ns-2"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job1)
    state.add_job(job2)

    # Same actor name, different namespace prefixes
    ep1 = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="ns-2/actor",
        address="10.0.0.2:8080",
        job_id=JobId("ns-2"),
    )
    state.add_endpoint(ep1)
    state.add_endpoint(ep2)

    # Each namespace prefix only sees its own endpoint
    results_ns1 = state.lookup_endpoints("ns-1/actor")
    assert len(results_ns1) == 1
    assert results_ns1[0].address == "10.0.0.1:8080"

    results_ns2 = state.lookup_endpoints("ns-2/actor")
    assert len(results_ns2) == 1
    assert results_ns2[0].address == "10.0.0.2:8080"


def test_list_endpoints_by_prefix():
    """Test prefix-based endpoint listing."""
    state = ControllerState()

    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    # Register multiple endpoints with shared prefix
    ep1 = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/inference/model-a",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="ns-1/inference/model-b",
        address="10.0.0.2:8080",
        job_id=JobId("ns-1"),
    )
    ep3 = ControllerEndpoint(
        endpoint_id="ep-3",
        name="ns-1/training/main",
        address="10.0.0.3:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep1)
    state.add_endpoint(ep2)
    state.add_endpoint(ep3)

    # List by prefix (includes namespace)
    results = state.list_endpoints_by_prefix("ns-1/inference/")
    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"ns-1/inference/model-a", "ns-1/inference/model-b"}

    results_training = state.list_endpoints_by_prefix("ns-1/training/")
    assert len(results_training) == 1
    assert results_training[0].name == "ns-1/training/main"

    # Listing all in namespace
    results_all = state.list_endpoints_by_prefix("ns-1/")
    assert len(results_all) == 3


def test_multiple_endpoints_for_same_name():
    """Test that multiple endpoints can be registered for the same name."""
    state = ControllerState()

    job1 = ControllerJob(
        job_id=JobId("ns-1/worker-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    job2 = ControllerJob(
        job_id=JobId("ns-1/worker-2"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job1)
    state.add_job(job2)

    # Register multiple endpoints with same name (for load balancing)
    ep1 = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/inference",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1/worker-1"),
    )
    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="ns-1/inference",
        address="10.0.0.2:8080",
        job_id=JobId("ns-1/worker-2"),
    )
    state.add_endpoint(ep1)
    state.add_endpoint(ep2)

    results = state.lookup_endpoints("ns-1/inference")
    assert len(results) == 2
    addresses = {r.address for r in results}
    assert addresses == {"10.0.0.1:8080", "10.0.0.2:8080"}


def test_remove_endpoint_by_id():
    """Test explicit endpoint removal by ID."""
    state = ControllerState()

    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Remove by ID
    removed = state.remove_endpoint("ep-1")
    assert removed is not None
    assert removed.endpoint_id == "ep-1"

    # Should no longer be found
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0

    # Removing again should be idempotent
    removed_again = state.remove_endpoint("ep-1")
    assert removed_again is None


def test_pending_job_endpoints_not_returned():
    """Test that endpoints for PENDING jobs are not returned."""
    state = ControllerState()

    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_PENDING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Should not return because job is pending
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0

    # Transition to running
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # Now should be visible
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 1


# =============================================================================
# transition_job Tests
# =============================================================================


def test_transition_job_retry_requeues_and_unassigns(make_job_request, make_resource_spec):
    """Test that transition_job re-queues job and unassigns worker on SHOULD_RETRY."""
    from fluster.cluster.controller.job import TransitionResult

    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
    )
    state.add_worker(worker)

    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_failure=2,
    )
    state.add_job(job)

    # Dispatch job
    job.mark_dispatched(WorkerId("w1"), now_ms=1000)
    state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))
    state.remove_from_queue(JobId("j1"))

    # Verify job not in queue and worker has it
    pending = state.peek_pending_jobs()
    assert len(pending) == 0
    assert JobId("j1") in worker.running_jobs

    # Transition to FAILED (with retries available)
    result, removed = state.transition_job(
        JobId("j1"),
        cluster_pb2.JOB_STATE_FAILED,
        now_ms=2000,
        error="Test failure",
    )

    assert result == TransitionResult.SHOULD_RETRY
    assert removed == []  # No endpoints to remove

    # Job should be re-queued and unassigned from worker
    pending = state.peek_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == JobId("j1")
    assert JobId("j1") not in worker.running_jobs


def test_transition_job_removes_from_queue_on_unschedulable(make_job_request):
    """Test that transition_job removes job from queue when UNSCHEDULABLE."""
    from fluster.cluster.controller.job import TransitionResult

    state = ControllerState()

    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
    )
    state.add_job(job)

    # Verify job is in queue
    pending = state.peek_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == JobId("j1")

    # Transition to UNSCHEDULABLE
    result, removed = state.transition_job(
        JobId("j1"),
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        now_ms=2000,
    )

    assert result == TransitionResult.COMPLETE
    assert removed == []  # No endpoints registered

    # Job should be removed from queue
    pending = state.peek_pending_jobs()
    assert len(pending) == 0

    # Job should be in terminal state
    retrieved_job = state.get_job(JobId("j1"))
    assert retrieved_job is not None
    assert retrieved_job.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE
    assert retrieved_job.is_finished()


# =============================================================================
# Job Lifecycle Integration Tests
# =============================================================================


def test_job_success_cleans_up_endpoints(make_job_request, make_resource_spec):
    """When job succeeds, endpoints are cleaned up and worker is unassigned."""
    from fluster.cluster.controller.job import TransitionResult

    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
    )
    state.add_worker(worker)

    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))
    state.add_job(job)

    job.mark_dispatched(WorkerId("w1"), now_ms=1000)
    state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep)

    # Verify endpoint visible while running
    results = state.lookup_endpoints("j1/my-actor")
    assert len(results) == 1

    # Job succeeds via transition_job
    result, removed = state.transition_job(JobId("j1"), cluster_pb2.JOB_STATE_SUCCEEDED, now_ms=2000)

    assert result == TransitionResult.COMPLETE
    assert len(removed) == 1

    # Verify cleanup
    assert state.lookup_endpoints("j1/my-actor") == []
    assert JobId("j1") not in worker.running_jobs


def test_job_failure_with_retry_unassigns_worker(make_job_request, make_resource_spec):
    """Job failure with retry unassigns worker and resets job to PENDING."""
    from fluster.cluster.controller.job import TransitionResult

    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
    )
    state.add_worker(worker)

    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_failure=1,
    )
    state.add_job(job)

    # First attempt
    job.mark_dispatched(WorkerId("w1"), now_ms=1000)
    state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))

    # First failure - should retry
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=2000)
    assert result == TransitionResult.SHOULD_RETRY

    # Unassign from worker (simulating what controller does)
    state.unassign_job_from_worker(WorkerId("w1"), JobId("j1"))

    # Worker unassigned and job reset to PENDING for retry
    assert JobId("j1") not in worker.running_jobs
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.worker_id is None


@pytest.mark.parametrize(
    "terminal_state",
    [
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
    ],
)
def test_terminal_states_clean_up_endpoints(make_job_request, terminal_state):
    """All terminal states clean up job endpoints."""
    state = ControllerState()

    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep)

    # Transition to terminal state via transition_job
    state.transition_job(JobId("j1"), terminal_state, now_ms=1000, error="terminated")

    assert state.lookup_endpoints("j1/actor") == []


def test_worker_timeout_job_cleanup(make_job_request, make_resource_spec):
    """Worker timeout triggers proper job cleanup including endpoints."""
    from fluster.cluster.controller.job import TransitionResult

    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
        last_heartbeat_ms=0,
    )
    state.add_worker(worker)

    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_preemption=0,
    )
    state.add_job(job)

    job.mark_dispatched(WorkerId("w1"), now_ms=1000)
    state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep)

    # Simulate worker timeout via transition_job
    result, removed = state.transition_job(
        JobId("j1"),
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        now_ms=60000,
        is_worker_failure=True,
        error="Worker timed out",
    )
    assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
    assert len(removed) == 1

    assert state.lookup_endpoints("j1/actor") == []
    assert JobId("j1") not in worker.running_jobs
