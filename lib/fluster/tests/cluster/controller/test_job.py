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

"""Tests for Job state transitions."""

import pytest

from fluster.cluster.controller.job import Job, TransitionResult, handle_gang_failure
from fluster.cluster.types import JobId, WorkerId
from fluster.rpc import cluster_pb2


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


class TestJobTransitions:
    """Tests for Job.transition() state machine."""

    def test_transition_to_succeeded(self, make_job_request):
        """Job transitions to SUCCEEDED with exit code."""
        job = Job(job_id=JobId("test"), request=make_job_request())
        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        result = job.transition(cluster_pb2.JOB_STATE_SUCCEEDED, now_ms=2000, exit_code=0)

        assert result == TransitionResult.COMPLETE
        assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED
        assert job.finished_at_ms == 2000
        assert job.exit_code == 0

    def test_transition_to_killed(self, make_job_request):
        """Job transitions to KILLED with error message."""
        job = Job(job_id=JobId("test"), request=make_job_request())
        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        result = job.transition(
            cluster_pb2.JOB_STATE_KILLED,
            now_ms=2000,
            error="Terminated by user",
        )

        assert result == TransitionResult.COMPLETE
        assert job.state == cluster_pb2.JOB_STATE_KILLED
        assert job.finished_at_ms == 2000
        assert job.error == "Terminated by user"

    def test_transition_to_unschedulable(self, make_job_request):
        """Job transitions to UNSCHEDULABLE with timeout error."""
        request = make_job_request()
        request.scheduling_timeout_seconds = 300
        job = Job(job_id=JobId("test"), request=request)

        result = job.transition(cluster_pb2.JOB_STATE_UNSCHEDULABLE, now_ms=2000)

        assert result == TransitionResult.COMPLETE
        assert job.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE
        assert job.finished_at_ms == 2000
        assert job.error is not None
        assert "300s" in job.error


class TestJobFailureRetry:
    """Tests for Job failure and retry logic."""

    def test_failure_with_retry_available(self, make_job_request):
        """Job failure returns RETRY when retries available."""
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            max_retries_failure=2,
        )
        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        result = job.transition(
            cluster_pb2.JOB_STATE_FAILED,
            now_ms=2000,
            error="oops",
        )

        assert result == TransitionResult.SHOULD_RETRY
        assert job.state == cluster_pb2.JOB_STATE_PENDING
        assert job.failure_count == 1
        assert job.worker_id is None
        assert job.started_at_ms is None
        assert job.error is None

    def test_failure_exceeds_retry_limit(self, make_job_request):
        """Job failure returns EXCEEDED_RETRY_LIMIT when limit exceeded."""
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            max_retries_failure=1,
        )
        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        # First failure - retry
        result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=2000)
        assert result == TransitionResult.SHOULD_RETRY
        assert job.failure_count == 1

        # Dispatch again
        job.mark_dispatched(WorkerId("w2"), now_ms=3000)

        # Second failure - no more retries
        result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=4000, error="final error")
        assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
        assert job.state == cluster_pb2.JOB_STATE_FAILED
        assert job.failure_count == 2
        assert job.error == "final error"
        assert job.finished_at_ms == 4000

    def test_worker_failure_uses_preemption_count(self, make_job_request):
        """Worker failure increments preemption_count, not failure_count."""
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            max_retries_preemption=1,
        )
        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        # First preemption - retry
        result = job.transition(
            cluster_pb2.JOB_STATE_WORKER_FAILED,
            now_ms=2000,
            is_worker_failure=True,
        )
        assert result == TransitionResult.SHOULD_RETRY
        assert job.preemption_count == 1
        assert job.failure_count == 0  # Separate counter

        # Dispatch again
        job.mark_dispatched(WorkerId("w2"), now_ms=3000)

        # Second preemption - no more retries
        result = job.transition(
            cluster_pb2.JOB_STATE_WORKER_FAILED,
            now_ms=4000,
            is_worker_failure=True,
        )
        assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
        assert job.state == cluster_pb2.JOB_STATE_WORKER_FAILED
        assert job.preemption_count == 2

    def test_default_zero_failure_retries(self, make_job_request):
        """Default max_retries_failure=0 means one try, no retries."""
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            # max_retries_failure defaults to 0
        )
        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        # First failure should NOT retry (count 1 > limit 0)
        result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=2000)
        assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
        assert job.failure_count == 1
        assert job.state == cluster_pb2.JOB_STATE_FAILED

    def test_high_preemption_retries_default(self, make_job_request):
        """Default max_retries_preemption=100 allows many retries."""
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            # max_retries_preemption defaults to 100
        )

        # Should be able to retry many times
        for i in range(100):
            job.mark_dispatched(WorkerId(f"w{i}"), now_ms=i * 1000)
            result = job.transition(
                cluster_pb2.JOB_STATE_WORKER_FAILED,
                now_ms=i * 1000 + 500,
                is_worker_failure=True,
            )
            assert result == TransitionResult.SHOULD_RETRY
            assert job.preemption_count == i + 1

        # 101st failure should not retry
        job.mark_dispatched(WorkerId("w100"), now_ms=100000)
        result = job.transition(
            cluster_pb2.JOB_STATE_WORKER_FAILED,
            now_ms=100500,
            is_worker_failure=True,
        )
        assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
        assert job.preemption_count == 101


class TestJobDispatch:
    """Tests for Job dispatch operations."""

    def test_mark_dispatched(self, make_job_request):
        """mark_dispatched sets correct state."""
        job = Job(job_id=JobId("test"), request=make_job_request())
        assert job.state == cluster_pb2.JOB_STATE_PENDING

        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        assert job.state == cluster_pb2.JOB_STATE_RUNNING
        assert job.worker_id == WorkerId("w1")
        assert job.started_at_ms == 1000

    def test_revert_dispatch(self, make_job_request):
        """revert_dispatch restores PENDING state."""
        job = Job(job_id=JobId("test"), request=make_job_request())
        job.mark_dispatched(WorkerId("w1"), now_ms=1000)

        job.revert_dispatch()

        assert job.state == cluster_pb2.JOB_STATE_PENDING
        assert job.worker_id is None
        assert job.started_at_ms is None


class TestJobQueries:
    """Tests for Job query methods."""

    def test_is_finished(self, make_job_request):
        """is_finished returns True for terminal states."""
        job = Job(job_id=JobId("test"), request=make_job_request())

        assert not job.is_finished()

        job.state = cluster_pb2.JOB_STATE_RUNNING
        assert not job.is_finished()

        for terminal_state in [
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
            cluster_pb2.JOB_STATE_WORKER_FAILED,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        ]:
            job.state = terminal_state
            assert job.is_finished()

    def test_can_retry_failure(self, make_job_request):
        """can_retry_failure checks failure_count vs max_retries_failure."""
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            max_retries_failure=2,
        )

        assert job.can_retry_failure()  # 0 < 2

        job.failure_count = 1
        assert job.can_retry_failure()  # 1 < 2

        job.failure_count = 2
        assert not job.can_retry_failure()  # 2 < 2 is False

    def test_can_retry_preemption(self, make_job_request):
        """can_retry_preemption checks preemption_count vs max_retries_preemption."""
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            max_retries_preemption=1,
        )

        assert job.can_retry_preemption()  # 0 < 1

        job.preemption_count = 1
        assert not job.can_retry_preemption()  # 1 < 1 is False


class TestGangFailure:
    """Tests for handle_gang_failure function."""

    def test_gang_all_or_nothing_retry(self, make_job_request):
        """Gang retry fails if any job has no retries left."""
        job1 = Job(
            job_id=JobId("j1"),
            request=make_job_request("job1"),
            gang_id="g1",
            max_retries_failure=1,
        )
        job2 = Job(
            job_id=JobId("j2"),
            request=make_job_request("job2"),
            gang_id="g1",
            max_retries_failure=0,  # No retries
        )

        # Mark both as running
        job1.state = cluster_pb2.JOB_STATE_RUNNING
        job2.state = cluster_pb2.JOB_STATE_RUNNING

        # Gang fails - j2 has 0 retries, so entire gang cannot retry
        retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=False, error="Gang g1 failed")
        assert retried == []

        # Both jobs should be marked KILLED
        assert job1.state == cluster_pb2.JOB_STATE_KILLED
        assert job2.state == cluster_pb2.JOB_STATE_KILLED

        # Failure counts should not be incremented since gang couldn't retry
        assert job1.failure_count == 0
        assert job2.failure_count == 0

    def test_gang_retry_success(self, make_job_request):
        """All jobs in gang are retried together."""
        job1 = Job(
            job_id=JobId("j1"),
            request=make_job_request("job1"),
            gang_id="g1",
            max_retries_preemption=2,
        )
        job2 = Job(
            job_id=JobId("j2"),
            request=make_job_request("job2"),
            gang_id="g1",
            max_retries_preemption=2,
        )

        # Mark both as running
        job1.state = cluster_pb2.JOB_STATE_RUNNING
        job2.state = cluster_pb2.JOB_STATE_RUNNING

        # Gang fails due to worker failure - both have retries left
        retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=True, error="Worker died")

        assert len(retried) == 2
        assert job1 in retried
        assert job2 in retried

        # Both jobs should be reset to PENDING
        assert job1.state == cluster_pb2.JOB_STATE_PENDING
        assert job2.state == cluster_pb2.JOB_STATE_PENDING

        # Both jobs should have preemption_count incremented
        assert job1.preemption_count == 1
        assert job2.preemption_count == 1

        # State should be cleared
        assert job1.worker_id is None
        assert job2.worker_id is None
        assert job1.error is None
        assert job2.error is None

    def test_gang_only_running_jobs_killed(self, make_job_request):
        """Only RUNNING jobs in gang marked KILLED on failure."""
        job1 = Job(
            job_id=JobId("j1"),
            request=make_job_request("job1"),
            gang_id="g1",
            max_retries_failure=0,
        )
        job2 = Job(
            job_id=JobId("j2"),
            request=make_job_request("job2"),
            gang_id="g1",
            max_retries_failure=0,
        )
        job3 = Job(
            job_id=JobId("j3"),
            request=make_job_request("job3"),
            gang_id="g1",
            max_retries_failure=0,
        )

        # Mix of states
        job1.state = cluster_pb2.JOB_STATE_RUNNING
        job2.state = cluster_pb2.JOB_STATE_PENDING  # Not started yet
        job3.state = cluster_pb2.JOB_STATE_RUNNING

        # Gang fails - no retries available
        retried = handle_gang_failure([job1, job2, job3], now_ms=1000, is_worker_failure=False, error="Gang failed")
        assert retried == []

        # Only running jobs should be marked KILLED
        assert job1.state == cluster_pb2.JOB_STATE_KILLED
        assert job2.state == cluster_pb2.JOB_STATE_PENDING  # Not running, so not killed
        assert job3.state == cluster_pb2.JOB_STATE_KILLED

    def test_gang_empty_list(self):
        """Empty job list returns empty list."""
        retried = handle_gang_failure([], now_ms=1000, is_worker_failure=True, error="error")
        assert retried == []

    def test_gang_tracks_correct_failure_type(self, make_job_request):
        """Gang retry increments correct counter based on failure type."""
        job1 = Job(
            job_id=JobId("j1"),
            request=make_job_request("job1"),
            gang_id="g1",
            max_retries_failure=2,
            max_retries_preemption=2,
        )
        job2 = Job(
            job_id=JobId("j2"),
            request=make_job_request("job2"),
            gang_id="g1",
            max_retries_failure=2,
            max_retries_preemption=2,
        )

        job1.state = cluster_pb2.JOB_STATE_RUNNING
        job2.state = cluster_pb2.JOB_STATE_RUNNING

        # Test worker failure (preemption)
        retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=True, error="Worker died")
        assert len(retried) == 2
        assert job1.preemption_count == 1
        assert job2.preemption_count == 1
        assert job1.failure_count == 0
        assert job2.failure_count == 0

        # Reset for next test
        job1.state = cluster_pb2.JOB_STATE_RUNNING
        job2.state = cluster_pb2.JOB_STATE_RUNNING

        # Test job failure (internal)
        retried = handle_gang_failure([job1, job2], now_ms=2000, is_worker_failure=False, error="Job crashed")
        assert len(retried) == 2
        assert job1.preemption_count == 1  # Unchanged
        assert job2.preemption_count == 1  # Unchanged
        assert job1.failure_count == 1
        assert job2.failure_count == 1
