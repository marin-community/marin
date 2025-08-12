import typing
import warnings
from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass
class ScheduleStep(typing.Generic[T]):
    start: int
    value: T


IntSchedule = list[ScheduleStep[int]]

BIG_INT = 2**63 - 1


def value_at_step(schedule_or_t: Sequence[ScheduleStep[T]] | T, step: int) -> T:
    """
    Given a schedule or a single value, return the value at the given step.

    """
    if not isinstance(schedule_or_t, Sequence) or (schedule_or_t and not isinstance(schedule_or_t[0], ScheduleStep)):
        return schedule_or_t  # type: ignore

    for i, step_ in enumerate(schedule_or_t):
        # we use start now
        if step >= step_.start:
            return step_.value

    raise ValueError(f"Step {step} isn't after any of the schedule steps.")


def validate_schedule_sorted(schedule: Sequence[ScheduleStep[T]]):
    if len(schedule) == 0:
        raise ValueError("Schedule must have at least one step.")
    if schedule[0].start != 0:
        raise ValueError("Schedule must start at step 0.")

    for i in range(1, len(schedule)):
        # the last step can be -1 to indicate that the value should be held indefinitely
        if schedule[i].start < schedule[i - 1].start:
            raise ValueError(f"Schedule is not sorted at index {i}")


def distinct_values(schedule: Sequence[ScheduleStep[T]] | T) -> set[T]:
    if not isinstance(schedule, Sequence) or (schedule and not isinstance(schedule[0], ScheduleStep)):
        return {schedule}  # type: ignore
    return set(step.value for step in schedule)


@dataclass
class BatchSegment:
    start: int  # The training step at which this batch size starts.
    until: int  # The training step at which this batch size stops. If -1, the segment is open-ended.
    value: int  # The batch size for steps before 'until'.
    offset: int  # The cumulative number of data points processed up to the start of this segment.


class BatchSchedule:
    """
    A class that schedules the batch size for each training step. The schedule can be either a fixed batch size or a
    schedule of batch sizes over time. The schedule is defined by a list of steps, each of which has a batch size and
    a step at which that batch size starts.

    This class also tracks cumulative data offsets and can return the "local" indices for a step and a sharding
    """

    def __init__(self, schedule: int | IntSchedule):
        self.schedule = schedule

        # Precompute the segments of the schedule.
        if isinstance(schedule, int):
            self.segments = [BatchSegment(0, BIG_INT, schedule, 0)]
        else:
            if len(schedule) == 0:
                raise ValueError("Batch schedule must have at least one step.")
            self.segments = []
            total_offset = 0
            for i, step in enumerate(schedule):
                start = step.start
                until = BIG_INT if i == len(schedule) - 1 else schedule[i + 1].start

                # Save the segment information.
                self.segments.append(BatchSegment(start, until, step.value, total_offset))

                # Update the offset and the next segment start, if the segment is bounded.
                if until < BIG_INT:
                    # (until - start) steps each process 'value' data points.
                    total_offset += (until - start) * step.value

    def batch_size_at_step(self, step: int) -> int:
        """
        Return the batch size (number of samples) at the given training step.
        """
        for seg in self.segments:
            if seg.start <= step < seg.until:
                return seg.value
        warnings.warn(f"Step {step} is beyond the last defined segment. Using the last segment's batch size.")
        return self.segments[-1].value

    def global_data_offset_by_step(self, step: int) -> int:
        """
        Return the number of data points that have been processed up to the given training step.
        (That is, the starting index of the data for the current batch.)
        """
        for seg in self.segments:
            if seg.start <= step < seg.until:
                # Within the current segment, add the number of steps passed times the batch size.
                return seg.offset + (step - seg.start) * seg.value
        # If step is beyond all segments, assume it continues in the last segment.
        last = self.segments[-1]
        return last.offset + (step - last.start) * last.value

    def find_step_containing_offset(self, offset: int) -> int:
        """
        Find the step that contains the given global data offset.
        """
        for seg in self.segments:
            if seg.offset <= offset < seg.offset + (seg.until - seg.start) * seg.value:
                return seg.start + (offset - seg.offset) // seg.value
        raise ValueError(f"Offset {offset} is beyond the last defined segment.")

    def batch_indices_at_step(self, bn):
        """
        Return the indices for the batch at the given training step.
        """
        for seg in self.segments:
            if seg.start <= bn < seg.until:
                base = seg.offset + (bn - seg.start) * seg.value
                return range(base, base + seg.value)

        # If step is beyond all segments, assume it continues in the last segment.
        last = self.segments[-1]
        base = last.offset + (bn - last.start) * last.value
        return range(base, base + last.value)

    def unique_batch_sizes(self):
        return set(seg.value for seg in self.segments)
