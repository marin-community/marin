from contextlib import contextmanager
import os
from collections.abc import Iterator

from loguru import logger
import torch
import torch.distributed


PROFILE_UPDATE_INDEX = 1  # Let one training call warm the kernels before capture.


class Profiler:
    """Capture one forward micro-batch on the selected ranks."""

    def __init__(self, config):
        if not config.save_path:
            raise ValueError("Profiler save_path is required when profiling is enabled")
        self.save_path = config.save_path
        self.rank = torch.distributed.get_rank()
        self.update_index = -1
        self.captured = False
        self.prof = None
        if self.rank in config.ranks:
            logger.info(f"[Profiler] Profiler init for rank {self.rank}")
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False)

    def begin_update(self) -> None:
        self.update_index += 1

    def for_mini_batch(self, index: int) -> "Profiler | None":
        if self.prof is not None and self.update_index == PROFILE_UPDATE_INDEX and index == 0 and not self.captured:
            return self
        return None

    @contextmanager
    def capture_forward(self) -> Iterator[None]:
        if self.captured:
            yield
            return
        assert self.prof is not None
        logger.info(f"[Profiler] Capturing first forward micro-batch of update {self.update_index} on rank {self.rank}")
        self.prof.start()
        try:
            yield
        finally:
            self.prof.stop()
            self.captured = True

    def save(self):
        if not self.captured or self.prof is None:
            return
        os.makedirs(self.save_path, exist_ok=True)
        table_path = os.path.join(self.save_path, f"prof_rank_{self.rank}.txt")
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        with open(table_path, "w", encoding="utf-8") as table_file:
            table_file.write(self.prof.key_averages().table(sort_by=sort_key, row_limit=50))
        logger.info(f"[Profiler] Saved operator table to {table_path}")
        self.prof = None


class CudaTimer:
    def __init__(self, device):
        self.device = device

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)  # Calculate the elapsed time
