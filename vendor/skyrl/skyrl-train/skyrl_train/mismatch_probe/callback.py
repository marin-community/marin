"""Schedule frozen-token probing after synchronous policy weight synchronization."""

from skyrl_train.callbacks.base import TrainerCallback, TrainerControl, TrainerState
from skyrl_train.mismatch_probe.collect import ProbeCollector, collect


class MismatchProbeCallback(TrainerCallback):
    """Collect requested relative updates from the run's starting policy."""

    error_behavior = "raise"

    def __init__(self, cfg):
        self.collector = ProbeCollector(cfg)

    async def on_train_begin_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        trainer = kwargs["trainer"]
        self.collector.starting_global_step = state.global_step
        control.step_limit = state.global_step + self.collector.updates
        await collect(self.collector, trainer, update=0)
        if self.collector.updates == 0:
            control.should_training_stop = True
        return control

    async def on_step_end_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        update = state.global_step - self.collector.starting_global_step
        if update <= self.collector.updates:
            await collect(self.collector, kwargs["trainer"], update=update)
        if update >= self.collector.updates:
            control.should_training_stop = True
        return control

    def on_train_end(self, state: TrainerState, control: TrainerControl, **kwargs):
        if self.collector.archive is not None:
            self.collector.archive.close()
            self.collector.archive = None
        return control
