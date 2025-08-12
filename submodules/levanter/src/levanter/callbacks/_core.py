import abc
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from jaxtyping import PyTree

from levanter.trainer_state import InsideJitInfo, TrainerState


M = TypeVar("M")  # Model
M_con = TypeVar("M_con", bound=PyTree, contravariant=True)
S = TypeVar("S", bound=TrainerState)
CBInfo = TypeVar("CBInfo")


@dataclass
class StepInfo(Generic[S]):
    """
    Information about a step that was just completed. This includes the trainer state, the loss, and the duration of the
    step.

    Note that the step is 0-indexed, so if you want the next step, use `next_step`.
    """

    state: S
    loss: float
    step_duration: float

    model = property(lambda self: self.state.model)
    opt_state = property(lambda self: self.state.opt_state)
    eval_model = property(lambda self: self.state.eval_model)

    step = property(lambda self: int(self.state.step) - 1)
    """
    The step that was just completed. If you want the next step, use `next_step`.
    """

    next_step = property(lambda self: int(self.state.step))


class Callback(ABC, Generic[S]):
    """
    A callback that can be called at the end of a step. This is useful for logging, profiling, and other side effects.
    """

    @abc.abstractmethod
    def on_step(self, info: StepInfo[S], force: bool = False):
        ...


class LambdaCallback(Callback[S]):
    def __init__(self, fn: Callable[[StepInfo[S]], Any]):
        self.fn = fn

    def on_step(self, info: StepInfo[S], force: bool = False):
        self.fn(info)


class JitCallback(ABC, Generic[S, M, CBInfo]):
    """
    A callback that gets called in two phases: inside the step (inside jit), and after the step (outside jit).
    You have access to [levanter.trainer_state.InsideJitInfo][] inside the step, which allows you to track gradients,
    updates, etc.

    Note that JitCallbacks tend to be quite heavy, and you should try to line up  all jit callbacks to
    proc on the same steps.

    The basic flow of a JitCallback is as follows:
    1. `inside_step` is called inside the JIT-compiled function. This is where you can compute gradients, updates,
       and other information that you want to track. You should return a PyTree of information that you want to log or use later.
    2. The returned information from `inside_step` is passed to `on_step` after the JIT-compiled function has completed.
    """

    @abc.abstractmethod
    def inside_step(self, state: S, inside_info: InsideJitInfo[M]) -> CBInfo:
        """
        This function is called inside the JIT-compiled function. You have access to the `inside_info` which contains
        information about the gradients, updates, and other information that was computed during the step.
        Args:
            state:
            inside_info:

        Returns:

        """
        ...

    @abc.abstractmethod
    def on_step(self, step_info: S, cb_info: CBInfo):
        """
        This function is called after the JIT-compiled function has completed. You have access to the `step_info`
        which contains information about the step that just completed, as well as the `cb_info` which is whatever
        was returned from `inside_step`.
        """
        ...
