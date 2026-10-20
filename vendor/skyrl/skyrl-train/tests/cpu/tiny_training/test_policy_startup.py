"""Policy startup and one-update parity through the production actor group."""

import multiprocessing
import sys
from multiprocessing.context import BaseContext
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY
from tests.cpu.tiny_training import experiment
from tests.cpu.tiny_training.cpu_backend import CHECKPOINT_FILE_TEMPLATE, CausalLMPolicy
from tests.cpu.tiny_training.fixed_batch import fixed_training_batch, run_fixed_update
from tests.cpu.tiny_training.tiny_model import build_tiny_policy

pytestmark = pytest.mark.slow
RUN_TIMEOUT_SECONDS = 300


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_policy(tmp_path_factory.mktemp("tiny_policy"))


@pytest.fixture(scope="module")
def runs() -> BaseContext:
    if sys.platform == "darwin":
        # Forkserver preload clones initialized macOS proxy/preferences libraries; use a fresh interpreter.
        return multiprocessing.get_context("spawn")
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload([experiment.__name__])
    return context


def test_one_step_is_independent_of_micro_batch_size(runs: BaseContext, tmp_path: Path, tiny_policy: Path):
    batch = fixed_training_batch(str(tiny_policy))
    model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(tiny_policy, dtype=torch.float32))
    log_probs = model(batch["sequences"], num_actions=4, attention_mask=batch["attention_mask"])
    ratios = (log_probs - batch["action_log_probs"]).exp()
    loss = (
        -(ratios * batch["advantages"] * batch["correction_weights"] * batch["loss_mask"]).sum()
        / batch["loss_mask"].sum()
    )
    loss.backward()
    reference_norm = torch.linalg.vector_norm(
        torch.stack([parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None])
    ).item()
    roots = [tmp_path / "micro_8", tmp_path / "micro_4"]
    for root, micro_batch_size in zip(roots, [8, 4], strict=True):
        run = runs.Process(target=run_fixed_update, args=(root, tiny_policy, micro_batch_size))
        run.start()
        run.join(RUN_TIMEOUT_SECONDS)
        if run.exitcode is None:
            run.kill()
            run.join()
            pytest.fail(f"the update did not finish within {RUN_TIMEOUT_SECONDS} seconds")
        assert run.exitcode == 0
    norms = [torch.load(root / "train_status.pt", weights_only=False)["raw_grad_norm"] for root in roots]
    for norm in norms:
        torch.testing.assert_close(norm, reference_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(norms[0], norms[1], rtol=1e-5, atol=1e-6)
    for rank in range(2):
        states = [
            torch.load(
                root
                / "ckpts/global_step_1"
                / POLICY_CHECKPOINT_SUBDIRECTORY
                / CHECKPOINT_FILE_TEMPLATE.format(rank=rank),
                weights_only=False,
            )
            for root in roots
        ]
        for key, tensor in states[0]["model"].items():
            torch.testing.assert_close(tensor, states[1]["model"][key], rtol=1e-5, atol=1e-6, msg=key)
