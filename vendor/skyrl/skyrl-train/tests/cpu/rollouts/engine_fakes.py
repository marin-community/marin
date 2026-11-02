from dataclasses import dataclass, field, replace

import numpy as np
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, PackageSpec
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutLease
from skyrl_train.trajectory_runners.types import TokenProvenance


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, **template_options):
        return [1, 2]


class InferenceClient:
    async def generate(self, request):
        return {
            "assistant_messages": [{"role": "assistant", "content": "12"}],
            "prompt_ids": [[1, 2]],
            "response_ids": [[3, 4]],
            "response_logprobs": [[-0.1, -0.2]],
            "stop_reasons": ["stop"],
            "token_provenance": TokenProvenance.ENGINE,
        }


@dataclass
class Writer:
    groups: list[tuple[RolloutLease, RolloutGroup]] = field(default_factory=list)

    async def write_rollout(self, lease, group):
        self.groups.append((lease, group))


class FixtureImageFactory:
    """Execute fixture image commands on the built-in filesystem."""

    backend = Backend.DOCKER

    def check_environment(self, _requirements, _spec):
        pass

    async def create(self, spec):
        return await ShellSimMachineFactory().create(
            replace(spec, source=PackageSpec(), workdir=spec.workdir or "/workspace")
        )


@dataclass
class ConversationClient:
    responses: list[str]
    requests: list[dict] = field(default_factory=list)
    messages: list[dict] | None = None

    async def generate(self, request):
        index = len(self.requests)
        self.requests.append(request)
        continuation = request["chat_continuations"][0]
        prompt = [1, 2] if continuation is None else continuation["served_prefix_token_ids"] + [90, 91]
        tokens = [3 + index * 2, 4 + index * 2]
        response = self.responses[index]
        return {
            "assistant_messages": [
                {"role": "assistant", "content": response} if self.messages is None else self.messages[index]
            ],
            "responses": [response],
            "prompt_ids": [prompt],
            "response_ids": [tokens],
            "response_logprobs": [[-0.1, -0.2]],
            "stop_reasons": ["stop"],
            "token_provenance": TokenProvenance.ENGINE,
            "student_topk_indices": [[[token, 99] for token in tokens]],
            "behavior_topk_logprobs": [[[-0.1, -0.2], [-0.2, -0.3]]],
            "routed_experts": [np.ones((2, 1, 1), dtype=np.uint8)],
        }
