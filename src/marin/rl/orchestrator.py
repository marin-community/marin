# Orchestrator for Marin RL
#
# This is the main entry point for the Marin RL system. It is responsible for:
# - Spinning up the other components, including the number of replicas for each env.
# - Coordinating the training process and rollout generation
# - Receiving and logging metrics from the other components
#

# Inspiration: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/orchestrator/orchestrator.py


import functools
import logging
import jax
import jaxtyping
import ray
from jaxtyping import PRNGKeyArray
from ray.actor import ActorHandle

from marin.rl.datatypes import InferenceEndpoint, RolloutRecord
from marin.rl.replay_buffer import ReplayBuffer

from .config import AbstractEnvConfig, MarinRlConfig
from .learner import Learner


@ray.remote
class Orchestrator:
    def __init__(self, config: MarinRlConfig):
        self.config = config

        self.logger = logging.getLogger(__name__)
        self.tracker = config.tracker.init(config.id)
        self.current_step = 0

        self.inference_endpoint = self._initialize_inference_endpoint()

        base_key = jax.random.PRNGKey(config.seed)
        env_key, learner_key, replay_key = jax.random.split(base_key, 3)

        # initialize replay buffers (one per env)
        self.replay_buffers = self._initialize_replay_buffers(config.envs, replay_key)

        self.envs = self._initialize_envs(config.envs, env_key)

        # initialize the learner
        self.learner = self._initialize_learner(learner_key)

        self.weight_broadcaster = ray.get(self.learner.get_weight_broadcaster.remote())


    def mark_step(self, step_number: int):
        """
        Records the current step number. Should be called by the learner every step.
        """
        self.current_step = step_number


    def log_metrics(self, metrics: dict[str, float]):
        self.tracker.log(metrics, step=self.current_step)


    def start(self):
        self.current_step = ray.get(self.learner.get_step.remote())

        futs = []
        futs.append(self.learner.start.remote())

        # start everything
        for env in self.envs.values():
            for e in env:
                futs.append(e.start.remote())

        ray.get(futs)


    def _initialize_envs(self, env_configs: list[AbstractEnvConfig], env_key: PRNGKeyArray) -> dict[str, list[ActorHandle]]:
        envs: dict[str, list[ActorHandle]] = {}
        env_keys = jax.random.split(env_key, len(env_configs))
        del env_key

        for env_config, env_key in zip(env_configs, env_keys):
            if self.config.env_replica_counts is None:
                replica_count = 1
            else:
                replica_count = self.config.env_replica_counts[env_config.name]

            envs[env_config.name] = []

            for _ in range(replica_count):
                env_key, k = jax.random.split(env_key)
                env_seed = jax.random.randint(k, (1,), 0, 1000000)
                env = env_config.build(self.inference_endpoint, env_seed)
                envs[env_config.name].append(env)

        return envs

    def _initialize_inference_endpoint(self) -> InferenceEndpoint:
        raise NotImplementedError("Inference endpoint not implemented")

    def _initialize_learner(self, learner_key: PRNGKeyArray) -> ActorHandle[Learner]:
        raise NotImplementedError("Learner not implemented")





# TODO: back pressure? async?
def _rollout_sink_to_buffer(buffer: ActorHandle[ReplayBuffer], rollouts):  # deprecated
    return ray.get(buffer.extend.remote(rollouts))
