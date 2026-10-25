import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

from omegaconf import OmegaConf
import ray
import torch

from cloud.iris.hf_model_cache import stage_artifact_model
from cloud.iris.launch_config import load_launch_config
from skyrl_train.distributed.megatron import model_utils
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.utils.utils import initialize_ray, validate_cfg
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup


class RecordingModelWrapper(MegatronModelWrapper):
    """Collect token outputs while executing the ordinary Megatron policy step."""

    def _token_logprobs(self, logits, sequences, attention_mask, packed_seq_params):
        output = super()._token_logprobs(logits, sequences, attention_mask, packed_seq_params)
        self.observations.append(
            {
                "kind": "logprob",
                "values": output.detach().cpu(),
                "sequences": sequences.detach().cpu(),
                "attention_mask": attention_mask.detach().cpu(),
                "logits_shape": list(logits.shape),
                "logits_stride": list(logits.stride()),
                "logits_sha256": (
                    hashlib.sha256(logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                ),
            }
        )
        return output

    def _token_entropies(self, logits, attention_mask, packed_seq_params):
        output = super()._token_entropies(logits, attention_mask, packed_seq_params)
        self.observations.append({"kind": "entropy", "values": output.detach().cpu()})
        return output


class ReplayWorker(MegatronPolicyWorkerBase):
    """Run one recorded PPO update and return its token tensors and metrics."""

    def init_model(self, model_path, num_training_steps=1):
        super().init_model(model_path, num_training_steps=num_training_steps)
        checksum = hashlib.sha256()
        for module in self.actor_module:
            for name, parameter in module.named_parameters():
                checksum.update(name.encode())
                checksum.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        self.initial_parameter_sha256 = checksum.hexdigest()
        self.model = RecordingModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
            logprob_chunk_size=self.cfg.trainer.policy.megatron_config.logprob_chunk_size,
            vocabulary_size=self.strategy.hf_config.vocab_size,
        )
        self.model.observations = []

    def replay(self, batch):
        result = self.ppo_train(batch)
        result.metadata["observations"] = self.model.observations
        result.metadata["production_source"] = str(Path(model_utils.__file__).resolve())
        result.metadata["production_sha256"] = hashlib.sha256(Path(model_utils.__file__).read_bytes()).hexdigest()
        result.metadata["gpu"] = torch.cuda.get_device_name()
        result.metadata["initial_parameter_sha256"] = self.initial_parameter_sha256
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    batch_bytes = args.batch.read_bytes()
    batch = pickle.loads(batch_bytes)
    cfg = load_launch_config(args.config).skyrl
    model_path = Path.home() / ".cache/oa/softmax-numerics" / hashlib.sha256(args.model_uri.encode()).hexdigest()[:16]
    started = time.monotonic()
    staged_bytes = stage_artifact_model(args.model_uri, args.model_uri, str(model_path))
    receipt = {
        "commit": args.commit,
        "batch_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model_uri": args.model_uri,
        "staged_bytes": staged_bytes,
        "staging_seconds": time.monotonic() - started,
        "torch": torch.__version__,
    }
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.policy.model.source_uri = None
    cfg.trainer.policy.model.revision = None
    cfg.trainer.policy.model.tokenizer_path = str(model_path)
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.logger = "console"
    validate_cfg(cfg)
    batch.metadata["global_step"] = 0
    OmegaConf.save(cfg, args.output / "replay-config.yaml")
    initialize_ray(cfg)
    try:
        policy = PPORayActorGroup(
            cfg,
            num_nodes=1,
            num_gpus_per_node=2,
            ray_actor_type=ray.remote(num_gpus=1)(ReplayWorker),
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            record_memory=False,
        )
        ray.get(policy.async_init_model(str(model_path), num_training_steps=1))
        outputs: list[TrainingOutputBatch] = ray.get(policy.async_run_ray_method("mesh", "replay", batch))
        receipt["ranks"] = []
        for rank, output in enumerate(outputs):
            observations = output.metadata.pop("observations")
            torch.save(observations, args.output / f"rank-{rank}-tensors.pt")
            receipt["ranks"].append(output.metadata)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
        print(json.dumps(receipt, default=str), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
