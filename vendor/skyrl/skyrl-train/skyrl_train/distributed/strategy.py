import random
from abc import ABC, abstractmethod

from loguru import logger
import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.tensor import DeviceMesh
from typing import Optional, Dict, Any, Union, TypeVar
import torch.optim as optim
from jaxtyping import Float
from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer
from skyrl_train.io import io


DataT = TypeVar("DataT", bound=Union[Dict[str, Any], torch.Tensor])


class DistributedStrategy(ABC):
    device_mesh: DeviceMesh | None = None
    ep_size: int = 1

    @abstractmethod
    def setup_distributed(self):
        pass

    @abstractmethod
    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs):
        """Perform backward pass"""
        pass

    @abstractmethod
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        pass

    @abstractmethod
    def save_checkpoint(self, model, ckpt_dir, node_local_rank, optimizer, scheduler, tokenizer):
        """Save checkpoint"""
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        model,
        ckpt_dir,
        optimizer,
        scheduler,
        load_module_strict,
        load_training_state,
    ):
        """Load checkpoint"""
        pass

    @abstractmethod
    def save_hf_model(self, model, output_dir: str, tokenizer=None, **kwargs):
        """Save model in HuggingFace safetensors format"""
        pass

    def log(self, *msg: object) -> None:
        """Log a message only on rank 0."""
        if self.is_rank_0():
            logger.opt(depth=1).info(" ".join(str(part) for part in msg))

    def is_rank_0(self) -> bool:
        """Check if current process is rank 0"""
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        """Get current process rank"""
        return dist.get_rank()

    def collective_device(self) -> torch.device:
        """Device holding tensors passed to this strategy's process-group collectives."""
        return torch.device("cuda", torch.cuda.current_device())

    def all_reduce(self, data: DataT, op="mean") -> DataT:
        """Perform all_reduce across all processes"""
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(self.collective_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data: DataT) -> DataT:
        """Perform all_gather across all processes"""
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            device = self.collective_device()
            ret = [torch.zeros_like(data).to(device) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(device))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def save_hf_configs(self, model_config: PretrainedConfig, hf_dir: str, tokenizer: PreTrainedTokenizer = None):
        """
        Save model and tokenizer configs to hf_dir

        Args:
            model_config: PretrainedConfig - huggingface model config
            hf_dir: str - the directory to save the huggingface configs to
            tokenizer: PreTrainedTokenizer - tokenizer to save
        """
        io.makedirs(hf_dir, exist_ok=True)

        with io.local_work_dir(hf_dir) as work_dir:
            model_config.save_pretrained(work_dir)
            if tokenizer:
                tokenizer.save_pretrained(work_dir)

            if hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    # with io.local_work_dir(hf_config_tokenizer_path) as work_dir:
                    generation_config.save_pretrained(work_dir)
                except Exception as e:
                    # if the generation config isn't available, we don't save it
                    logger.warning(f"Could not save generation config for '{model_config.name_or_path}'. Error: {e}")
                    pass

    @staticmethod
    def get_rng_state():
        """Get current RNG state for reproducibility"""
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        # Only save CUDA RNG state if CUDA is available and being used
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            rng_state["cuda"] = torch.cuda.get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        """Load RNG state for reproducibility"""
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        # Only restore CUDA RNG state if it was saved and CUDA is available
        if "cuda" in rng_state and torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.set_rng_state(rng_state["cuda"])
