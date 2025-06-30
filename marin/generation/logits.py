import logging
import os
import tempfile
from dataclasses import dataclass

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin.processing.classification.inference import read_dataset, write_dataset
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger("ray")


@dataclass
class TextLogitsConfig:
    """Configuration for running forward passes on a dataset."""

    model_name: str
    input_path: str
    output_path: str
    batch_size: int = 8
    max_length: int = 2048
    memory_gb: int = 10


def compute_logits(config: TextLogitsConfig) -> None:
    """Run a model forward pass and store logits for each example on TPU."""

    logger.info(
        f"Computing logits for {config.input_path} using {config.model_name}"
    )

    @ray.remote(
        memory=config.memory_gb * 1024 * 1024 * 1024,
        resources={"TPU": 4, "TPU-v4-8-head": 1},
    )
    @remove_tpu_lockfile_on_exit
    def run(cfg: TextLogitsConfig):
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp

        def _mp_fn(index: int, cfg: TextLogitsConfig, tmp_dir: str):
            dataset = read_dataset(cfg.input_path)
            dataset = dataset.shard(xmp.xrt_world_size(), index)

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            device = xm.xla_device()
            model.to(device)
            model.eval()

            def _forward(batch):
                tokens = tokenizer(
                    batch["text"],
                    truncation=True,
                    padding=True,
                    max_length=cfg.max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                with torch.no_grad():
                    outputs = model(**tokens)
                xm.mark_step()
                batch["logits"] = outputs.logits.cpu().tolist()
                return batch

            dataset = dataset.map(
                _forward, batched=True, batch_size=cfg.batch_size
            )

            shard_path = os.path.join(tmp_dir, f"logits_{index}.jsonl.gz")
            write_dataset(dataset, shard_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            xmp.spawn(_mp_fn, args=(cfg, tmp_dir))
            import glob
            import datasets

            shard_files = sorted(glob.glob(os.path.join(tmp_dir, "logits_*.jsonl.gz")))
            shards = [read_dataset(p) for p in shard_files]
            combined = datasets.concatenate_datasets(shards)
            write_dataset(combined, cfg.output_path)

    ray.get(run.remote(config))


if __name__ == "__main__":
    import draccus

    @draccus.wrap()
    def main(cfg: TextLogitsConfig) -> None:  # pragma: no cover - CLI entrypoint
        compute_logits(cfg)

    main()
