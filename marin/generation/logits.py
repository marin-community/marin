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
    span_chars: int = 4096  # size of character window to slice text files


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
        import torch_xla.runtime as xr  # new runtime API in torch-xla >=2.6

        logger.info(
            "[TPU-VM] Starting logits computation with %s on input %s (batch_size=%d, max_len=%d)",
            cfg.model_name,
            cfg.input_path,
            cfg.batch_size,
            cfg.max_length,
        )

        def _mp_fn(index: int, cfg: TextLogitsConfig, tmp_dir: str):
            logger.info(
                "[Core %d] Loading dataset slice and model…",
                index,
            )
            # Load dataset depending on file type
            if cfg.input_path.endswith('.txt'):
                import datasets, fsspec
                fs_file = fsspec.open(cfg.input_path, 'r')
                with fs_file as f:
                    full_text = f.read()
                spans = [full_text[i:i + cfg.span_chars] for i in range(0, len(full_text), cfg.span_chars) if full_text[i:i + cfg.span_chars].strip()]
                dataset = datasets.Dataset.from_dict({"text": spans})
                logger.info("[Core %d] Loaded %d spans from raw txt file", index, len(dataset))
            else:
                dataset = read_dataset(cfg.input_path)
                logger.info("[Core %d] Loaded dataset from %s with %d rows", index, cfg.input_path, len(dataset))
            world_size = xr.world_size()
            dataset = dataset.shard(world_size, index)

            logger.info("[Core %d] Dataset size after sharding: %d", index, len(dataset))
            logger.info("[Core %d] Dataset columns: %s", index, dataset.column_names)
            try:
                first_example = dataset[0]
                logger.info("[Core %d] First example keys: %s", index, list(first_example.keys()))
            except Exception as e:
                logger.warning("[Core %d] Could not inspect first example: %s", index, e)

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            device = xm.xla_device()
            model.to(device)
            model.eval()

            logger.info("[Core %d] Model and tokenizer loaded; beginning forward pass…", index)

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

            logger.info("[Core %d] Finished forward pass over %d examples", index, len(dataset))

            # Determine shard output path
            if cfg.output_path.endswith(".parquet"):
                # Write each core directly to its own parquet file alongside the desired output path
                base, _ = os.path.splitext(cfg.output_path)
                shard_path = f"{base}_{index}.parquet"
            else:
                shard_path = os.path.join(tmp_dir, f"logits_{index}.jsonl.gz")

            write_dataset(dataset, shard_path)

            logger.info("[Core %d] Wrote shard to %s", index, shard_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info("[TPU-VM] Spawning processes across %d TPU cores…", xr.world_size())
            world_size = xr.world_size()
            xmp.spawn(_mp_fn, args=(cfg, tmp_dir), nprocs=world_size)

            # If we wrote parquet shards directly, merging is optional; simply exit.
            if cfg.output_path.endswith(".parquet"):
                logger.info("[TPU-VM] Parquet shards written; merge skipped.")
                return

            import glob
            import datasets

            shard_files = sorted(glob.glob(os.path.join(tmp_dir, "logits_*.jsonl.gz")))
            shards = [read_dataset(p) for p in shard_files]
            combined = datasets.concatenate_datasets(shards)
            write_dataset(combined, cfg.output_path)

            logger.info(
                "[TPU-VM] Successfully wrote combined dataset (%d examples) to %s",
                len(combined),
                cfg.output_path,
            )

    ray.get(run.remote(config))


if __name__ == "__main__":
    import draccus

    @draccus.wrap()
    def main(cfg: TextLogitsConfig) -> None:  # pragma: no cover - CLI entrypoint
        compute_logits(cfg)

    main()
