# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise CPU packing through TPU scoring in the locked Marin environment.

Run with python inside an Iris job using the tpu and lm_eval extras. This uses
a tiny random Qwen3 and synthetic local documents, not a paper checkpoint.
"""

import argparse
import hashlib
import json
import logging
import math
import tempfile
from pathlib import Path

import fsspec
import haliax as hax
import jax
import jmp
from levanter.distributed import DistributedConfig
from levanter.eval_harness import LmEvalHarnessConfig, SampleLoggingConfig, run_lm_eval_harness
from levanter.eval_harness_config import TaskConfig
from levanter.models.qwen import Qwen3Config
from levanter.tokenizers import load_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.py_utils import FailSafeJSONEncoder
from tokenizers import Tokenizer, models, pre_tokenizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "tpu"), default="tpu")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    DistributedConfig().initialize()
    assert jax.default_backend() == args.backend
    assert jax.device_count() == 4 and jax.process_count() == 1
    if args.report:
        assert args.report.startswith("gs://marin-us-east5/")
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        per_device_eval_parallelism=2,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_jaxprs=False,
        log_xla_hlo=False,
        shutdown_at_exit=False,
    )
    trainer.initialize()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        vocab = {token: index for index, token in enumerate(["[UNK]", "[EOS]", "q", "yes", "no"])}
        vocab.update({f"unused_{index}": index for index in range(len(vocab), 32)})
        raw_tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        raw_tokenizer.save(str(root / "tokenizer.json"))
        (root / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "unk_token": "[UNK]",
                    "eos_token": "[EOS]",
                    "pad_token": "[EOS]",
                }
            )
        )
        tokenizer = load_tokenizer(str(root))
        # Each continuation occupies its own pack: 12 requests need a full batch and a padded tail.
        question = " ".join(["q"] * (args.length // 2))
        docs = [{"question": question, "choices": ["yes", "no"], "gold": i % 2} for i in range(6)]
        data = root / "questions.jsonl"
        data.write_text("\n".join(json.dumps(doc) for doc in docs))
        metrics = ["acc", "bpb", "logprob", "choice_logprob", "choice_prob_norm", "choice_logprob_norm"]
        task = TaskConfig(
            task="batch_canary",
            dataset_path="json",
            dataset_kwargs={"data_files": {"test": str(data)}},
            test_split="test",
            output_type="multiple_choice",
            doc_to_text="{{question}}",
            doc_to_choice="{{choices}}",
            doc_to_target="{{gold}}",
            num_fewshot=0,
            metric_list=[
                {"metric": metric, "aggregation": "mean", "higher_is_better": metric != "bpb"} for metric in metrics
            ],
        )
        config = LmEvalHarnessConfig(
            task_spec=[task], max_length=args.length, log_samples=True, sample_logging=SampleLoggingConfig(log_all=True)
        )
        with trainer.use_device_mesh():
            model_config = Qwen3Config(
                max_seq_len=args.length,
                hidden_dim=256,
                intermediate_dim=512,
                num_layers=1,
                num_heads=2,
                num_kv_heads=2,
                head_dim=128,
            )
            model = model_config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(0))
            model = hax.shard(trainer.mp.cast_to_param(model), trainer.parameter_axis_mapping)
            result = run_lm_eval_harness(config, model, tokenizer, 8, trainer.compute_axis_mapping, trainer.mp)
        assert result is not None
        task_result = result["results"]["batch_canary"]
        samples = result["samples"]["batch_canary"]
        assert result["n-samples"]["batch_canary"]["effective"] == 6
        assert len(samples) == 6 and len(task_result["outputs"]) == 12
        assert task_result["acc,none"] == 0.5
        for sample in samples:
            assert len(sample["resps"]) == 2
            assert all(math.isfinite(response[0][0]) for response in sample["resps"])
        for metric in metrics:
            assert math.isfinite(task_result[metric + ",none"])
        report = {
            "passed": True,
            "backend": jax.default_backend(),
            "devices": jax.device_count(),
            "batch_size": 8,
            "max_length": args.length,
            "precision": "bfloat16",
            "model": "tiny random Qwen3",
            "documents": 6,
            "requests": 12,
            "metrics": {metric: task_result[metric + ",none"] for metric in metrics},
            "packing_sha256": hashlib.sha256(Path("lib/levanter/src/levanter/data/packing.py").read_bytes()).hexdigest(),
        }
        if args.report:
            with fsspec.open(args.report, "wt") as handle:
                json.dump(report, handle, cls=FailSafeJSONEncoder)
        print(json.dumps(report, cls=FailSafeJSONEncoder))


if __name__ == "__main__":
    main()
