# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The V1 smoke: the Deletion Tuning main end to end on CPU at toy scale.

Runs the production V1 pod entrypoint (``run_deletion_train_lm``) over a tiny
Snowball config, a tiny local corpus, and the vendored offline tokenizer
fixture. Everything about the entry is real except the scale and the data: the
pod preparation, the DT seam, the HF reload with the Deletion block preserved,
the Wanda calibration over a training batch, the gate-logit init, the AdamW
logit-only optimizer state, the alpha ramp, and the end-of-run HF export.

Scale deviations from the production spec (recorded): local offline tokenizer
fixture and toy corpus instead of the Exp3 V5 caches, toy geometry, three
steps, and no Snowball SFT controls (the fixture tokenizer has no chat
special-token IDs and plain batches carry no prediction weights).
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

import haliax as hax
import jax
import jmp
import jax.numpy as jnp
from fray.types import ResourceConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.main.train_lm import TrainLmConfig
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.models.snowball_deletion import DeletionTuningConfig, DeletionTuningScope
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.training.training import TrainLmOnPodConfig

from experiments.deletion_tuning.levanter_train import run_deletion_train_lm
from experiments.deletion_tuning.v1_sft import (
    V1_ALPHA_END,
    V1_ALPHA_START,
    V1_ETA,
    V1_LAMBDA,
    V1_LOGIT_LR,
    v1_optimizer,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TOKENIZER_FILE = REPO_ROOT / "lib" / "levanter" / "tests" / "gpt2_tokenizer.json"
LOCAL_TOKENIZER_CONFIG = REPO_ROOT / "lib" / "levanter" / "tests" / "gpt2_tokenizer_config.json"
LOCAL_TOKENIZER_VOCAB_SIZE = 5027
LOCAL_TOKENIZER_EOS_TOKEN_ID = 5026
SMOKE_STEPS = 3
SEQ_LEN = 64


def _materialize_tokenizer(dest_dir: Path) -> str:
    """Lay out the vendored offline tokenizer fixture as a loadable tokenizer directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(LOCAL_TOKENIZER_FILE, dest_dir / "tokenizer.json")
    shutil.copy(LOCAL_TOKENIZER_CONFIG, dest_dir / "tokenizer_config.json")
    (dest_dir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": LOCAL_TOKENIZER_VOCAB_SIZE}))
    return str(dest_dir)


def _tiny_corpus(corpus_path: Path) -> None:
    """A toy corpus long enough to fill the smoke's three batches."""
    lines = []
    for index in range(64):
        words = " ".join(str((index * 7 + word) % 5000) for word in range(40))
        lines.append(json.dumps({"text": f"{words} {words}"}))
    corpus_path.write_text("\n".join(lines) + "\n")


def _tiny_config(tokenizer: str) -> SnowballConfig:
    return SnowballConfig(
        deletion_tuning=DeletionTuningConfig(
            scope=DeletionTuningScope.ROUTED_GATE,
            eta=V1_ETA,
            lam=V1_LAMBDA,
            alpha_start=V1_ALPHA_START,
            alpha_end=V1_ALPHA_END,
            lr=V1_LOGIT_LR,
            calibration_sequences=1,
            seed=0,
        ),
        vocab_size=LOCAL_TOKENIZER_VOCAB_SIZE,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_kv_heads=2,
        num_heads=2,
        head_dim=8,
        max_seq_len=SEQ_LEN,
        sliding_window=SEQ_LEN,
        attention_implementation="reference",
        moe_implementation="scatter",
        tokenizer=tokenizer,
    )


def _save_tiny_hf(config: SnowballConfig, hf_dir: Path) -> str:
    """A tiny HF-format checkpoint for ``initialize_from_hf`` (the weight-source path)."""
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        vocab = hax.Axis("vocab", config.vocab_size)
        model = SnowballLMHeadModel.init(vocab, config, key=jax.random.PRNGKey(0))
    config.hf_checkpoint_converter().save_pretrained(model, str(hf_dir), dtype=jnp.float32)
    return str(hf_dir)


def _smoke_train_config(tokenizer: str, hf_dir: str, corpus: Path, cache: Path, output_path: Path) -> TrainLmConfig:
    tokenize_config = TokenizeConfig(
        train_paths=[str(corpus)],
        validation_paths=[],
        cache_path=str(cache),
        tokenizer=tokenizer,
    )
    data = lm_mixture_data_config(components={"smoke": tokenize_config}, weights={"smoke": 1.0})
    mesh = MeshConfig(
        axes={"data": 1, "expert": 1, "model": 1, "replica": 1},
        dcn_axes={"context": 1, "replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
    )
    return TrainLmConfig(
        data=data,
        model=_tiny_config(tokenizer),
        optimizer=v1_optimizer(),
        trainer=TrainerConfig(
            train_batch_size=2,
            num_train_steps=SMOKE_STEPS,
            per_device_parallelism=1,
            per_device_eval_parallelism=1,
            seed=0,
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            mesh=mesh,
            use_explicit_mesh_axes=True,
            require_accelerator=False,
            steps_per_eval=SMOKE_STEPS,
            jax_compilation_cache_dir=str(output_path / "jax-cache"),
        ),
        train_seq_len=SEQ_LEN,
        initialize_from_hf=hf_dir,
        use_hf_model_config=True,
        pad_tokenizer_to_match_model=True,
        hf_save_steps=SMOKE_STEPS + 1,
        hf_save_dtype="bfloat16",
        hf_generation_eos_token_ids=[LOCAL_TOKENIZER_EOS_TOKEN_ID],
        global_gradient_clip=1,
        data_seed=0,
    )


def run_smoke(output_root: Path | None = None) -> dict:
    """Run the V1 pod entrypoint end to end at toy scale and report what ran."""
    with tempfile.TemporaryDirectory(prefix="v1-sft-smoke-") as tmp:
        root = Path(tmp)
        tokenizer = _materialize_tokenizer(root / "tokenizer")
        corpus = root / "corpus.jsonl"
        _tiny_corpus(corpus)
        cache = root / "cache"
        tokenize(TokenizeConfig(train_paths=[str(corpus)], validation_paths=[], cache_path=str(cache), tokenizer=tokenizer))
        hf_dir = _save_tiny_hf(_tiny_config(tokenizer), root / "base-hf")
        output_path = root / "run"
        pod_config = TrainLmOnPodConfig(
            output_path=str(output_path),
            resources=ResourceConfig.with_cpu(),
            env_vars={"WANDB_API_KEY": "", "WANDB_MODE": "disabled", "JAX_TRACEBACK_FILTERING": "off"},
            train_config=_smoke_train_config(tokenizer, hf_dir, corpus, cache, output_path),
        )
        run_deletion_train_lm(pod_config)

        exports = sorted(str(path) for path in output_path.rglob("*.safetensors"))
        assert exports, "the smoke must produce the end-of-run HF export"
        summary = {
            "steps": SMOKE_STEPS,
            "calibration_batches": 1,
            "sequence_length": SEQ_LEN,
            "batch_size": 2,
            "vocab": LOCAL_TOKENIZER_VOCAB_SIZE,
            "optimizer": "AdamW constant lr 5e-4, weight decay 0, on the gate logits only",
            "alpha_ramp": [V1_ALPHA_START, V1_ALPHA_END],
            "hf_export": exports[:1],
        }
        logger.info("smoke complete: %s", json.dumps(summary, indent=2))
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "v1-smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
