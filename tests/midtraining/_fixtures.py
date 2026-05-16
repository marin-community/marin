# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight fixtures for midtraining unit tests."""

from dataclasses import dataclass

from marin.midtraining.budget import BudgetPolicy
from marin.midtraining.data_manifest import DataCacheComponent, DataCacheManifest
from marin.midtraining.identity import build_run_identity
from marin.midtraining.modes import (
    CheckpointSourceKind,
    CooldownMode,
    CooldownResume,
    CptInit,
    CptMode,
)
from marin.midtraining.spec import ComputeProfile, MidtrainSpec
from marin.midtraining.tokenizers import LLAMA3_TOKENIZER, TokenizerRef


@dataclass(frozen=True)
class FakeBase:
    """Stand-in for ``experiments.delphi_models.DelphiModel`` in tests."""

    flops_key: str
    params: int
    hidden_dim: int
    num_layers: int
    batch_size: int
    num_train_steps: int
    hf_repo: str
    hf_revision: str
    gcs_run_root: str
    verified_checkpoint_step: int
    seq_len: int = 4096

    @property
    def tokens(self) -> int:
        return self.num_train_steps * self.batch_size * self.seq_len

    @property
    def verified_checkpoint_path(self) -> str:
        return self.levanter_checkpoint_path(self.verified_checkpoint_step)

    def levanter_checkpoint_path(self, step: int | None = None) -> str:
        return f"{self.gcs_run_root}/checkpoints/step-{step or self.verified_checkpoint_step}"


FAKE_1E21 = FakeBase(
    flops_key="1e21",
    params=3_400_000_000,
    hidden_dim=2560,
    num_layers=26,
    batch_size=512,
    num_train_steps=22_057,
    hf_repo="marin-community/delphi-1e21-3.4Bparams-46.3Btokens",
    hf_revision="586f00015fed3a11a72db392c1236d9947e8b652",
    gcs_run_root="gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021",
    verified_checkpoint_step=21_979,
)


FAKE_1E22 = FakeBase(
    flops_key="1e22",
    params=9_700_000_000,
    hidden_dim=3840,
    num_layers=37,
    batch_size=1024,
    num_train_steps=38_235,
    hf_repo="marin-community/delphi-1e22-9.7Bparams-160Btokens",
    hf_revision="ca7b0e7c0a6b9ea8e3a4bbe847efa8b53f793902",
    gcs_run_root="gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e",
    verified_checkpoint_step=38_206,
)


def make_data_manifest(
    *,
    mix_name: str = "p33m67_test",
    region: str = "us-east5",
    seq_len: int = 4096,
    tokenizer: TokenizerRef = LLAMA3_TOKENIZER,
    bos_sample: tuple[int, ...] = (128_000, 128_000, 128_000),
) -> DataCacheManifest:
    components = (
        DataCacheComponent(
            logical_name="pretrain",
            cache_path=f"gs://marin-{region}/tokenized/pretrain",
            cache_digest="sha256:test-pretrain",
            tokenizer=tokenizer,
            total_sequences=10_000,
            total_tokens=10_000 * seq_len,
            bos_sample=bos_sample,
        ),
        DataCacheComponent(
            logical_name="nemotron_cc_math_v1/4plus",
            cache_path=f"gs://marin-{region}/tokenized/nemotron_cc_math",
            cache_digest="sha256:test-math",
            tokenizer=tokenizer,
            total_sequences=20_000,
            total_tokens=20_000 * seq_len,
            bos_sample=bos_sample,
        ),
    )
    return DataCacheManifest(
        mix_name=mix_name,
        mix_spec_digest="sha256:test-mix",
        region=region,
        components=components,
        weights={"pretrain": 0.33, "nemotron_cc_math_v1/4plus": 0.67},
        seq_len=seq_len,
    )


def make_model_config(base: FakeBase) -> dict:
    return {
        "type": "llama",
        "hidden_dim": base.hidden_dim,
        "num_layers": base.num_layers,
        "num_heads": 32,
        "num_kv_heads": 8,
        "intermediate_dim": 4 * base.hidden_dim,
        "vocab_size": 128_256,
        "max_seq_len": 4096,
    }


def make_optimizer_config(*, learning_rate: float = 1e-3) -> dict:
    return {
        "type": "adam_h",
        "learning_rate": learning_rate,
        "adam_lr": learning_rate / 10,
        "beta1": 0.9,
        "beta2": 0.99,
        "epsilon": 1e-8,
        "max_grad_norm": 0.1,
        "warmup": 500,
        "decay": 4000,
        "min_lr_ratio": 0.0,
    }


def make_cpt_spec(
    *,
    base: FakeBase = FAKE_1E21,
    region: str = "us-east5",
    banned_substrings: frozenset[str] = frozenset({"adamh_scaling_v5"}),
    expected_min_step: int | None = None,
    extra_compute_kwargs: dict | None = None,
    model_config_override: dict | None = None,
    data_manifest_uri: str | None = None,
) -> MidtrainSpec:
    run = build_run_identity(
        logical_cell_id="delphi-1e21-p33m67-k0p20-lr0p5",
        attempt=1,
        output_region_name=region,
        wandb_project="delphi-midtraining",
    )
    compute_kwargs = {"tpu_type": "v5p-64", "batch_size": base.batch_size, "regions": (region,)}
    if extra_compute_kwargs:
        compute_kwargs.update(extra_compute_kwargs)
    return MidtrainSpec(
        base=base,
        run=run,
        compute=ComputeProfile(**compute_kwargs),
        mode=CptMode(
            init=CptInit(source_kind=CheckpointSourceKind.NATIVE_LEVANTER, registry_model=base),
            budget=BudgetPolicy.pretrain_fraction(0.20),
        ),
        data_manifest_uri=data_manifest_uri or f"gs://marin-{region}/midtrain-manifests/data/p33m67/abc.json",
        tokenizer=LLAMA3_TOKENIZER,
        model_config=model_config_override or make_model_config(base),
        optimizer_config=make_optimizer_config(),
        banned_substrings=banned_substrings,
        expected_min_step=expected_min_step,
    )


def make_cooldown_spec(
    *,
    base: FakeBase = FAKE_1E22,
    region: str = "us-east5",
    resume_step: int = 30_000,
    stop_step_override: int | None = None,
) -> MidtrainSpec:
    run = build_run_identity(
        logical_cell_id=f"true-midtrain-{base.flops_key}-step{resume_step}",
        attempt=1,
        output_region_name=region,
        wandb_project="delphi-midtraining",
    )
    mode = CooldownMode(
        resume=CooldownResume(
            pretrain_checkpoint_path=base.levanter_checkpoint_path(resume_step),
            resume_step=resume_step,
            staged_output_path=run.output_path,
        ),
        stop_step_override=stop_step_override,
    )
    return MidtrainSpec(
        base=base,
        run=run,
        compute=ComputeProfile(tpu_type="v5p-64", batch_size=base.batch_size, regions=(region,)),
        mode=mode,
        data_manifest_uri=f"gs://marin-{region}/midtrain-manifests/data/p33m67/abc.json",
        tokenizer=LLAMA3_TOKENIZER,
        model_config=make_model_config(base),
        optimizer_config=make_optimizer_config(),
    )
