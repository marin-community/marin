"""Train the local OPD nightly from a typed recipe."""

import argparse
from pathlib import Path

from omegaconf import DictConfig

from skyrlrecipe import ContextBudget, SkyRLRecipe, sections
from skyrlrecipe.declared import Teacher, TeacherEndpoint, TeacherModel, TeacherResources, TeacherRoute, TeacherRouting
from examples.recipe import compose_recipe
from skyrl_train.entrypoints.main_base import run


def opd_config(args: argparse.Namespace) -> DictConfig:
    model = TeacherModel(path=args.teacher_model, revision=args.teacher_revision)
    if args.teacher_source == "local_inference":
        teacher = Teacher(
            source="local_inference",
            placement="pinned",
            model=model,
            backend="vllm",
            evidence="chosen_token",
            resources=TeacherResources(
                num_nodes=1, gpus_per_node=1, tensor_parallel_size=1, colocation_group="teacher"
            ),
        )
    else:
        if not args.tokenizer_fingerprint:
            raise ValueError("the HTTP teacher requires a tokenizer fingerprint")
        teacher = Teacher(
            source="openai_compatible",
            placement="external",
            model=model,
            endpoints=(TeacherEndpoint(url="http://127.0.0.1:18080/v1", max_concurrency=8),),
            tokenizer_fingerprint=args.tokenizer_fingerprint,
            max_sequence_length=32768,
            request_timeout_seconds=120,
            evidence="chosen_token",
        )
    recipe = SkyRLRecipe(
        context_budget=ContextBudget(
            request_window_tokens=320,
            max_new_tokens_per_turn=64,
            max_turns=1,
            overlong_cache_fraction=0,
        ),
        data=sections.Data(
            train_data=(str(args.data_dir / "train.parquet"),),
            val_data=(str(args.data_dir / "validation.parquet"),),
        ),
        trainer=sections.Trainer(
            algorithm=sections.Algorithm(
                advantage_estimator="grpo",
                use_kl_loss=False,
                # Stale rollouts need the behavior-policy anchor instead of the current-policy PPO anchor.
                policy_loss_type="behavior_clip" if args.max_staleness_steps > 0 else "regular",
                distillation=sections.Distillation(
                    objective="sampled_reverse_kl",
                    routing_plan="opd",
                    coefficient=0.5,
                    reward_mode="add",
                ),
            ),
            strategy="megatron",
            flash_attn=False,
            use_sample_packing=False,
            placement=sections.Placement(
                colocate_all=False,
                policy_num_gpus_per_node=1,
                ref_num_gpus_per_node=1,
            ),
            resume_mode="none",
            epochs=1,
            max_steps=args.steps,
            rollout_buffer=sections.RolloutBuffer(max_staleness_steps=args.max_staleness_steps),
            train_batch_size=2,
            policy_mini_batch_size=2,
            micro_train_batch_size_per_gpu=1,
            micro_forward_batch_size_per_gpu=1,
            update_epochs_per_batch=1,
            eval_before_train=False,
            eval_interval=-1,
            ckpt_interval=-1,
            hf_save_interval=-1,
            dump_eval_results=False,
            logger="console",
            project_name="marin_nightly",
            run_name="opd_qwen3_sync",
        ),
        generator=sections.Generator(
            backend="vllm",
            num_inference_engines=1,
            inference_engine_tensor_parallel_size=1,
            n_samples_per_prompt=2,
            gpu_memory_utilization=0.7,
            run_engines_locally=True,
            weight_sync_backend="nccl",
        ),
        environment=sections.Environment(env_class="gsm8k"),
        teachers={"primary": teacher},
        teacher_routing={
            "opd": TeacherRouting(
                revision="qwen3-sync-v1",
                routes={"default": TeacherRoute(teacher="primary", weight=1.0)},
            )
        },
    )
    config = compose_recipe(recipe)
    config.trainer.policy.model.path = args.model
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-source", choices=("local_inference", "openai_compatible"), required=True)
    parser.add_argument("--tokenizer-fingerprint")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-staleness-steps", type=int, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(opd_config(parse_args()))
