# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Per-field decisions for the marin_belay_v1_clean cleanup; renders scratchpad/clean/INVENTORY.md from
# inventory.json (written by inventory.py on the unmodified w2026-10-19 tree).
import json
from pathlib import Path

KEEP, RESEARCH, DEAD, GB200 = "keep", "keep (research)", "delete (dead)", "delete (GB200/MFU)"

MODEL_DEAD = """moe_dynamic moe_dynamic_layers moe_k_static moe_k_low attn_head_skip moe_small_expert_width
moe_dynamic_position moe_dynamic_frac moe_relu_l1 moe_relu_offset moe_eps moe_top_p moe_k_layers moe_null_layers
moe_null_renorm_scale moe_null_learned_gain moe_null_target_frac latent_out_dim latent_out_full_layers
attn_res_shared_query extra_mlp_layers extra_mlp_dim mid_head_layer mid_head_loss_weight mid_head_trunk_grad
mid_confidence_dim final_conf_layers skip_probe_layer skip_probe_target mid_skip_layers mid_skip_threshold
entropy_head_input_layer attn_res_partial_scale kda_onchip_erase_gate kda_write_gate kda_erase_gate moe_expert_waves
moe_expert_remat moe_serial_pair moe_drop_renorm second_embed embed2_rows bigram_gate bigram_gate_rank embed2_fsdp
embed2_row_sharded_gather embed2_ngram embed2_dim embed2_lambda_init second_embed_bigram""".split()
MODEL_GB200 = """attn_res_two_phase scan_offload_partials scan_save_routed_output attn_res_fused_source_grads scan_layers
value_embed_fsdp fp8_recipe mxfp8_fused_grad_quant moe_overlap_weight_grads kda_fused_projections moe_mxfp8_dispatch
moe_shared_placement""".split()
MODEL_RESEARCH = """attn_res_block_starts attn_res_plain_sum attn_res_kv_layers attn_res_kv_source_layers
entropy_skip_layers extra_mlp_select extra_mlp_frac skip_adapter_dim skip_router_layers skip_router_frac
lm_head_grad_probe lm_head_grad_probe_site lm_head_grad_dump_every token_embed_stats lm_head_grad_clip sconv_impl
gpu_attention kda_state_block_v moe_ragged_chunks""".split()

OPT_DEAD = "embed2_lr_mult embed2_beta2 muon_bimaxwell max_grad_norm".split()
OPT_RESEARCH = "embed_optimizer embed_weight_decay".split()

TRAINER_DEAD = "data_seed watch_mode routing_dump_dir routing_dump_steps ema_beta ema_last_steps pipeline_depth".split()
TRAINER_GB200 = """xla_autotune_cache_path xla_autotune_seed_topology preload_library_kernels first_step_timeout
shared_executable_dir offload_opt_state master_param_mode""".split()
RUN_DEAD = ["collective_overlap_limit"]
RUNG_GB200 = ["device", "hero_memory"]

NOTES = {
    "moe_drop_renorm": "Ckpt 1 sets it to its default (false); entry dropped from MODEL_SETTINGS",
    "moe_shared_placement": "FUSED (default) kept as the only path; OVERLAP / UNFUSED were GB200 scheduling",
    "attn_res_two_phase": "scan_layers only",
    "gpu_attention": "kernel name passed to levanter attention; 28 recent H100 runs set gpu_pallas_fa4",
    "kda_state_block_v": "kernel tile; 28 recent H100 runs set 64",
    "moe_ragged_chunks": "H100 rung default 2; 6 recent H100 runs set 1",
    "max_grad_norm": "never set (MuonH path)",
    "watch_mode": "DIAGNOSTIC watch never used; inline watch is the only mode",
    "pipeline_depth": "opt-in 1 crashed near step 515; only 15 H100 MFU probes used it",
    "ragged_transport": "DEVICE (the GB200 NCCL GIN kernel, old default) removed; ONE_SHOT is the default",
    "master_param_mode": "fp32 pinned-host master, GB200 hero memory",
}

LAUNCH_OPTIONS = {
    # option: (decision, note)
    "--run-id": (KEEP, ""),
    "--size": (KEEP, "gb200-d6144 and the unused d1536 rung removed"),
    "--match": (KEEP, ""),
    "--batch-size": (KEEP, ""),
    "--num-steps": (KEEP, ""),
    "--no-eval": (KEEP, ""),
    "--dense": (KEEP, "belay_tracks Vanilla"),
    "--save-checkpoints": (KEEP, "Ckpt 1 ladder"),
    "--temporary-checkpoints-only": (KEEP, "76 recent runs"),
    "--optimizer": (KEEP, "belay_tracks Vanilla uses adamw"),
    "--recipe": (KEEP, ""),
    "--attn-res-remat-attention": (DEAD, "the d1280 rung sets it; flag unused"),
    "--seed": (KEEP, ""),
    "--profile-start-step": (KEEP, "profiling"),
    "--dump-hlo": (KEEP, "profiling"),
    "--extra-mlp-start-frac": (RESEARCH, "entropy-skip studies"),
    "--layer-skip-layers": (RESEARCH, ""),
    "--layer-skip-rate": (RESEARCH, ""),
    "--layer-skip-positions": (RESEARCH, ""),
    "--layer-skip-token-ids-file": (RESEARCH, ""),
    "--token-dump": (RESEARCH, ""),
    "--token-dump-skip-layers": (RESEARCH, ""),
    "--seed-xla-autotune-cache": (GB200, ""),
    "--watch-interval": (KEEP, ""),
    "--xla-autotune": (GB200, ""),
    "--routing-dump-steps": (DEAD, "no recent run"),
    "--resume-from": (KEEP, "47 recent runs"),
    "--master-param-mode": (GB200, ""),
    "--xla-memory-report": (KEEP, "OOM diagnosis"),
    "--pgle-runs": (GB200, "PGLE plumbing"),
    "--pipeline-depth": (DEAD, "see pipeline_depth"),
    "--loader-buffer-batches": (KEEP, ""),
    "--ema-beta": (DEAD, ""),
    "--ema-last-steps": (DEAD, ""),
    "--max-retries": (KEEP, ""),
    "--z-loss-weight": (KEEP, ""),
    "--ragged-transport": (KEEP, "one_shot / nccl"),
    "--collective-overlap-limit": (DEAD, ""),
    "--single-process": (KEEP, "Ckpt 1 ladder"),
    "--checkpoint-keep": (KEEP, ""),
    "--stop-after-steps": (KEEP, ""),
    "--nodes": (KEEP, "multi-node H100"),
    "--expert-axis-size": (KEEP, "multi-node H100"),
    "--grad-accum": (KEEP, ""),
    "--model-set": (KEEP, ""),
    "--opt-set": (KEEP, ""),
    "--priority": (KEEP, ""),
    "--job-env": (KEEP, ""),
    "--submit": (KEEP, ""),
    "--target-cluster": (KEEP, ""),
    "--version/--override/--run/--max-concurrent": (KEEP, "build_options"),
}

CKPT1 = {
    "--size d512..d1280": (KEEP, "H100 ladder (LADDER_RUNGS)"),
    "--size gb200-d6144": (GB200, "GB200_ARGS / GB200_MODEL_SETTINGS / GB200_XLA_FLAGS removed"),
    "MODEL_SETTINGS moe_drop_renorm=false": (DEAD, "field deleted"),
    "LADDER_ARGS --ragged-transport one_shot": (KEEP, "now the default; kept explicit"),
}


def decision(section: str, name: str) -> str:
    if section == "GrugModelConfig":
        return (
            DEAD
            if name in MODEL_DEAD
            else GB200 if name in MODEL_GB200 else RESEARCH if name in MODEL_RESEARCH else KEEP
        )
    if section == "GrugMoeMuonHConfig":
        return DEAD if name in OPT_DEAD else RESEARCH if name in OPT_RESEARCH else KEEP
    if section == "GrugTrainerConfig":
        return DEAD if name in TRAINER_DEAD else GB200 if name in TRAINER_GB200 else KEEP
    if section == "GrugRunConfig":
        return DEAD if name in RUN_DEAD else KEEP
    if section == "H100LadderRung":
        return GB200 if name in RUNG_GB200 else KEEP
    return KEEP


def cell(d: dict) -> str:
    vals = sorted(set(d.values()))
    return ", ".join(vals) if vals else ""


def main() -> None:
    inv = json.loads(Path("scratchpad/clean/inventory.json").read_text())
    lines = [
        "# fast_track config inventory (marin_belay_v1_w2026-10-19 @ 87dfa66f24)",
        "",
        "Columns: the field's default; the values Belay Ckpt 1 sets at the H100 rungs d512-d1280 (`belay`) and at",
        "`gb200-d6144`; the values the other live launchers set where they differ from the default (`launch.py` default",
        "d512 baseline, `--dense` d512); and how many of the last six days' Iris launches set the field",
        "(`H100 runs`: the",
        "221 H100 research launches on cw-us-east-02a, `all`: 1029 launches including GB200/MFU). `belay_tracks.py`",
        "(Vanilla / Basic MoE) is read from its source. Decisions: keep, keep (research) = only today's d512 studies",
        "use it, delete (dead) = off by default and set by no recipe or launcher on the branch, delete (GB200/MFU) =",
        "only the GB200 recipe or MFU probes reach it.",
        "",
    ]
    counts: dict[str, int] = {}
    for section, rows in inv.items():
        lines += [
            f"## {section} ({len(rows)} fields)",
            "",
            "| field | default | belay (H100) | gb200-d6144 | baseline / dense | H100 runs | all | decision |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            d = decision(section, r["field"])
            counts[d] = counts.get(d, 0) + 1
            note = NOTES.get(r["field"], "")
            dtext = d + (f": {note}" if note else "")
            row = [
                f"`{r['field']}`",
                f"`{r['default']}`",
                cell(r["belay"]),
                r["gb200"],
                cell(r["baseline"]),
                str(r["recent_h100"]),
                str(r["recent_all"]),
                dtext,
            ]
            lines.append("| " + " | ".join(x.replace("|", "\\|") for x in row) + " |")
        lines.append("")
    lines += ["## launch.py options", "", "| option | decision | note |", "|---|---|---|"]
    lines += [f"| `{k}` | {d} | {n} |" for k, (d, n) in LAUNCH_OPTIONS.items()]
    lines += ["", "## belay_ckpt1.py", "", "| option | decision | note |", "|---|---|---|"]
    lines += [f"| `{k}` | {d} | {n} |" for k, (d, n) in CKPT1.items()]
    lines += ["", "## Totals (config fields)", ""] + [f"- {k}: {v}" for k, v in sorted(counts.items())]
    Path("scratchpad/clean/INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(counts)


if __name__ == "__main__":
    main()
