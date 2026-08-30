STAGE3_CHECKPOINT = (
    "s3://marin-us-east-02a/marin/grug/snowball_step105149_sft_s3_agentic_eot_5ep/"
    "2026.08.13.1/checkpoints/step-1888/"
)
STAGE3_OUTPUT = (
    "s3://marin-us-east-02a/marin/exports/grug/snowball_step105149_sft_s3_agentic_eot_5ep/"
    "2026.08.13.1/step-1888/hf-bf16-vllm/"
)


def main() -> None:
    source_path = __file__.replace("export_second_cooldown_stage3.py", "export_second_cooldown_stage2.py")
    with open(source_path) as source_file:
        source = source_file.read()
    stage2_checkpoint = (
        'CHECKPOINT = (\n'
        '    "s3://marin-us-east-02a/marin/grug/"\n'
        '    "snowball_step105149_sft_s2_thinking/"\n'
        '    "2026.08.13.1/checkpoints/step-630/"\n'
        ')'
    )
    stage2_output = (
        'OUTPUT = (\n'
        '    "s3://marin-us-east-02a/marin/exports/grug/"\n'
        '    "snowball_step105149_sft_s2_thinking/2026.08.13.1/step-630/hf-bf16-vllm/"\n'
        ')'
    )
    assert stage2_checkpoint in source
    assert stage2_output in source
    source = source.replace(stage2_checkpoint, f"CHECKPOINT = {STAGE3_CHECKPOINT!r}")
    source = source.replace(stage2_output, f"OUTPUT = {STAGE3_OUTPUT!r}")
    exec(compile(source, source_path, "exec"), {"__name__": "__main__", "__file__": source_path})


if __name__ == "__main__":
    main()
