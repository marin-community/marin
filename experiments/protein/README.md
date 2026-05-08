# Protein-docs experiments

Train and evaluate the `protein-docs` LM (Llama 1.4B) on the
`contacts-and-distances-v1-5x` document format.

## Files

| File | Purpose |
|---|---|
| `create_protein_tokenizer.py` | Build + push the custom WordLevel tokenizer to HF Hub. |
| `train_protein_1b.py` | Main training experiment. Includes the distogram benchmark as validation sets. |
| `export_protein_1b.py` | Convert a Levanter checkpoint to HuggingFace format. |
| `protein_distogram_eval.py` | Distogram benchmark: registry of target PDBs + builder for pre-tokenized validation datasets. |
| `redesign_sequences.py` | Local script that runs ProteinMPNN to redesign benchmark sequences. |
| `eval_protein_contacts.py` | Offline contact-prediction eval (forced-scaffold generation via vLLM). |
| `eval_protein_distogram.py` | Offline distogram eval (single-token queries via vLLM). Produces detailed artifacts for plotting. |
| `plot_distogram.py` | Generate per-run PDF reports from distogram eval outputs. |

## In-training evaluation (the distogram benchmark)

Every `steps_per_eval` steps, levanter's standard tagged-eval harness computes
cross-entropy loss on pre-tokenized validation examples from our benchmark.

The benchmark consists of 1000 randomly-sampled CA–CA residue pairs per
(target, N) combination, where:
- **target**: a PDB entry (see `TARGETS` in `protein_distogram_eval.py`)
- **N** ∈ {0, 1, 2, 3, 4, 5}: number of ground-truth long-range contacts
  prepended to the prompt before the `<distance>` query

Metrics appear in wandb as:

```
eval/protein_dist/loss                     # micro-avg over everything (token-weighted)
eval/protein_dist/macro_loss               # dataset-weighted macro across leaf (target, N)
eval/protein_dist/<target>/macro_loss      # per-target macro across N
eval/protein_dist/N<N>/macro_loss          # per-N macro across targets
eval/protein_dist/<target>/N<N>/loss       # leaf loss
eval/protein_dist/origin=mpnn/macro_loss   # across redesigned targets (if present)
```

(Replace `loss` with `bpb` for bits-per-byte versions.)

### Adding a new target

1. Append a `ProteinTarget(...)` to `TARGETS` in `protein_distogram_eval.py`.
2. Re-run `experiments/protein/train_protein_1b.py` (or its iris command).
   Marin's executor auto-builds 6 new parquets (one per N) and caches them.

Fields:
- `pdb_id`: RCSB PDB ID, e.g. `"1QYS"`.
- `chain_id`: optional; defaults to the first chain in the fetched file.
- `assembly`: 0 (default) = deposited asym unit, 1+ = biological assembly.
  For FoldBench-style entries (monomeric assemblies) use `assembly=1`.
- `label`: optional display name; defaults to `pdb_id.lower()`.
- `sequence_override` / `sequence_origin`: see MPNN redesign section below.

Caveats:
- Target's first chain must fit in the 8192-token training seq length
  (roughly ≤ 7000 residues — not a concern for reasonable proteins).
- Target must have ≥ 5 long-range (|i-j| ≥ 24) GT contacts to support N=5.

## ProteinMPNN / SolubleMPNN sequence redesign

To eval the model on redesigned sequences (prompt with MPNN-designed sequence,
score against native CA-CA distances):

```bash
# 1. Run the redesign script (local, CPU is fine; ~1 s per target).
#    --method picks the backbone-network weights:
#      "mpnn"    (default) = stock ProteinMPNN
#      "soluble" = SolubleMPNN (same code + --use_soluble_model, weights trained
#                  on soluble proteins only).
uv run python -m experiments.protein.redesign_sequences \
    --method soluble \
    --output gs://marin-us-east5/protein-structure/protein-mpnn-redesigns/soluble-v1/redesigns.jsonl \
    --targets top7 7bny \
    --num-redesigns 2 \
    --temperature 0.1 --seed 0

# 2. Next training run picks up the redesigns automatically via
#    train_protein_1b.py's `PROTEIN_MPNN_REDESIGNS_SOURCE` setting.
```

Each JSONL record produces a new `ProteinTarget` labeled
`<base_label>-<method>-<idx>`, with the redesigned sequence in
`sequence_override` and `sequence_origin=<method>`. In-training metrics appear
at `eval/protein_dist/<target>-<method>-<idx>/...` plus aggregated at
`eval/protein_dist/origin=<method>/...`.

Requirements: `git`, `torch`, `biopython`. The script auto-clones ProteinMPNN
into `third_party/ProteinMPNN/` on first use (or point `PROTEINMPNN_DIR` at an
existing clone).

## Offline distogram analysis (for plots and detailed dives)

When loss isn't enough and you want heatmaps / E|err| / PMFs:

```bash
# 1. Export a specific checkpoint to HF format. `export_protein_1b.py` pins the
#    unmasked 2.5e-4 run; `export_protein_1b_distance_masked.py` pins the
#    3.5e-4 distance-masked run at step-8432. Mirror those for other runs.
uv run iris --config=lib/iris/examples/marin.yaml job run \
    --memory=32GB --disk=16GB --cpu=4 \
    -- python -m experiments.protein.export_protein_1b_distance_masked

# 2a. Run the vLLM-based distogram eval on the *native* PDB sequence:
uv run iris --config=lib/iris/examples/marin.yaml job run \
    --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu \
    -- python -m experiments.protein.eval_protein_distogram \
        --model gs://.../hf --pdb-id 1QYS \
        --prompt-contact-counts 0 1 2 3 4 5 \
        --output-dir gs://.../protein-distogram/<run>/1qys/run-01

# 2b. Or eval a *redesigned* sequence (SolubleMPNN shown). Prompt's
#     <begin_sequence> uses the override; ground-truth CA coords still come
#     from the PDB — the question is "given this redesigned sequence, what
#     CA–CA distances does the model predict for the native-like structure?"
uv run iris --config=lib/iris/examples/marin.yaml job run \
    --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu \
    -- python -m experiments.protein.eval_protein_distogram \
        --model gs://.../hf --pdb-id 1QYS \
        --prompt-contact-counts 0 1 2 3 4 5 \
        --sequence-override-source gs://.../soluble-v1/redesigns.jsonl \
        --sequence-override-target-label top7 \
        --sequence-override-method soluble \
        --sequence-override-idx 0 \
        --output-dir gs://.../protein-distogram/<run>/top7-soluble-0/run-01

# 3. Build a PDF report:
uv run python -m experiments.protein.plot_distogram \
    --input-dir gs://.../protein-distogram/<run>/<variant>/run-01 \
    --output-pdf /Users/tim/Dropbox/OpenAthena/projects/202604-LlamaFold/reports/<run>-<variant>.pdf
```

## ProteinGym DMS-substitutions zero-shot eval

Treats the model as a (causal) protein language model and scores variants in
ProteinGym's [DMS substitutions benchmark](https://proteingym.org/benchmarks)
by log-likelihood ratio under the document prefix
`<contacts-and-distances-v1> <begin_sequence> <AA_1>...<AA_n>`.

- Single-mutant `score(i, ref->alt)` = `log P(<alt>|prefix_<i) - log P(<ref>|prefix_<i)`,
  pulled from a single forward pass on the WT sequence.
- Multi-mutant variants get an additional forward pass over the mutated
  sequence and compute the joint LL difference (sum over positions).

Position-bias caveat: this is a causal LM, so the model has no left context at
the N-terminus and progressively more toward the C-terminus. The eval always
emits per-position perplexity profiles and position-stratified Spearman so
the bias is visible.

Inference path: HuggingFace transformers + torch (CPU by default; GPU if
`torch.cuda.is_available()`). vLLM's TPU backend doesn't yet support
`prompt_logprobs` (vllm-project/tpu-inference#1333), and the 1B model fits in
3 GB bf16, so a CPU forward pass on 16 vCPU is fine for ~hundreds of variants
per dataset. For the bigger DMS (GFP/GRB2/PABP have 30k-60k variants), GPU
makes sense.

```bash
# Data: ProteinGym v1.3 (217 datasets) is mirrored at
#   gs://marin-us-east5/protein-structure/proteingym/v1.3/
#     reference/DMS_substitutions.csv                  (target sequences + metadata)
#     reference/DMS_substitutions_Spearman_DMS_level.csv  (leaderboard, 102 methods x 217 datasets)
#     DMS_ProteinGym_substitutions/<DMS_id>.csv         (per-dataset variants with `mutant`, `mutated_sequence`, `DMS_score`)

# 20-dataset representative subset (mix of size, organism, selection type):
#   BLAT_ECOLX_Stiffler_2015 TPMT_HUMAN_Matreyek_2018 CALM1_HUMAN_Weile_2017
#   GFP_AEQVI_Sarkisyan_2016 P53_HUMAN_Giacomelli_2018_Null_Etoposide
#   PA_I34A1_Wu_2015 PABP_YEAST_Melamed_2013 GRB2_HUMAN_Faure_2021
#   KKA2_KLEPN_Melnikov_2014 BRCA1_HUMAN_Findlay_2018 UBE4B_MOUSE_Starita_2013
#   YAP1_HUMAN_Araya_2012 MK01_HUMAN_Brenan_2016 SUMO1_HUMAN_Weile_2017
#   HSP82_YEAST_Mishra_2016 CCDB_ECOLI_Adkar_2012 A4_HUMAN_Seuma_2022
#   ENV_HV1B9_DuenasDecamp_2016 BLAT_ECOLX_Jacquier_2013 NCAP_I34A1_Doud_2015

# CPU run (one iris worker, ~minutes per dataset for single-mut, longer for multi-mut datasets):
HF_TOKEN=... uv run iris --config=lib/iris/examples/marin.yaml job run \
    --memory=32GB --disk=64GB --cpu=16 --extra=cpu --zone=us-east5-a \
    --enable-extra-resources \
    -e WANDB_API_KEY $WANDB_API_KEY -e HF_TOKEN $HF_TOKEN \
    -- python -m experiments.protein.eval_protein_proteingym \
        --model gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e/hf/step-15049 \
        --proteingym-dir gs://marin-us-east5/protein-structure/proteingym/v1.3 \
        --datasets BLAT_ECOLX_Stiffler_2015 ... NCAP_I34A1_Doud_2015 \
        --output-dir gs://marin-us-east5/eval/protein-proteingym/<run>/step-15049/subset20

# GPU run (any single-GPU box, model auto-loads on cuda):
python -m experiments.protein.eval_protein_proteingym \
    --model gs://.../hf/step-15049 \
    --proteingym-dir gs://marin-us-east5/protein-structure/proteingym/v1.3 \
    --output-dir gs://.../full217 \
    --device cuda

# PDF report (run anywhere with matplotlib + fsspec):
python -m experiments.protein.plot_protein_proteingym \
    --input-dir gs://.../subset20 \
    --proteingym-dir gs://marin-us-east5/protein-structure/proteingym/v1.3 \
    --output /Users/tim/Dropbox/.../reports/proteingym-step-15049-subset20.pdf
```

Output layout per run:
```
<output-dir>/
  summary.json                        # per-dataset Spearman + position-stratified
  per_dataset/<DMS_id>/
    aa_logprobs.npz                   # (n, 20) WT-conditioned logprobs at each AA position
    per_variant.json                  # row-by-row: mutant string, model_score, position bucket
```

The plotter additionally pulls the leaderboard CSV (`DMS_substitutions_Spearman_DMS_level.csv`)
and overlays 6 reference baselines (ESM-1v, ESM-2 650M/3B, Tranception L,
ProGen2 medium, Site-Independent) on the headline bar chart.

## Continuing an existing training run

When a training run is preempted or we want more steps past the current
checkpoint, create a thin wrapper that pins `override_output_path` to the run's
existing output directory. `continue_train_protein_1b_distance_masked.py` is an
example: it reproduces the original 3.5e-4 distance-masked config and pins the
output path to `.../protein-contacts-1b-3.5e-4-distance-masked-7d355e/`.
Levanter auto-detects the latest checkpoint in that dir and resumes from it.

## Gotcha: marin step-hashing only sees `versioned()` fields

Marin's executor hashes a step's config by walking only `VersionedValue` and
`InputName` entries — plain scalar dataclass fields are **ignored**. Wrap
semantically-significant fields in `versioned(...)` to force a cache-bust when
they change. `BuildDistogramEvalDataConfig` wraps `pdb_id`, `chain_id`,
`assembly`, `n_prompt_contacts`, `tokenizer`, `n_pairs`, `seed`, `max_seq_len`,
and `sequence_override` for this reason. Forgetting to wrap a field silently
reuses stale caches on the next run.
