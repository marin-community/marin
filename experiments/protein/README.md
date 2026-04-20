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

## ProteinMPNN sequence redesign

To eval the model on redesigned sequences (prompt model with MPNN-designed
sequence, score against native CA-CA distances):

```bash
# 1. Run the redesign script (local, CPU is fine):
uv run python -m experiments.protein.redesign_sequences \
    --output gs://marin-us-east5/protein-structure/protein-mpnn-redesigns/v1/redesigns.jsonl \
    --temperature 0.1 --seed 0

# 2. Next training run picks up the redesigns automatically via
#    train_protein_1b.py's `PROTEIN_MPNN_REDESIGNS_SOURCE` setting.
```

Each record in the JSONL produces a new `ProteinTarget` labeled
`<base_label>-mpnn-<idx>`, with the MPNN sequence in `sequence_override` and
`sequence_origin="mpnn"`. Metrics appear at `eval/protein_dist/<target>-mpnn-0/...`
plus aggregated at `eval/protein_dist/origin=mpnn/...`.

Requirements: `git`, `torch`, `biopython`. The script auto-clones ProteinMPNN
into `third_party/ProteinMPNN/` on first use.

## Offline distogram analysis (for plots and detailed dives)

When loss isn't enough and you want heatmaps / E|err| / PMFs:

```bash
# 1. Export a specific checkpoint to HF format:
uv run iris --config=lib/iris/examples/marin.yaml job run \
    --memory=32GB --disk=16GB --cpu=4 \
    -- python -m experiments.protein.export_protein_1b

# 2. Run the vLLM-based distogram eval:
uv run iris --config=lib/iris/examples/marin.yaml job run \
    --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu \
    -- python -m experiments.protein.eval_protein_distogram \
        --model gs://.../hf --pdb-id 1QYS \
        --prompt-contact-counts 0 1 2 3 4 5 \
        --output-dir gs://.../protein-distogram/<run>

# 3. Build a PDF report:
uv run python -m experiments.protein.plot_distogram \
    --input-dir gs://.../protein-distogram/<run> \
    --output-pdf /Users/tim/Dropbox/OpenAthena/projects/202604-LlamaFold/reports/<run>.pdf
```
