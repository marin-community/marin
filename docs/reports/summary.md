# Chronological Summary of Issues

1. **[Train 1B models on prior datasets](https://github.com/marin-community/marin/issues/72)** - July 31, 2024
2. **[Replicate DCLM OH-2.5 + ELI5 fasttext classifier](https://github.com/marin-community/marin/issues/102)** - August 11, 2024
3. **[Train a simple Dolma/Olmo baseline to flex the pipeline](https://github.com/marin-community/marin/issues/442)** - October 17, 2024
4. **[compare html -> text methods](https://github.com/marin-community/marin/issues/246)** - September 11, 2024
5. **[train MMLU quality classifier](https://github.com/marin-community/marin/issues/274)** - September 18, 2024
6. **[Reproduce Olmo SFT for quickstart](https://github.com/marin-community/marin/issues/227)** - September 9, 2024
7. **[Evaluate baselines at 1B scale on LegalBench](https://github.com/marin-community/marin/issues/230)** - September 10, 2024
8. **[Train 1B models with different amounts of law data, eval on LegalBench](https://github.com/marin-community/marin/issues/231)** - September 10, 2024
9. **[SFT on all turns](https://github.com/marin-community/marin/issues/904)** - March 21, 2025
10. **[Experiment: See if WSD models are less amenable to SFT](https://github.com/marin-community/marin/issues/950)** - April 8, 2025
11. **[Update issues with latest entries; results for synthetic data curation](https://github.com/marin-community/marin/issues/640)** - December 9, 2024
12. **[Update issues with latest entries; run experiments on FineQA](https://github.com/marin-community/marin/issues/958)** - April 12, 2025

# Summary Grouped by Topic

## General Experiments

### [Train 1B models on prior datasets](https://github.com/marin-community/marin/issues/72)

Establishes baselines for datasets including FineWeb, Dolma, and DCLM-Baseline. Goals involve replicating published numbers and aligning configurations using current defaults.

## Quality Classifiers

### [Replicate DCLM OH-2.5 + ELI5 fasttext classifier](https://github.com/marin-community/marin/issues/102)

This experiment replicated the fasttext classifier from the DCLM paper using ELI5 and OH-2.5 data, aiming to measure downstream performance via data filtering and training.

### [train MMLU quality classifier](https://github.com/marin-community/marin/issues/274)

Training a classifier on MMLU versus non-MMLU data to analyze impacts on MMLU scores. It aimed to correlate classifier performance on distinguishing MMLU with the resulting LM's performance.

## SFT and Instruction Following

### [Reproduce Olmo SFT for quickstart](https://github.com/marin-community/marin/issues/227) & [SFT on all turns](https://github.com/marin-community/marin/issues/904) & [Experiment: See if WSD models are less amenable to SFT](https://github.com/marin-community/marin/issues/950)

These experiments focus on SFT (Supervised Fine-Tuning) and its interaction with model architectures and data inputs, investigating potential improvements or identifying hindrances such as WSD model limitations.

## Legal Domain Experiments

### [Evaluate baselines at 1B scale on LegalBench](https://github.com/marin-community/marin/issues/230) & [Train 1B models with different amounts of law data, eval on LegalBench](https://github.com/marin-community/marin/issues/231)

These efforts target the legal domain using LegalBench benchmarks to assess performance influenced by varying legal data proportions, scrutinizing model enhancements through substantial legal dataset integration. 

## Synthetic Data Curation

### [Update issues with latest entries; results for synthetic data curation](https://github.com/marin-community/marin/issues/640) & [Update issues with latest entries; run experiments on FineQA](https://github.com/marin-community/marin/issues/958)

Focused on generating diverse instruction-following data and testing the effectiveness through curated datasets like FineQA, highlighting performance impacts on models trained with enhanced synthetic content. 

## HTML to Text Conversion

### [compare html -> text methods](https://github.com/marin-community/marin/issues/246)

This issue compared different HTML-to-text conversion methods to identify the most efficient way to preprocess data for language model training, aiming for improved accuracy and content retention.
