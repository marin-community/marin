# Selected Experiment Reports

This is a semi-automatically generated list of experiment issues and reports. It is periodically updated by a script,
and then curated by hand.

This page includes only experiments that have at least one run or report.

## Marin 8B Base

- Tootsie 8B (Main Issue) [![#600](https://img.shields.io/github/issues/detail/state/marin-community/marin/600)](https://github.com/marin-community/marin/issues/600)
    - [GitHub Issue #600](https://github.com/marin-community/marin/issues/600)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Tootsie-8B---VmlldzoxMTY3MzU3OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp600_tootsie-4c95ae.json)

### Cooldowns

- Try deepening the cooldown of "monumental-jellyfish" (tootsie 8b cooldown 1) to see if it improves SFT [![#898](https://img.shields.io/github/issues/detail/state/marin-community/marin/898)](https://github.com/marin-community/marin/issues/898)
    - [GitHub Issue #898](https://github.com/marin-community/marin/issues/898)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/898-Tootsie-Soft-Raccoon--VmlldzoxMTk3NjUwNg?accessToken=06f87pmmvhdulczenkg3349jxk7e1pwbd4pdci2i8wvyxg9289122gfnckr9ymwc)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp898_deeper_cooldown-242d4a.json)
- not-quite-so-deep cooldown (Spoonbill) [![#916](https://img.shields.io/github/issues/detail/state/marin-community/marin/916)](https://github.com/marin-community/marin/issues/916)
    - [GitHub Issue #916](https://github.com/marin-community/marin/issues/916)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/916-Tootsie-Hypnotic-Spoonbill--VmlldzoxMjA1NjU2Nw)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp916_tootsie_spoonbill_cooldown-9f5976.json)
    - Conclusion: Exploding logits in deep parts of cooldown can be mitigated by Z-Loss.
- Tootsie Phoenix Cooldown (sensible-starling) [![#977](https://img.shields.io/github/issues/detail/state/marin-community/marin/977)](https://github.com/marin-community/marin/issues/977)
    - [GitHub Issue #977](https://github.com/marin-community/marin/issues/977)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Tootsie-8B-phoenix-cooldown-starling---VmlldzoxMjQ2MjM5Ng)
    - [WandB Run: tootsie-8b-sensible-starling](https://wandb.ai/marin-community/marin/runs/tootsie-8b-sensible-starling?nw=nwuserdlwh)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp977_phoenix_cooldown-8e3456.json)

## Big Runs

- \[EPIC\] Big Runs [![#859](https://img.shields.io/github/issues/detail/state/marin-community/marin/859)](https://github.com/marin-community/marin/issues/859)
    - [GitHub Issue #859](https://github.com/marin-community/marin/issues/859)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA?accessToken=st1rajwy32etqi5rrlm3kuhgqa4ods6fnwsbyk8azjc8ar3eikf4dnz1p2ldz8yx)
- Marin 13B [![#860](https://img.shields.io/github/issues/detail/state/marin-community/marin/860)](https://github.com/marin-community/marin/issues/860)
    - [GitHub Issue #860](https://github.com/marin-community/marin/issues/860)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA?accessToken=st1rajwy32etqi5rrlm3kuhgqa4ods6fnwsbyk8azjc8ar3eikf4dnz1p2ldz8yx)
- Marin 24B [![#861](https://img.shields.io/github/issues/detail/state/marin-community/marin/861)](https://github.com/marin-community/marin/issues/861)
    - [GitHub Issue #861](https://github.com/marin-community/marin/issues/861)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA)
- Marin 70b [![#750](https://img.shields.io/github/issues/detail/state/marin-community/marin/750)](https://github.com/marin-community/marin/issues/750)
    - [GitHub Issue #750](https://github.com/marin-community/marin/issues/750)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA)

## Modeling

- Pick Tokenizer type [![#524](https://img.shields.io/github/issues/detail/state/marin-community/marin/524)](https://github.com/marin-community/marin/issues/524)
    - [GitHub Issue #524](https://github.com/marin-community/marin/issues/524)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Tokenizer-Comparison--VmlldzoxMDI0Njg3Nw)
    - Conclusion: Llama3 tokenizer is the best.
- Default z-loss? [![#935](https://img.shields.io/github/issues/detail/state/marin-community/marin/935)](https://github.com/marin-community/marin/issues/935)
    - [GitHub Issue #935](https://github.com/marin-community/marin/issues/935)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/ZLoss-vs-Not-1-4B--VmlldzoxMjEzMzA1NA)
    - Conclusion: z-loss seems not harmful. We'll use it.
- Figuring out learning rate schedule! [![#764](https://img.shields.io/github/issues/detail/state/marin-community/marin/764)](https://github.com/marin-community/marin/issues/764)
    - [GitHub Issue #764](https://github.com/marin-community/marin/issues/764)
    - [WandB Report](https://wandb.ai/marin-community/marin-optimizer/reports/Deciding-the-optimal-lr-schedule-which-is-cosine---VmlldzoxMTIxNDk5NA)
    - Conclusion: Cosine is best. High LR is important. WSD isn't terrible.
- Mixture of Experts [![#929](https://img.shields.io/github/issues/detail/state/marin-community/marin/929)](https://github.com/marin-community/marin/issues/929)
  - [GitHub Issue #929](https://github.com/marin-community/marin/issues/929)
  - [WandB Report](https://api.wandb.ai/links/marin-community/0lspgzn3)
- Hybrid Norm and Input Embedding Norm [![#961](https://img.shields.io/github/issues/detail/state/marin-community/marin/961)](https://github.com/marin-community/marin/issues/961)
  - [GitHub Issue #961](https://github.com/marin-community/marin/issues/961)
  - [WandB Report](https://wandb.ai/marin-community/hybrid-norm/reports/Hybrid-Norm--VmlldzoxMjY2MDgxMA)

## Training and Performance

- INT8 training in Levanter [![#620](https://img.shields.io/github/issues/detail/state/marin-community/marin/620)](https://github.com/marin-community/marin/issues/620)
    - [GitHub Issue #620](https://github.com/marin-community/marin/issues/620)
    - [WandB Report](https://api.wandb.ai/links/marin-community/yhrb0xik)
    - Conclusion: Int8 training is much faster on the right hardware, but might lead to worse performance in terms of time-to-loss except in the early stages.
- MuP for scaling laws [![#621](https://img.shields.io/github/issues/detail/state/marin-community/marin/621)](https://github.com/marin-community/marin/issues/621)
    - [GitHub Issue #621](https://github.com/marin-community/marin/issues/621)
    - [WandB Report](https://api.wandb.ai/links/marin-community/h723u2ws)
    - Conclusion: not worth it compared to our heuristic version.
- Figuring out learning rate schedule! [![#764](https://img.shields.io/github/issues/detail/state/marin-community/marin/764)](https://github.com/marin-community/marin/issues/764)
    - [GitHub Issue #764](https://github.com/marin-community/marin/issues/764)
    - [WandB Report](https://wandb.ai/marin-community/marin-optimizer/reports/Deciding-the-optimal-lr-schedule-which-is-cosine---VmlldzoxMTIxNDk5NA)
- Try out different remat strategies to get the 70b working on fewer slices [![#906](https://img.shields.io/github/issues/detail/state/marin-community/marin/906)](https://github.com/marin-community/marin/issues/906)
    - [GitHub Issue #906](https://github.com/marin-community/marin/issues/906)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Remat-Strategies--VmlldzoxMTkxNzk3Ng)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/expXXX_fancier_checkpointing-453042.json)
    - Conclusion: Substantial performance hit but helpful. Still need to iterate.

## Data Experiments

### Datashop
- MEDU filtering across MMLU subsets [![#923](https://img.shields.io/github/issues/detail/state/stanford-crfm/marin/923)](https://github.com/marin-community/marin/issues/923)
   - [GitHub Issue #923](https://github.com/marin-community/marin/issues/923)
   - [WandB Report](https://api.wandb.ai/links/marin-community/9b1v0zwa)
   - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment?path=gs%3A%2F%2Fmarin-us-east1%2Fexperiments%2Fmedu_mmlu-051285.json)
- Reproduce Finemath performance using Datashop [![#939](https://img.shields.io/github/issues/detail/state/stanford-crfm/marin/939)](https://github.com/marin-community/marin/issues/939) [![#939](https://img.shields.io/github/issues/detail/state/stanford-crfm/marin/939)](https://github.com/marin-community/marin/issues/939)
   - [GitHub Issue #939](https://github.com/marin-community/marin/issues/939)
   - [WandB Report](https://wandb.ai/marin-community/marin/reports/Reproducing-Finemath--VmlldzoxMjc2NDMyNg?accessToken=xnwbosaz2es5lkoyjmtclq0xoqbbqzyyg7gjt8qkmjz7zs0vx1p4g8t5nloyx9ft)


### High Quality Data Ablations

- Ablations on Cooldown for Markdownified Wikipedia [![#845](https://img.shields.io/github/issues/detail/state/marin-community/marin/845)](https://github.com/marin-community/marin/issues/845)
    - [GitHub Issue #845](https://github.com/marin-community/marin/issues/845)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/845-6-Wiki-and-Arxiv-Quality-Ablations--VmlldzoxMTg4MzY2OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-custom-fork-2569de%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%5D)
    - Conclusion: No major improvement compared to control.
- Ablations on Cooldown for Markdownified Arxiv [![#846](https://img.shields.io/github/issues/detail/state/marin-community/marin/846)](https://github.com/marin-community/marin/issues/846)
    - [GitHub Issue #846](https://github.com/marin-community/marin/issues/846)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/845-6-Wiki-and-Arxiv-Quality-Ablations--VmlldzoxMTg4MzY2OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Far5iv%2Far5iv-04-2024-no-problem-3971ff%2Fresiliparse-custom-fork%2F0001.jsonl.gz%22%5D)
    - Conclusion: No major improvement compared to control.
- Ablations on Cooldown for Markdownified StackExchange [![#847](https://img.shields.io/github/issues/detail/state/marin-community/marin/847)](https://github.com/marin-community/marin/issues/847)
    - [GitHub Issue #847](https://github.com/marin-community/marin/issues/847)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/845-6-Wiki-and-Arxiv-Quality-Ablations--VmlldzoxMTg4MzY2OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fstackexchange-resiliparse-custom-fork-ab41ad%2F3dprinting.jsonl.gz%22%5D)
    - Conclusion: No major improvement compared to control.
- Mixture of Formats Training on Wikipedia and Arxiv [![#818](https://img.shields.io/github/issues/detail/state/marin-community/marin/818)](https://github.com/marin-community/marin/issues/818)
    - [GitHub Issue #818](https://github.com/marin-community/marin/issues/818)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/818-Mixture-of-Formats--VmlldzoxMTg4MzU0NA)
    - Conclusion: No major difference observed, switch to @Helw150's annealing setup for evaluations.
- High Quality Many Epochs vs. Low Quality Few Epochs [![#636](https://img.shields.io/github/issues/detail/state/marin-community/marin/636)](https://github.com/marin-community/marin/issues/636)
    - [GitHub Issue #636](https://github.com/marin-community/marin/issues/636)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/High-Quality-Many-Epochs-vs-Lower-quality-fewer-epoch--VmlldzoxMDU2MTI1Mg)
    - Conclusion: There's no data like more data.

### Data Filtering

- Stack Exchange Quality Classifier [![#596](https://img.shields.io/github/issues/detail/state/marin-community/marin/596)](https://github.com/marin-community/marin/issues/596)
    - [GitHub Issue #596](https://github.com/marin-community/marin/issues/596)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/Quality-Classifier-Comparison--VmlldzoxMDI2MzI1MQ)
    - Conclusion: Seems to lead to better loss than using Reddit ELI5 or OpenHermes.
    - NOTE: this seems like a loose end, we should pursue this further.

### Text Extraction and Formatting

- Compare HTML -> text methods [![#246](https://img.shields.io/github/issues/detail/state/marin-community/marin/246)](https://github.com/marin-community/marin/issues/246)
    - [GitHub Issue #246](https://github.com/marin-community/marin/issues/246)
    - [WandB Report](https://api.wandb.ai/links/marin-community/0uoys8gp)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/246_web_extraction_method_training-efe0cf.json)
    - Conclusion: some amount of format preservation is helpful for loss on Paloma.
- Wikipedia Training Runs with DOLMA source substitution [![#647](https://img.shields.io/github/issues/detail/state/marin-community/marin/647)](https://github.com/marin-community/marin/issues/647)
    - [GitHub Issue #647](https://github.com/marin-community/marin/issues/647)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/647-Wikipedia-Training-Runs-with-DOLMA-source-substitution--VmlldzoxMDkyNjIxNw)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-no-references-no-links-cleaned-7971bb%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%2C%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-no-references-with-links-cleaned-b89dd3%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%2C%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-with-references-no-links-cleaned-0fd095%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%2C%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-with-references-with-links-cleaned-infobox-0203ff%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%5D)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fexperiments%2Fexp647_wikipedia_training-dd35e8.json%22%5D)
- Ar5iv Training Runs with DOLMA source substitution [![#648](https://img.shields.io/github/issues/detail/state/marin-community/marin/648)](https://github.com/marin-community/marin/issues/648)
    - [GitHub Issue #648](https://github.com/marin-community/marin/issues/648)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fexperiments%2Fexp648_ar5iv_training-f49d7f.json%22%5D)
- [Markdownification Processing Report](./markdownified-datasets.md)

## Supervised Fine Tuning

- Reproduce olmov2 SFT [![#606](https://img.shields.io/github/issues/detail/state/marin-community/marin/606)](https://github.com/marin-community/marin/issues/606)
    - [GitHub Issue #606](https://github.com/marin-community/marin/issues/606)
    - [WandB Run: marin_olmo_tulu_sft_v3-acca67](https://wandb.ai/marin-community/marin/runs/marin_olmo_tulu_sft_v3-acca67)
- Create Mixture from All SFT Datasets [![#804](https://img.shields.io/github/issues/detail/state/marin-community/marin/804)](https://github.com/marin-community/marin/issues/804)
    - [GitHub Issue #804](https://github.com/marin-community/marin/issues/804)
    - [WandB Report](https://api.wandb.ai/links/marin-community/m8ak7uah)
- SFT on further cool-downed tootsie checkpoints [![#897](https://img.shields.io/github/issues/detail/state/marin-community/marin/897)](https://github.com/marin-community/marin/issues/897)
    - [GitHub Issue #897](https://github.com/marin-community/marin/issues/897)
- SFT Deeper Starling [![#1237](https://img.shields.io/github/issues/detail/state/stanford-crfm/marin/1237)](https://github.com/marin-community/marin/issues/1237)
  - [GitHub Issue #1237](https://github.com/marin-community/marin/issues/1237)
  - [WandB Run: deeper_mixture_sft_starling_1e-4-longer-2](https://wandb.ai/marin-community/marin/runs/deeper_mixture_sft_starling_1e-4-longer-2?nw=nwuserheld)

## Scaling Laws

- Scaling laws to predict tootsie performance [![#654](https://img.shields.io/github/issues/detail/state/marin-community/marin/654)](https://github.com/marin-community/marin/issues/654)
    - [GitHub Issue #654](https://github.com/marin-community/marin/issues/654)
    - [WandB Report](https://api.wandb.ai/links/marin-community/3xojrl9v)
    - [WandB Report on Soft Metrics](https://api.wandb.ai/links/marin-community/got35r4i)
- Optimizer Scaling Law Part 1: AdamW [![#725](https://img.shields.io/github/issues/detail/state/marin-community/marin/725)](https://github.com/marin-community/marin/issues/725)
    - [GitHub Issue #725](https://github.com/marin-community/marin/issues/725)
    - [WandB Report](https://wandb.ai/marin-community/marin-optimizer/reports/AdamW-Sweeping--VmlldzoxMTE3Nzc5OA)
    - Conclusion: After sweeping, we discovered that the (near) optimal set of hyperparameters for AdamW remains surprisingly stable across three settings.

## Baselines and Reproductions

- Train a simple Dolma/Olmo baseline to flex the pipeline [![#442](https://img.shields.io/github/issues/detail/state/marin-community/marin/442)](https://github.com/marin-community/marin/issues/442)
    - [GitHub Issue #442](https://github.com/marin-community/marin/issues/442)
    - [WandB Report](https://api.wandb.ai/links/marin-community/e20j5423)
- Build DCLM 7b baseline [![#143](https://img.shields.io/github/issues/detail/state/marin-community/marin/143)](https://github.com/marin-community/marin/issues/143)
    - [GitHub Issue #143](https://github.com/marin-community/marin/issues/143)
    - [WandB Report](https://wandb.ai/marin-community/marin/reports/DCLM-7B-Replication--Vmlldzo5MTA3NjU5/edit)

## Other Projects

### Compel
- Compression-Ratio Quality Filter 1.4B Models [![#633](https://img.shields.io/github/issues/detail/state/marin-community/marin/633)](https://github.com/marin-community/marin/issues/633)
    - [GitHub Issue #633](https://github.com/marin-community/marin/issues/633)
    - [WandB Run: compel-fineweb-edu-baseline](https://wandb.ai/marin-community/marin/runs/compression-fineweb-edu-0.6-0.8-da53bf?nw=nwusereobbad)
    - [WandB Run: compel-fineweb-edu-0.65-0.8](https://wandb.ai/marin-community/marin/runs/compression-ratio-filter-fineweb-edu-786908?nw=nwusereobbad)
    - [WandB Run: compel-fineweb-baseline](https://wandb.ai/marin-community/marin/runs/compression-train-full-dataset-llama1.4b-20fa75?nw=nwusereobbad)
    - [WandB Run: compel-fineweb-0.65-0.8-e5cdae](https://wandb.ai/marin-community/marin/runs/compression-ratio-filter-llama1.4b-0.6-0.8-e5cdae?nw=nwusereobbad)


## Uncategorized

(This is for experiments that have been added via the script but have not yet been curated.)
