# Selected Experiment Reports

This is a semi-automatically generated list of experiment issues and reports. It is periodically updated by a script,
and then curated by hand.

This page includes only experiments that have at least one run or report.

## Tootsie Runs

### Tootsie 8b

- Tootsie 8B (Main Issue)
    - [GitHub Issue #600](https://github.com/marin-community/marin/issues/600) [![#600](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/600)](https://github.com/marin-community/marin/issues/600)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Tootsie-8B---VmlldzoxMTY3MzU3OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp600_tootsie-4c95ae.json)

#### Cooldowns

- Try deepening the cooldown of "monumental-jellyfish" (tootsie 8b cooldown 1) to see if it improves SFT
    - [GitHub Issue #898](https://github.com/marin-community/marin/issues/898) [![#898](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/898)](https://github.com/marin-community/marin/issues/898)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/898-Tootsie-Soft-Raccoon--VmlldzoxMTk3NjUwNg?accessToken=06f87pmmvhdulczenkg3349jxk7e1pwbd4pdci2i8wvyxg9289122gfnckr9ymwc)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp898_deeper_cooldown-242d4a.json)
- not-quite-so-deep cooldown (Spoonbill)
    - [GitHub Issue #916](https://github.com/marin-community/marin/issues/916) [![#916](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/916)](https://github.com/marin-community/marin/issues/916)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/916-Tootsie-Hypnotic-Spoonbill--VmlldzoxMjA1NjU2Nw)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp916_tootsie_spoonbill_cooldown-9f5976.json)
    - Conclusion: Exploding logits in deep parts of cooldown can be mitigated by Z-Loss.
- Tootsie Phoenix Cooldown (sensible-starling)
    - [GitHub Issue #977](https://github.com/marin-community/marin/issues/977) [![#977](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/977)](https://github.com/marin-community/marin/issues/977)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Tootsie-8B-phoenix-cooldown-starling---VmlldzoxMjQ2MjM5Ng)
    - [WandB Run: tootsie-8b-sensible-starling](https://wandb.ai/stanford-mercury/marin/runs/tootsie-8b-sensible-starling?nw=nwuserdlwh)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp977_phoenix_cooldown-8e3456.json)

## Big Tootsies

- \[EPIC\] Big Tootsies
    - [GitHub Issue #859](https://github.com/marin-community/marin/issues/859) [![#859](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/859)](https://github.com/marin-community/marin/issues/859)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA?accessToken=st1rajwy32etqi5rrlm3kuhgqa4ods6fnwsbyk8azjc8ar3eikf4dnz1p2ldz8yx)
- Tootsie 13B
    - [GitHub Issue #860](https://github.com/marin-community/marin/issues/860) [![#860](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/860)](https://github.com/marin-community/marin/issues/860)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA?accessToken=st1rajwy32etqi5rrlm3kuhgqa4ods6fnwsbyk8azjc8ar3eikf4dnz1p2ldz8yx)
- Tootsie 24B
    - [GitHub Issue #861](https://github.com/marin-community/marin/issues/861) [![#861](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/861)](https://github.com/marin-community/marin/issues/861)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA)
- Tootsie 70b
    - [GitHub Issue #750](https://github.com/marin-community/marin/issues/750) [![#750](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/750)](https://github.com/marin-community/marin/issues/750)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Big-Tootsies--VmlldzoxMTEyOTQ0MA)


## Modeling

- Pick Tokenizer type
    - [GitHub Issue #524](https://github.com/marin-community/marin/issues/524) [![#524](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/524)](https://github.com/marin-community/marin/issues/524)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Tokenizer-Comparison--VmlldzoxMDI0Njg3Nw)
    - Conclusion: Llama3 tokenizer is the best.
- Default z-loss?
    - [GitHub Issue #935](https://github.com/marin-community/marin/issues/935) [![#935](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/935)](https://github.com/marin-community/marin/issues/935)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/ZLoss-vs-Not-1-4B--VmlldzoxMjEzMzA1NA)
    - Conclusion: z-loss seems not harmful. We'll use it.
- Figuring out learning rate schedule!
    - [GitHub Issue #764](https://github.com/marin-community/marin/issues/764) [![#764](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/764)](https://github.com/marin-community/marin/issues/764)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin-optimizer/reports/Deciding-the-optimal-lr-schedule-which-is-cosine---VmlldzoxMTIxNDk5NA)
    - Conclusion: Cosine is best. High LR is important. WSD isn't terrible.
- Mixture of Experts
  - [GitHub Issue #929](https://github.com/marin-community/marin/issues/929) [![#929](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/929)](https://github.com/marin-community/marin/issues/929)
  - [WandB Report](https://api.wandb.ai/links/stanford-mercury/0lspgzn3)
- Hybrid Norm and Input Embedding Norm
  - [GitHub Issue #961](https://github.com/marin-community/marin/issues/961) [![#961](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/961)](https://github.com/marin-community/marin/issues/961)
  - [WandB Report](https://wandb.ai/stanford-mercury/hybrid-norm/reports/Hybrid-Norm--VmlldzoxMjY2MDgxMA)

## Training and Performance

- INT8 training in Levanter
    - [GitHub Issue #620](https://github.com/marin-community/marin/issues/620) [![#620](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/620)](https://github.com/marin-community/marin/issues/620)
    - [WandB Report](https://api.wandb.ai/links/stanford-mercury/yhrb0xik)
    - Conclusion: Int8 training is much faster on the right hardware, but might lead to worse performance in terms of time-to-loss except in the early stages.
- MuP for scaling laws
    - [GitHub Issue #621](https://github.com/marin-community/marin/issues/621) [![#621](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/621)](https://github.com/marin-community/marin/issues/621)
    - [WandB Report](https://api.wandb.ai/links/stanford-mercury/h723u2ws)
    - Conclusion: not worth it compared to our heuristic version.
- Figuring out learning rate schedule!
    - [GitHub Issue #764](https://github.com/marin-community/marin/issues/764) [![#764](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/764)](https://github.com/marin-community/marin/issues/764)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin-optimizer/reports/Deciding-the-optimal-lr-schedule-which-is-cosine---VmlldzoxMTIxNDk5NA)
- Try out different remat strategies to get the 70b working on fewer slices
    - [GitHub Issue #906](https://github.com/marin-community/marin/issues/906) [![#906](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/906)](https://github.com/marin-community/marin/issues/906)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Remat-Strategies--VmlldzoxMTkxNzk3Ng)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/expXXX_fancier_checkpointing-453042.json)
    - Conclusion: Substantial performance hit but helpful. Still need to iterate.

## Data Experiments

### High Quality Data Ablations

- Ablations on Cooldown for Markdownified Wikipedia
    - [GitHub Issue #845](https://github.com/marin-community/marin/issues/845) [![#845](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/845)](https://github.com/marin-community/marin/issues/845)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/845-6-Wiki-and-Arxiv-Quality-Ablations--VmlldzoxMTg4MzY2OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-custom-fork-2569de%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%5D)
    - Conclusion: No major improvement compared to control.
- Ablations on Cooldown for Markdownified Arxiv
    - [GitHub Issue #846](https://github.com/marin-community/marin/issues/846) [![#846](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/846)](https://github.com/marin-community/marin/issues/846)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/845-6-Wiki-and-Arxiv-Quality-Ablations--VmlldzoxMTg4MzY2OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Far5iv%2Far5iv-04-2024-no-problem-3971ff%2Fresiliparse-custom-fork%2F0001.jsonl.gz%22%5D)
    - Conclusion: No major improvement compared to control.
- Ablations on Cooldown for Markdownified StackExchange
    - [GitHub Issue #847](https://github.com/marin-community/marin/issues/847) [![#847](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/847)](https://github.com/marin-community/marin/issues/847)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/845-6-Wiki-and-Arxiv-Quality-Ablations--VmlldzoxMTg4MzY2OA)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fstackexchange-resiliparse-custom-fork-ab41ad%2F3dprinting.jsonl.gz%22%5D)
    - Conclusion: No major improvement compared to control.
- Mixture of Formats Training on Wikipedia and Arxiv
    - [GitHub Issue #818](https://github.com/marin-community/marin/issues/818) [![#818](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/818)](https://github.com/marin-community/marin/issues/818)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/818-Mixture-of-Formats--VmlldzoxMTg4MzU0NA)
    - Conclusion: No major difference observed, switch to @Helw150's annealing setup for evaluations.
- High Quality Many Epochs vs. Low Quality Few Epochs
    - [GitHub Issue #636](https://github.com/marin-community/marin/issues/636) [![#636](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/636)](https://github.com/marin-community/marin/issues/636)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/High-Quality-Many-Epochs-vs-Lower-quality-fewer-epoch--VmlldzoxMDU2MTI1Mg)
    - Conclusion: There's no data like more data.

### Data Filtering

- Stack Exchange Quality Classifier
    - [GitHub Issue #596](https://github.com/marin-community/marin/issues/596) [![#596](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/596)](https://github.com/marin-community/marin/issues/596)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/Quality-Classifier-Comparison--VmlldzoxMDI2MzI1MQ)
    - Conclusion: Seems to lead to better loss than using Reddit ELI5 or OpenHermes.
    - NOTE: this seems like a loose end, we should pursue this further.

### Text Extraction and Formatting

- Compare HTML -> text methods
    - [GitHub Issue #246](https://github.com/marin-community/marin/issues/246) [![#246](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/246)](https://github.com/marin-community/marin/issues/246)
    - [WandB Report](https://api.wandb.ai/links/stanford-mercury/0uoys8gp)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/experiment/?path=gs%3A//marin-us-central2/experiments/246_web_extraction_method_training-efe0cf.json)
    - Conclusion: some amount of format preservation is helpful for loss on Paloma.
- Wikipedia Training Runs with DOLMA source substitution
    - [GitHub Issue #647](https://github.com/marin-community/marin/issues/647) [![#647](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/647)](https://github.com/marin-community/marin/issues/647)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/647-Wikipedia-Training-Runs-with-DOLMA-source-substitution--VmlldzoxMDkyNjIxNw)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-no-references-no-links-cleaned-7971bb%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%2C%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-no-references-with-links-cleaned-b89dd3%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%2C%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-with-references-no-links-cleaned-0fd095%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%2C%22gs%3A%2F%2Fmarin-us-central2%2Fdocuments%2Fwikipedia-resiliparse-with-preserving-formatting-with-references-with-links-cleaned-infobox-0203ff%2F20241201%2Fenwiki_namespace_0_0.jsonl.gz%22%5D)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fexperiments%2Fexp647_wikipedia_training-dd35e8.json%22%5D)
- Ar5iv Training Runs with DOLMA source substitution
    - [GitHub Issue #648](https://github.com/marin-community/marin/issues/648) [![#648](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/648)](https://github.com/marin-community/marin/issues/648)
    - [Data Browser](https://crfm.stanford.edu/marin/data_browser/view?paths=%5B%22gs%3A%2F%2Fmarin-us-central2%2Fexperiments%2Fexp648_ar5iv_training-f49d7f.json%22%5D)

## Supervised Fine Tuning

- Reproduce olmov2 SFT
    - [GitHub Issue #606](https://github.com/marin-community/marin/issues/606) [![#606](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/606)](https://github.com/marin-community/marin/issues/606)
    - [WandB Run: marin_olmo_tulu_sft_v3-acca67](https://wandb.ai/stanford-mercury/marin/runs/marin_olmo_tulu_sft_v3-acca67)
- Create Mixture from All SFT Datasets
    - [GitHub Issue #804](https://github.com/marin-community/marin/issues/804) [![#804](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/804)](https://github.com/marin-community/marin/issues/804)
    - [WandB Report](https://api.wandb.ai/links/stanford-mercury/m8ak7uah)
- SFT on further cool-downed tootsie checkpoints
    - [GitHub Issue #897](https://github.com/marin-community/marin/issues/897) [![#897](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/897)](https://github.com/marin-community/marin/issues/897)

## Scaling Laws

- Scaling laws to predict tootsie performance
    - [GitHub Issue #654](https://github.com/marin-community/marin/issues/654) [![#654](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/654)](https://github.com/marin-community/marin/issues/654)
    - [WandB Report](https://api.wandb.ai/links/stanford-mercury/3xojrl9v)
    - [WandB Report on Soft Metrics](https://api.wandb.ai/links/stanford-mercury/got35r4i)
- Optimizer Scaling Law Part 1: AdamW
    - [GitHub Issue #725](https://github.com/marin-community/marin/issues/725) [![#725](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/725)](https://github.com/marin-community/marin/issues/725)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin-optimizer/reports/AdamW-Sweeping--VmlldzoxMTE3Nzc5OA)
    - Conclusion: After sweeping, we discovered that the (near) optimal set of hyperparameters for AdamW remains surprisingly stable across three settings.

## Baselines and Reproductions

- Train a simple Dolma/Olmo baseline to flex the pipeline
    - [GitHub Issue #442](https://github.com/marin-community/marin/issues/442) [![#442](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/442)](https://github.com/marin-community/marin/issues/442)
    - [WandB Report](https://api.wandb.ai/links/stanford-mercury/e20j5423)
- Build DCLM 7b baseline
    - [GitHub Issue #143](https://github.com/marin-community/marin/issues/143) [![#143](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/143)](https://github.com/marin-community/marin/issues/143)
    - [WandB Report](https://wandb.ai/stanford-mercury/marin/reports/DCLM-7B-Replication--Vmlldzo5MTA3NjU5/edit)

## Other Projects

### Compel
- Compression-Ratio Quality Filter 1.4B Models
    - [GitHub Issue #633](https://github.com/marin-community/marin/issues/633) [![#633](https://img.shields.io/github/issues/detail/title/stanford-crfm/marin/633)](https://github.com/marin-community/marin/issues/633)
    - [WandB Run: compel-fineweb-edu-baseline](https://wandb.ai/stanford-mercury/marin/runs/compression-fineweb-edu-0.6-0.8-da53bf?nw=nwusereobbad)
    - [WandB Run: compel-fineweb-edu-0.65-0.8](https://wandb.ai/stanford-mercury/marin/runs/compression-ratio-filter-fineweb-edu-786908?nw=nwusereobbad)
    - [WandB Run: compel-fineweb-baseline](https://wandb.ai/stanford-mercury/marin/runs/compression-train-full-dataset-llama1.4b-20fa75?nw=nwusereobbad)
    - [WandB Run: compel-fineweb-0.65-0.8-e5cdae](https://wandb.ai/stanford-mercury/marin/runs/compression-ratio-filter-llama1.4b-0.6-0.8-e5cdae?nw=nwusereobbad)


## Uncategorized

(This is for experiments that have been added via the script but have not yet been curated.)
