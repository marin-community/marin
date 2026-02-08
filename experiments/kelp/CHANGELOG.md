# Changelog

All notable changes to the Kelp experiment will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Write report on initial Kelp tree diffusion findings with scaling law predictions (#47)
- Evaluate best checkpoint and file improvement research issues (#42)
- Run 5-hour training run with augmented subtree bank and checkpoint saving (#41)
- Research programmatic subtree augmentation for the subtree bank (#39)
- Run end-to-end training and evaluation loop on laptop to validate tree diffusion pipeline (#38)
- Add execution-guided reranking using test feedback (#36)
- Add beam search with scoring for tree diffusion inference (#35)
- Add grammar-constrained decoding using incremental Python parsing (#33)
- Add position token vocabulary for referencing AST nodes in token sequences (#31)
- Add TreeDiff algorithm for computing edit paths between Python ASTs (#30)
- Add AST-based forward process that corrupts programs via subtree replacement (#29)
- Add AST subtree bank that indexes real Python code fragments by node type (#28)
- Improve kelp experiment by reusing Marin/Levanter ML infrastructure (#10)
- Add gradient checkpointing for large model training (#16)
- Use Levanter AdamConfig for robust optimizer with skip-bad-steps and weight decay masking (#15)
- Add W&B experiment tracking via Levanter tracker (#14)
- Add bf16 mixed precision training support (#13)
- Add padding-aware attention masking using Grug AttentionMask (#12)
- Replace hand-rolled transformer primitives with Grug building blocks (#11)
- Add pad_token_id to TreeDiffusionConfig instead of hardcoding 0 (#9)

### Changed
- Consolidate dead code: remove unused modules, extract shared corpus and checkpointing (#48)
- Update training pipeline for tree diffusion with TreeDiff supervision (#34)
- Replace bidirectional parallel prediction with causal AR edit prediction (#32)

### Removed
- Remove dead modules: data/, eval/, transfer/, tokenizer.py, tree/parser.py (#48)
- Remove unused SIZE_PRESETS and factory functions from model/config.py (#48)
- Remove D3PM noise process and parallel prediction code (#37)
- Remove dead code: freeze_embeddings and freeze_attention config options (#8)

### Fixed
- Fix unused tree/ module - model operates on flat tokens, not ASTs (#3)
- Fix fsspec filesystem detection that breaks for local paths (#7)
- Fix loss computed on padding tokens instead of only real content (#2)
- Fix SimpleTokenizer duplication across four modules (#1)
- Fix dropout_rate config field that is declared but never applied (#5)
- Fix attention scaling bug - einsum contraction indices are wrong (#6)

### Security
- Fix pickle-based checkpointing - insecure and fragile (#4)
