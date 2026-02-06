# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Removed
- Remove dead code: freeze_embeddings and freeze_attention config options (#8)

### Added
- Add pad_token_id to TreeDiffusionConfig instead of hardcoding 0 (#9)

### Fixed
- Fix unused tree/ module - model operates on flat tokens, not ASTs (#3)
- Fix fsspec filesystem detection that breaks for local paths (#7)
- Fix loss computed on padding tokens instead of only real content (#2)
- Fix SimpleTokenizer duplication across four modules (#1)
- Fix dropout_rate config field that is declared but never applied (#5)
- Fix attention scaling bug - einsum contraction indices are wrong (#6)

### Changed
