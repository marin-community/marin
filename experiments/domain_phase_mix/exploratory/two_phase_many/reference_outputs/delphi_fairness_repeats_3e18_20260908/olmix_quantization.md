# Olmix repeat mixtures

The runtime mixture of a continuous-weight run is Levanter's per-block quantization (`MixtureDataset._compute_expected_counts_per_block`: weight x 2048 truncated to an integer, the remainder added to the largest bucket). The repeats train exactly those runtime counts; the three `lwspu_*` rows are copied verbatim from the frozen-procedure validation table.

- olmix_u_kl0p1_cap04: trained run olmix_onephase_uncheatable_d001_kl0p1_cap4_3e18-464fd1 (measured 1.0022, mixture sha256 806581e84149aa8bd30519aa815e92374c29e5c79fd3e54a681b83b49266f571); Levanter block quantization of its weights (truncation to 2048 blocks, remainder of 22 blocks to dolmino_common_crawl_hq); TV to the continuous weights 0.0103; runtime max epoch 3.9908; zero-count buckets 2
- olmix_t9_kl0p005_cap05: trained run olmix_onephase_table9_d001_kl0p005_cap4_3e18-eff7f7 (measured 1.0769, mixture sha256 fc3d40c42d08895b01a28f5aac104f9c4dcb39456af36d986aebfca015ac6807); Levanter block quantization of its weights (truncation to 2048 blocks, remainder of 17 blocks to dolmino_synth_qa); TV to the continuous weights 0.0079; runtime max epoch 4.0953; zero-count buckets 14

Candidate ids end with the whole-run epoch cap the launcher checks: `cap04` for the Uncheatable policy (runtime max 3.99 epochs) and `cap05` for the Table-9 policy, whose runtime mixture reaches 4.10 epochs on dolmino_synth_qa because Levanter's remainder rule adds the truncated blocks to the largest bucket. Both were proposed by Olmix under its cap of 4.
