# Review record and disposition

Two independent reviews of this brief, both with `gh` and `git` access so they
could check the cited sources themselves rather than take the brief's word.

- [`codex.md`](codex.md) — codex-cli, reviewing commit `ff39ed3bb`.
- [`opencode.md`](opencode.md) — opencode running kimi-k3, reviewing commit
  `ff39ed3bb`.

Both reviewed the original commit. Corrections from a parallel research pass had
already landed as `04402aab8` before either review returned, so a few findings
arrived already fixed; those are marked below.

Every finding was checked against its cited source or commit before being applied.
**Nothing was accepted on the reviewer's assertion alone**, and nothing was
refuted — all findings from both reviews held up.

## codex — 9 findings, all accepted

Applied in `4ea9f3f53`.

| # | Severity | Finding | Disposition |
|--:|---|---|---|
| 1 | blocking | The expert-only MXFP8 port *was* tested at EP64 and lost −2.582pp p50 in a matched QB-on drop-reported A/B; the brief generalised a narrower unknown about the hybrid recipe into "never measured" | **Accepted.** Verified against `24d411b38` directly. Added evidence card G1b and table row E1d; rewrote Tier 5, the G5 dependency, and derisking D-9/D-10. |
| 2 | blocking | Leg-batching *was* composed with QB-on and regressed −3.66pp; the original 25.39% patch is uncommitted | **Accepted.** Partly fixed already in `04402aab8`; added the result SHA `081450952f` and the uncommitted-patch fact, and rewrote D-3 to start with code recovery. |
| 3 | blocking | Sender-local QB and both integral arms were already tested and negative; the damped arm is *unavailable*, not measured | **Accepted.** Partly fixed already in `04402aab8`; added SHAs `6ac4bbeee` and `a48a8a9e3` and corrected the damped arm's status in both files. |
| 4 | blocking | The 24.153% figure is the treatment arm of the padded-Muon A/B — both arms ran ECHO — so it measures the optimizer change, not the transport | **Accepted.** This was a structural error. Receiver-ECHO is no longer ranked on it; the #7670 matched isolation (drops 1.32% → 0.02% for ~1pp) is now the basis, and item 13 gained its dependency on the QuACK substrate. |
| 5 | major | The −0.58pp-per-+0.05 capacity price was retracted in the source; the real curve is a 1.179pp cliff then flat | **Accepted.** Removed the linear estimate; the measured points are now used directly. |
| 6 | major | Two shared experts (+0.29pp) and Muon shape-grouping (+0.09pp) are unreplicated single screens, below the ~2pp threshold this brief's own protocol sets | **Accepted.** Labelled unreplicated and removed from the commit series. |
| 7 | major | Dependency order violated twice — `sonic_cute` was sequenced before the QuACK dependency it imports, and Receiver-ECHO before the substrate its kernel path uses | **Accepted.** Phase B reordered; item 13's dependency list corrected. |
| 8 | major | "4-of-256 at EP64 does not fit one rack" is false as stated — that is the d6144/i3072 candidate; the d5120/i2048 arm completed on one rack | **Accepted.** Qualified everywhere it appears. |
| 9 | minor | Branch edge counts wrong | **Accepted after independent check.** Measured from the shared-base tip `9bf2ee02e` (which is what the diagram labels): +17, +19, +2, not +13, +21, +1. |

## opencode — see [`opencode.md`](opencode.md)

Disposition table below, filled in on the same terms: verify first, then apply or
refute with a reason.
