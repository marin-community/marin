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
**Nothing was accepted on the reviewer's assertion alone.** 22 of 23 findings held
up in full; one (opencode #5) was half wrong and is refuted below with the source
text it claimed did not exist.

Between them the two reviews found five blocking errors, two of which were
structural: a headline number credited to the wrong change, and an extraction plan
pointing at a stale proof-of-concept rather than the validated code. The reviews
also caught three places where this document violated its own measurement protocol.

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

## opencode (kimi-k3) — 14 findings, 13 accepted, 1 accepted in part

Applied in `4947231fc`. Every finding was checked against the source before being
applied; **one was partly wrong and is refuted below with evidence.**

| # | Severity | Finding | Disposition |
|--:|---|---|---|
| 1 | blocking | The `sonic_cute` extraction points at `5cf76b64a`, whose `sonic_cute.py` is 105 lines with no `quack_symmetric_cute.py`; the byte-identical substrate is the 272-line branch-tip version | **Accepted.** Verified by `git ls-tree`: `5cf76b64a` carries blob `c747e6d2ce`, both branch tips carry `4d53627060` plus `628f77fdb2`. Phase B2 re-pointed at the tip file set (~560 lines, not 309); the byte-identity argument in README §4 now says *at the tips*. |
| 2 | major | v143 is labelled 8-of-256 but is a **top-4** run, and the "beats C2's compliant point" claim crosses architectures | **Accepted.** Verified against the logbook config line ("L48, d5120, E256, top-4, routed intermediate 2,048"). Relabelled in both files; the cross-arm comparison is deleted and replaced with an explicit warning. |
| 3 | major | A compliant **25.501% @ 2.024%, 200-step** run exists and is omitted, contradicting the "22–24%" band | **Accepted.** Verified in the logbook (`e6124dac2`, run `…sh21504-pad2-qb200-v119…`). Added to the §1 table with its architecture caveat — it widens the shared expert 5,120 → 21,504, and its own entry seals it as a capacity option, not a kernel result. Band claim now scoped to the production-candidate architectures. |
| 4 | major | E7 inverts the source: the shim does **not** fix the TE-at-tip regression | **Accepted.** Verified verbatim in [#7331 c5076118699](https://github.com/marin-community/marin/issues/7331#issuecomment-5076118699). Rewritten; the unsourced "~17% vs 18.05%" removed. |
| 5 | major | The "~8% relative GEMM error vs bf16's 0.5%" claim "appears nowhere in #7332", and the 18.2%/417K figures are cited to a bf16-only comment | **Accepted in part — the first half is wrong.** See the refutation below. The citation half is correct and is fixed. |
| 6 | major | B8's `#7279 c5028967210` citation does not mention `107476c8d` or QuACK-under-ring | **Accepted.** Dropped; the ring → `ring_cute` pair is cited to #7012 c4994519151, with the commit→result link attributed to PR #7490's inventory. |
| 7 | minor | The E22 tile seal (+0.14pp) conflicts with the ECHO line's +0.861pp keep decision | **Accepted, and it exposed a second error of mine.** B9 had attributed +0.861pp to a grouped weight-gradient feature; the handoff table shows v134 is the *tile* change (`…-qb-quack256-v134-…`). B9 rewritten and the cross-stack conflict recorded on both cards. |
| 8 | minor | Branch edge counts wrong; two FP8 refs are local-only and unflagged | **Accepted.** `mcwitt/moe-standalone-ep` corrected 25 → 26 (the other three were already fixed after the codex pass). Durability note extended to `224a0081` and `0a37854`, neither of which is on an origin branch. |
| 9 | minor | Eight citations point at the wrong comment (facts check out) | **Accepted.** All repointed. |
| 10 | minor | Four diff-size and line-number claims don't match | **Accepted.** Verified by `numstat`: grugmuon hunk +37/−4 (not +33), `b0c7a1b56` +232/−20 total, `main`'s `ragged_all_to_all` calls at `:77`/`:105` of a 124-line file, ECHO branch +10,785/−785 against merge-base. |
| 11 | minor | G2's "+0.111%" Paloma disagrees with the source and with the brief's own other two mentions | **Accepted.** Arithmetic confirms +0.1105%. Changed to +0.110%. |
| 12 | minor | "none of … the QuACK dependency" is wrong — `quack-kernels` 0.5.0 is on `main` transitively | **Accepted.** Verified in `origin/main:uv.lock` and the `from quack import layout_utils` in an on-main file. Reworded to name the missing *pinned* dependency. |
| 13 | minor | "for weeks", "exit 134", and a 512-expert profile detail are unsourced or overstated | **Accepted.** The wheel-shadowing timeline corrected to its actual ~2 days; "exit 134" and the profile parenthetical dropped. (The separate "for weeks" in A6 describes the CUBIN loader bug and is supported — left as is.) |
| 14 | minor | Omits the ECHO-line PGLE screens (v152/v153), the 200-step v128 baseline, and `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` | **Accepted.** v128 added to the §1 table and C3; v152/v153 added to D-6, which narrows the PGLE-under-EP question rather than leaving it open; the allocator added to the derisking protocol as rule 9. |

### Refutation of finding 5, first half

opencode wrote that "~8% relative GEMM error vs bf16's 0.5%" *"appears **nowhere**
in #7332 (body or any comment)"* and called it "an unsourced, precise-looking
numerics claim". It is sourced. The [#7332](https://github.com/marin-community/marin/issues/7332)
issue body contains, verbatim:

> a pod microbench puts our e4m3 GEMM at ~8% relative error vs bf16's ~0.5%
> (fp8-class, comparable to #7282's MXFP8 ~6.6%, but uncalibrated)

The reviewer's full-text search missed it. The claim is kept, now with the extra
context that it is comparable to MXFP8's ~6.6% and uncalibrated.

The **second half of the finding is correct and was applied**: the 18.2% / 417K
figures and the error numbers live in the issue *body*, while the brief cited
comment 5007041962, which is explicitly bf16-only. That citation is fixed.
