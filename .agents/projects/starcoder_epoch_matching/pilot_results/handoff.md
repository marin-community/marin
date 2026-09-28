# Epoch-matching pilot results

All ten training runs completed successfully. The parent `/calvinxu/starcoder-epoch-matching-pilot` finished at 21:05:30 PDT on 8 September 2026. The matched p=0.7 run recovered from one worker failure counted by Iris as a preemption; all final artifacts succeeded and charged failure counts are zero.

**The pilot shows a strong effect of repetition, but it does not demonstrate improved target-mixture selection from epoch matching.** The unmatched proxy selects pure StarCoder; the matched proxy has an interior minimum but selects too little StarCoder relative to the target on the measured grid.

| Selection | Chosen StarCoder fraction | Target BPB at that fraction | Regret on the pilot target grid |
|---|---:|---:|---:|
| Target grid minimum | 0.70 | 0.788043 | 0 |
| Unmatched proxy | 1.00 | 0.819869 | 0.031826 |
| Epoch-matched proxy | 0.30 | 0.821393 | 0.033350 |

Matched-minus-unmatched target regret is **+0.001523 BPB**. This is a descriptive difference under the fixed target measurements. There is no repeat-based uncertainty estimate or statistical superiority test. Absolute distance from the target's selected fraction is 0.30 for unmatched and 0.40 for matched.

## Measured curves

The five-point grid was fixed before launch: p=0, 0.1, 0.3, 0.7, 1.0. Lines connect observed points; no fitted minimum is used. The web-only proxy is shared by both labels. The target's old p=1 outcome is excluded and replaced by the new run using the verified interior parent corpus.

| StarCoder fraction | Unmatched proxy BPB | Matched proxy BPB | Target BPB |
|---|---:|---:|---:|
| 0.00 | 2.014069 | 2.014069 | 1.573216 |
| 0.10 | 1.204831 | 1.221240 | 0.913296 |
| 0.30 | 1.052185 | 1.190170 | 0.821393 |
| 0.70 | 0.941884 | 1.763008 | 0.788043 |
| 1.00 | 0.908917 | 2.592288 | 0.819869 |

The unmatched curve decreases through p=1. The matched curve rises from 1.19017 BPB at p=0.3 to 2.59229 at p=1, whereas the target rises only from 0.78804 at p=0.7 to 0.81987 at p=1. Epoch matching therefore introduces turnover but substantially overstates the high-repetition penalty relative to the target in this setting.

[Raw curves](curves.pdf) and [curves above each observed grid minimum](excess_curves.pdf) are available as PDF and PNG. Both renderings were visually checked. The normalized plot preserves loss differences within each curve; it does not calibrate proxy predictions to the target.

## Interpretation and next decision

The comparison keeps model architecture fixed but leaves a large training-horizon difference: 277.873M proxy tokens versus 7.408B target tokens. The matched proxy uses only 10.486M StarCoder tokens, nested in the target's 279.970M-token parent. Matching nominal epochs does not remove differences in unique data or training horizon. An unrepresentative fixed subset and stronger repetition harm at the shorter horizon are possible explanations; this pilot does not identify their separate contributions.

The measurements are conditional on one trainer/data seed and one fixed matched subset. Historical target interior runs used JAX 0.10.1; the new proxies and corrected p=1 target used 0.11.1. Index identity was verified across runtimes, but numerical training equivalence was not established. In particular, the 0.001523-BPB comparison between the two selected target points spans historical and new runs and should not be treated as a statistically resolved loss difference.

Retain this result as evidence that matching epochs alone was insufficient in this very short proxy regime. Do not replace Figure 2A with a claim that this pilot recovered the target optimum. The primary and replicated stages remain unsubmitted. A denser grid could locate the discrepancy more precisely, but should not be presumed to repair it. Before expanding the full matrix, discuss a separately specified subset-replication diagnostic or the previously identified fallback with a larger finite parent and a new target, allowing a longer nonrepeating proxy. Simply lengthening this unmatched proxy would violate its no-repetition condition.

## Verification and provenance

The collector verified successful durable artifacts, matching configuration fingerprints and the exact endpoint metric `eval/paloma/dolma_100_programing_languages-llama3/bpb`: step 1059 for all nine proxies and 28259 for the corrected target endpoint. All ten rows match the frozen design and saved submission plan. An independent calculation from the CSV and frozen JSON reproduced every minimum, chosen target loss and regret exactly.

The actual runtime metadata from the web-only child confirms JAX/JAXlib 0.11.1, NumPy 2.3.5 and libtpu 0.0.46. That is direct runtime evidence for this child; all children were submitted with the same locked TPU environment.

- [Machine-readable analysis](analysis.json)
- [Curves and source identities](curves.csv)
- [Verified new measurements](../pilot_measurements.csv)
- [Durable submission plan](../pilot_submission_plan.json)
- [Completion snapshot, mappings and attempt history](../pilot_completion_snapshot.json)
- [Full protocol and commands](../review_packet.md)

Fieldbook experiment: `exp_01m21p4aw15bpvhnswz8gtwn0d`. No new jobs were submitted during collection or analysis.
