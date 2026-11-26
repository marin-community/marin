# Snowball model family tree

[Open the Snowball model family tree](https://public.applets.marina.oa.dev/a/d3b5517a-c119-4fd9-999d-362183422bf3/)
to explore 144 model and run nodes across the 2.7T, 5.7T and 10T cooldown families
without signing in. This is a curated snapshot of the evidence available on
November 4, 2026. The stable URL opens the current app revision.

The lineage map opens with the whole tree visible and its shared pretrain centered.
Click a node to expand its summary, algorithms, compute estimate and evidence.
Wheel or trackpad input zooms; dragging pans. The side panel has zoom controls,
ancestry, direct children and **Focus this branch**. **Read all** presents the
summaries as cards. Search includes model repository IDs and GitHub issue numbers.

**Timeline** defaults to the 10T family and a September 7 opening date. The cooldown
stays on the left while descendants appear in chronological columns. Switch to
2.7T or 5.7T with the timeline's cooldown selector. Space starts or pauses playback
outside editable controls. Normal playback takes 12 seconds; Fast takes 6 and Slow
24. The date scrubber and **Show all** allow direct navigation. Click a timeline
node for its date basis and full evidence.

Organization color records experiment ownership. The complete 2.7T branch belongs
to open-athena, including exports hosted under other Hub namespaces. Dashed links
identify inferred ancestry. Timeline dates can identify a checkpoint label, run
name, source report or Hub repository creation; they do not all identify training
completion. Undated nodes appear at the end. Column spacing follows chronology
without encoding elapsed duration.

The graph includes [the broad-domain RLVR campaign](https://github.com/marin-community/marin/issues/9735),
including the Antidoom step-20 checkpoint trained with a frozen-reference KL
penalty from public Antidoom step 12. The step-45 export's exact earlier weight
ancestry remains qualified. Compute labels identify the estimate's scope; unknown
training and rollout totals remain unknown.

The October 12 snapshot adds the [weight-merging campaigns](https://github.com/marin-community/marin/issues/9975)
(serial expert transfers and the campaign 2/3 DARE merges, each with one arrow
from every pinned parent), the [BFCL recovery negative result](https://github.com/marin-community/marin/issues/10040)
and the [protocol FTPO repair](https://github.com/marin-community/marin/issues/10043).
The campaign 3 repository originally inherited the campaign 2 model card and
provenance byte-for-byte; both were corrected on the Hub on October 12, 2026.

The October 19 update adds the Smartie RL branch from campaign-2 DARE-TIES-02:
[`.32` step 14](https://huggingface.co/open-athena/Snowball-Smartie-RL-2026.10.15.32-step14#provenance-report),
the internal `.45` step-2 and `.47` step-8 resume checkpoints, and the public
[`.49` step 12](https://huggingface.co/open-athena/Snowball-Smartie-RL-2026.10.17.49-step12).
The first run starts with a fresh optimizer; the continuation starts another
fresh optimizer from `.32` step 14, then retains optimizer and step state across
`.45`, `.47` and `.49`. The continuation counter excludes the initializer's
14 updates. `.32` step 14 is the best surviving native-development checkpoint;
`.49` step 12 is a user-selected export with incomplete evaluation, not a
complete-evaluation winner. Public model cards, pinned source configurations,
W&B links and the complete recorded `.32` W&B archive support these nodes.

The October 23 update adds the transfer experiments built from the sync Step92
student. [The single-expert experiment](https://github.com/marin-community/marin/issues/9920)
adds the CruxEval expert E and its plain-SFT, rejection-sampled SFT and on-policy
distillation transfers back into Step92. [The two-expert experiment](https://github.com/marin-community/marin/issues/9984)
adds the MMLU-Pro expert E₂, the mixed plain-SFT, mixed RS-SFT and multi-teacher
on-policy distillation arms, and the task-arithmetic (a = b = 0.5) and TIES
(d = 0.5) merges. Each merge has an arrow from Step92, E and E₂. Distillation
teachers appear in node summaries, not as weight parents. Each node pins the
weight revision named in its experiment record; later Hub commits changed only
model cards.

The November 4 update adds the [identity SFT candidates](https://github.com/marin-community/marin/issues/10695)
trained from `.32` step 14 (Marin Snowball v0.1) on identity data mixed with replay
of v0.1's own answers: the lead candidate A13, the A8a recipe it builds on, the
fallback A3, and the merge M25 (0.75 A3 + 0.25 A8a), which has an arrow from each
of its two parents. Each node pins its Hugging Face weight revision; these are
research checkpoints, not releases.

## Maintain the app

The source package is `infra/marina/apps/snowball_family_tree/`.
Follow the [contributor README](https://github.com/marin-community/marin/blob/main/infra/marina/apps/snowball_family_tree/README.md)
to add a checkpoint to `graph-data.json`, verify its parent and timeline, and
submit a reviewed Marin PR. `graph-data.json` on `main` is the source of truth;
every graph change goes through a PR. The extracts
`evidence/evidence-glm53async48-telemetry.json` and
`evidence/evidence-antidoom12-telemetry.json` support scoped RL compute estimates
for their respective runs. Update or add an extract when new token accounting
changes an estimate, and link it from the node's evidence. Run records link to the
public campaign archive.

### Validate and release

Build and validate from the repository root:

```bash
uv run marina validate infra/marina/apps/snowball_family_tree
uv run marina release infra/marina/apps/snowball_family_tree --dry-run --json
uv run marina release infra/marina/apps/snowball_family_tree --local --json
```

The local command starts a disposable Marina/Postgres instance and prints its
revision URL. Ctrl-C removes it. The manifest declares the existing app UUID.
For production updates, release only from a clean checkout at current remote main.
Inspect the release status and live revision before publishing:

```bash
uv run marina release-status infra/marina/apps/snowball_family_tree --json
uv run marina apps versions d3b5517a-c119-4fd9-999d-362183422bf3 --json
uv run marina release infra/marina/apps/snowball_family_tree \
  --base-version CURRENT_VERSION --json
```

Replace `CURRENT_VERSION` with the inspected current version. Release preserves
the app's public mode and records Git source provenance. The manifest's build
command validates graph lineage and produces `dist/` from the HTML, JavaScript,
CSS, graph and token-count extracts. Release packages that output and restores
previous local build output afterward. Generated files are ignored by Git. Asset
URLs are relative, and executable scripts use packaged files to satisfy Marina's
content security policy.

Do not use `--allow-dirty` for a public tree release. Merging source or
deploying Marina does not publish the app. If the live package differs from
main, record the difference in the tree PR or tracking issue before publishing.
After a manual release, record the published version, revision URL and main
commit there.

### Nightly publication

[Marin - Release Snowball Tree](https://github.com/marin-community/marin/actions/workflows/marin-release-snowball-tree.yaml)
runs daily at 04:17 UTC when the repository variable `SNOWBALL_TREE_RELEASE` is
`enabled`. Manual dispatch from `main` works without the variable. The job skips
the release when the live revision already matches main. Otherwise it runs the
layout check, releases against the observed live version and checks that the
new revision is live. It does not retry a version conflict; inspect the new
state, then rerun. The run summary and the `snowball-release-<run>-<attempt>`
artifact hold the release record, also when verification fails.

To activate it:

1. An IAM operator applies the `marin` stack, which grants
   `serviceAccount:marin-cd-cloud-run-deploy@hai-gcp-models.iam.gserviceaccount.com`
   `roles/iap.httpsResourceAccessor` on Marina. CI only previews this stack.
   Follow the [IAM review procedure](https://github.com/marin-community/marin/blob/main/infra/pulumi/README.md#user-grants)
   and the [Cloud Run IAP access check](https://github.com/marin-community/marin/blob/main/infra/pulumi/README.md#cloud-run-iap-access):

   ```bash
   pulumi -C infra/pulumi preview --stack marin
   pulumi -C infra/pulumi up --stack marin
   ```

2. Dispatch a dry run and inspect it:

   ```bash
   gh workflow run marin-release-snowball-tree.yaml --ref main -f dry_run=true
   ```

3. If the dry run stops on an unrecorded live package (a release without source
   provenance), compare it with main and record the differences in the tree PR
   or tracking issue. Then preview and run a one-time bootstrap:

   ```bash
   gh workflow run marin-release-snowball-tree.yaml --ref main \
     -f dry_run=true -f bootstrap_unrecorded=true
   gh workflow run marin-release-snowball-tree.yaml --ref main \
     -f dry_run=false -f bootstrap_unrecorded=true
   ```

4. Enable the schedule. Delete the variable to pause it.

   ```bash
   gh variable set SNOWBALL_TREE_RELEASE --repo marin-community/marin --body enabled
   ```

Local dry run, with a synced Marina environment and an authenticated account:

```bash
uv run --no-sync python scripts/ci/snowball_release.py --dry-run --record /tmp/snowball-release.json
```
