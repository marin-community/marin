# Snowball model family tree

[Open the Snowball model family tree](https://public.applets.marina.oa.dev/a/d3b5517a-c119-4fd9-999d-362183422bf3/)
to explore 130 model and run nodes across the 2.7T, 5.7T and 10T cooldown families
without signing in. This is a curated snapshot of the evidence available on
October 19, 2026. The stable URL opens the current applet revision.

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

## Maintain the applet

The source package is `infra/marina/applets/snowball_family_tree/`.
Follow the [contributor README](https://github.com/marin-community/marin/blob/main/infra/marina/applets/snowball_family_tree/README.md)
to add a checkpoint to `graph-data.json`, verify its parent and timeline, and
submit a focused PR. The extracts
`evidence/evidence-glm53async48-telemetry.json` and
`evidence/evidence-antidoom12-telemetry.json` support scoped RL compute estimates
for their respective runs. Update or add an extract when new token accounting
changes an estimate, and link it from the node's evidence. Run records link to the
public campaign archive.

Build and validate from the repository root:

```bash
uv run marina validate infra/marina/applets/snowball_family_tree
uv run marina publish infra/marina/applets/snowball_family_tree --local --mode public
```

The local command starts a disposable Marina/Postgres instance and prints its
revision URL. Ctrl-C removes it. Before updating the public applet, inspect its
current revision:

```bash
uv run marina applets versions d3b5517a-c119-4fd9-999d-362183422bf3 --json
uv run marina publish infra/marina/applets/snowball_family_tree \
  --update d3b5517a-c119-4fd9-999d-362183422bf3 \
  --base-version CURRENT_VERSION --mode public
```

Replace `CURRENT_VERSION` with the reported current version. The manifest's build
command produces `dist/` from the HTML, JavaScript, CSS, graph and token-count
extracts. Publication uses that generated package. Generated files are ignored by
Git. Asset URLs are relative, and executable scripts use packaged files to satisfy
Marina's content security policy.
